"""Replay published predictions with physical-unit and degenerate-correlation reporting."""

import csv
import hashlib
import importlib.metadata
import io
import json
import resource
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import presto
from datacache import materialize
from presto.training import holdout_eval

FAMILY = Path(__file__).resolve().parents[1]
ROOT = FAMILY.parents[1]
BASE = "https://github.com/pirl-unc/presto/releases/download/effective-target-fit-qualification-v1/"
INPUTS = {
    "run.tar.gz": {
        "url": BASE + "run.tar.gz",
        "sha256": "134c5ce774ad473f7f5a61273cdad864f31d17e1428e5703405a3e5243448bc9",
        "size": 17475241,
    },
    "inventory.json": {
        "url": BASE + "inventory.json",
        "sha256": "8539de79868d541c53688c7af30e388f834042338bfdb5dd77a4b40abca88494",
        "size": 12810,
    },
}


def pin(path):
    path = Path(path)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": path.stat().st_size,
    }


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def independent_binary(raw, prediction, qualifier):
    usable = (qualifier == 0) | ((qualifier < 0) & (raw <= 500)) | ((qualifier > 0) & (raw > 500))
    labels, predicted = raw[usable] <= 500, prediction[usable] <= np.log10(500)
    tp, tn = int((labels & predicted).sum()), int((~labels & ~predicted).sum())
    fp, fn = int((~labels & predicted).sum()), int((labels & ~predicted).sum())
    return {
        "n_usable_observations": int(usable.sum()),
        "n_positive_observations": int(labels.sum()),
        "accuracy": (tp + tn) / len(labels),
        "balanced_accuracy": 0.5 * (tp / (tp + fn) + tn / (tn + fp)),
    }


def main():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True), (
        "Freeze a clean recipe before this audit"
    )
    assert presto.__version__ == "0.1.18"
    assert Path(holdout_eval.__file__).resolve() == ROOT / "training/holdout_eval.py"
    metadata = {
        "agent": "codex",
        "model": "GPT-6",
        "commit": commit,
        "dirty": False,
        "auditor_version": presto.__version__,
        "recipe": pin(__file__),
        "metric_implementation": pin(holdout_eval.__file__),
        "inputs": INPUTS,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "datacache", "torch")
        },
        "execution": "CPU data-only replay; no model fitting or new predictions",
    }

    def build(inputs, outputs):
        started = time.monotonic()
        inventory = json.loads(Path(inputs["inventory.json"]).read_text())
        files = {}
        with tarfile.open(inputs["run.tar.gz"]) as archive:
            for member in archive.getmembers():
                assert member.isfile() and member.name in inventory and member.name not in files
                data = archive.extractfile(member).read()
                assert inventory[member.name] == {
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
                files[member.name] = data
        assert files.keys() == inventory.keys() and len(files) == 90
        original = json.loads(files["comparison.json"])
        original_train = json.loads(files["training-diagnostic.json"])
        records, changed, canonical = [], [], {}
        total = 0
        n_undefined = 0
        for split, filename, prediction_file in (
            ("train", "training-known-examples.tsv", "train_selected_predictions.csv"),
            ("val", "val-known-examples.tsv", "val_predictions.csv"),
            ("test", "test-known-examples.tsv", "test_predictions.csv"),
        ):
            frame = pd.read_csv(
                io.BytesIO(files[filename]), sep="\t", keep_default_na=False, dtype=str
            )
            assert frame.sample_id.is_unique and set(frame.source_units) == {"nM"}
            total += len(frame)
            raw = frame.source_value.to_numpy(float)
            q = frame.qualifier.to_numpy(int)
            effective = frame.effective_target_log10_nM.to_numpy(np.float32).astype(float)
            raw_log = frame.raw_target_log10_nM.to_numpy(float)
            assert np.allclose(raw_log, np.log10(raw), rtol=0, atol=1e-12)
            differences = (effective <= np.log10(500)) != (raw <= 500)
            for row in frame.loc[differences].to_dict("records"):
                changed.append(
                    {
                        "split": split,
                        **{
                            key: row[key]
                            for key in (
                                "sample_id",
                                "evidence_row_id",
                                "peptide",
                                "source_value",
                                "source_units",
                                "source_inequality",
                                "source_mhc_alleles",
                                "resolved_mhc_alleles",
                                "pmid",
                                "doi",
                                "reference_iri",
                                "assay_iri",
                                "source_endpoint",
                                "native_mhc_class",
                                "native_cell_line_name",
                                "native_cell_type",
                                "native_condition_id",
                                "native_condition_status",
                                "native_assay_method",
                            )
                        },
                        "original_effective_class": "nonbinder",
                        "physical_class": "binder",
                    }
                )
                assert float(row["source_value"]) == 500 and int(row["qualifier"]) == 0
            old_spaces = original_train if split == "train" else original[split]
            for space, target in (("effective", effective), ("raw", raw_log)):
                for predictor, old in old_spaces[space].items():
                    prediction = frame[predictor + "_prediction_log10_nM"].to_numpy(float)
                    exact = q == 0
                    regression = holdout_eval.regression_metrics(target[exact], prediction[exact])
                    assert np.isclose(regression["rmse"], old["rmse"], atol=1e-12, rtol=1e-12)
                    assert np.isclose(regression["mae"], old["mae"], atol=1e-12, rtol=1e-12)
                    if old["spearman"] is not None:
                        assert np.isclose(regression["spearman"], old["spearman"], atol=1e-12)
                    if np.ptp(prediction[exact]) == 0:
                        assert "pearson" not in regression and abs(old["pearson"]) < 1e-14
                        n_undefined += 1
                    else:
                        assert np.isclose(regression["pearson"], old["pearson"], atol=1e-12)
                    threshold = holdout_eval.binding_threshold_metrics(
                        target, prediction, q, raw_target_nm=raw
                    )
                    independent = independent_binary(raw, prediction, q)
                    for key in ("accuracy", "balanced_accuracy"):
                        assert np.isclose(threshold[key], independent[key], atol=1e-12)
                    assert (
                        threshold["n_physical_target_observations"]
                        == independent["n_usable_observations"]
                    )
                    assert threshold["n_transformed_target_observations"] == 0
                    if split != "train" or space == "raw":
                        for key in (
                            "accuracy",
                            "balanced_accuracy",
                            "auroc",
                            "auprc",
                            "recall",
                            "f1",
                        ):
                            assert np.isclose(threshold[key], old["binding_500nM"][key], atol=1e-12)
                    records.append(
                        {
                            "split": split,
                            "original_target_space": space,
                            "predictor": predictor,
                            "n_observations": len(frame),
                            "n_exact_observations": int(exact.sum()),
                            "unchanged_rmse": regression["rmse"],
                            "old_pearson": old["pearson"],
                            "new_pearson": regression.get("pearson"),
                            "old_accuracy": old["binding_500nM"]["accuracy"],
                            "new_accuracy": threshold["accuracy"],
                            "old_auroc": old["binding_500nM"]["auroc"],
                            "new_auroc": threshold["auroc"],
                            "old_auprc": old["binding_500nM"]["auprc"],
                            "new_auprc": threshold["auprc"],
                            **independent,
                        }
                    )
            predictions = [
                row
                for row in csv.DictReader(io.StringIO(files[prediction_file].decode()))
                if row["task"] == "binding_ic50"
            ]
            assert {row["sample_id"] for row in predictions} == set(frame.sample_id)
            accumulator = holdout_eval.TaskPredictionAccumulator("binding_ic50", "censor")
            accumulator.add(
                [float(r["y_true"]) for r in predictions],
                [float(r["y_pred"]) for r in predictions],
                [1] * len(predictions),
                sample_ids=[r["sample_id"] for r in predictions],
                qualifiers=[int(r["qualifier"]) for r in predictions],
                observations=predictions,
                lineage={
                    key: [r.get(key, "") for r in predictions]
                    for key in holdout_eval.PREDICTION_LINEAGE_FIELDS
                },
            )
            metrics = accumulator.metrics()
            assert metrics["threshold_500nm_n_transformed_target_observations"] == 0
            canonical[split] = {
                key: value for key, value in metrics.items() if key.startswith("threshold_500nm_")
            }
        assert total == 2614 and len(changed) == 4 and n_undefined == 6
        assert {row["split"] for row in changed} == {"train"}
        original_decision = json.loads(files["decision.json"])
        assert original_decision["validation_quality_screen_passed"] is False
        assert original_decision["preferred_over_ridge"] is False
        pd.DataFrame(records).to_csv(outputs["metric-changes.tsv"], sep="\t", index=False)
        pd.DataFrame(changed).to_csv(outputs["boundary-observations.tsv"], sep="\t", index=False)
        write(outputs["canonical-thresholds.json"], canonical)
        write(
            outputs["audit.json"],
            {
                **metadata,
                "original_producer": {
                    "presto": "0.1.16",
                    "hitlist": "1.78.0",
                    "recipe_commit": "1067714c768af2e4f3184154c85903bb3c46cf4b",
                },
                "runtime_seconds": time.monotonic() - started,
                "runtime_scope": "audit builder only; source acquisition/imports excluded",
                "process_peak_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if sys.platform == "darwin" else 1024)
                ),
                "n_verified_archive_files": len(files),
                "n_source_observations": total,
                "n_corrected_training_boundary_observations": len(changed),
                "n_corrected_undefined_correlations": n_undefined,
                "n_changed_heldout_binding_labels": 0,
                "regression_metrics_unchanged_except_undefined_correlations": True,
                "unchanged_historical_decision": original_decision,
                "limits": [
                    "No retraining, changed prediction or new biological evidence.",
                    "Shared studies, unknown donor/specimen/acquisition identities, species-context and narrow-prediction limitations remain.",
                    "Conservative censor-direction grouping retained; strict/inclusive operators remain original lineage fields.",
                ],
            },
        )
        shutil.copyfile(__file__, outputs["audit_recipe.py"])
        shutil.copyfile(holdout_eval.__file__, outputs["holdout_eval.py"])

    names = [
        "metric-changes.tsv",
        "boundary-observations.tsv",
        "canonical-thresholds.json",
        "audit.json",
        "audit_recipe.py",
        "holdout_eval.py",
    ]
    paths = materialize(
        Path.home() / ".hitlist/presto-physical-affinity-metrics" / commit,
        INPUTS,
        transform={
            "version": "physical-affinity-metrics-audit-v1",
            "options": metadata,
        },
        outputs={name: {} for name in names},
        builder=build,
    )
    receipt = {
        "commit": commit,
        "files": {name: {"path": str(path), **pin(path)} for name, path in paths.items()},
    }
    write(FAMILY / "results/audit-receipt.json", receipt)
    for name, path in paths.items():
        if name.endswith((".json", ".tsv")):
            shutil.copyfile(path, FAMILY / "results" / name)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
