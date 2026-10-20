"""Independently check published numerical corrections without importing Presto."""

import hashlib
import importlib.metadata
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from datacache import install_bundle, materialize
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

FAMILY = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".hitlist/presto-physical-affinity-metrics/public-verification"


def pin(path):
    path = Path(path)
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}


def verify(publication):
    payload = json.loads(Path(publication).read_text())
    assets = {
        name: {key: record[key] for key in ("url", "sha256", "size")}
        for name, record in payload["files"].items()
    }
    paths = install_bundle(CACHE / pin(publication)["sha256"], assets)
    audit = json.loads(Path(paths["audit.json"]).read_text())
    inputs = audit["inputs"]
    identity = pin(__file__)["sha256"]

    def build(source, outputs):
        for name in inputs:
            Path(outputs[name]).write_bytes(Path(source[name]).read_bytes())

    sources = materialize(
        CACHE / "original-inputs" / identity,
        inputs,
        transform={"version": "independent-metric-inputs-v1", "options": inputs},
        outputs={name: {} for name in inputs},
        builder=build,
    )
    inventory = json.loads(Path(sources["inventory.json"]).read_text())
    with tarfile.open(sources["run.tar.gz"]) as archive:
        members = archive.getmembers()
        assert all(member.isfile() for member in members)
        assert len(members) == len(inventory) == 90
        files = {member.name: archive.extractfile(member).read() for member in members}
    assert files.keys() == inventory.keys()
    for name, data in files.items():
        assert inventory[name] == {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
        }
    tables = {
        split: pd.read_csv(io.BytesIO(files[name]), sep="\t", dtype=str, keep_default_na=False)
        for split, name in (
            ("train", "training-known-examples.tsv"),
            ("val", "val-known-examples.tsv"),
            ("test", "test-known-examples.tsv"),
        )
    }
    changes = pd.read_csv(paths["metric-changes.tsv"], sep="\t", dtype=str, keep_default_na=False)
    boundary = pd.read_csv(
        paths["boundary-observations.tsv"], sep="\t", dtype=str, keep_default_na=False
    )
    canonical = json.loads(Path(paths["canonical-thresholds.json"]).read_text())
    old_train = json.loads(files["training-diagnostic.json"])
    old_heldout = json.loads(files["comparison.json"])
    n_constant, verified_pairs = 0, set()
    expected_boundary = {}
    for split, frame in tables.items():
        assert frame.sample_id.is_unique
        raw = frame.source_value.to_numpy(float)
        effective = frame.effective_target_log10_nM.to_numpy(np.float32).astype(float)
        altered = (raw <= 500) != (effective <= np.log10(500))
        for row in frame.loc[altered].to_dict("records"):
            expected_boundary[row["sample_id"]] = (split, row)
    assert len(boundary) == len(expected_boundary) == 4
    for row in boundary.to_dict("records"):
        split, source = expected_boundary[row["sample_id"]]
        assert row["split"] == split == "train"
        assert row["original_effective_class"] == "nonbinder" and row["physical_class"] == "binder"
        for field in boundary.columns:
            if field not in ("split", "original_effective_class", "physical_class"):
                assert row[field] == source[field], (row["sample_id"], field)
    assert len(changes) == 20
    for row in changes.to_dict("records"):
        split, space, predictor = row["split"], row["original_target_space"], row["predictor"]
        key = split, space, predictor
        assert key not in verified_pairs
        verified_pairs.add(key)
        frame = tables[split]
        raw, q = frame.source_value.to_numpy(float), frame.qualifier.to_numpy(int)
        target = (
            frame.effective_target_log10_nM.to_numpy(np.float32).astype(float)
            if space == "effective"
            else np.log10(raw)
        )
        pred = frame[predictor + "_prediction_log10_nM"].to_numpy(float)
        exact = q == 0
        usable = exact | ((q < 0) & (raw <= 500)) | ((q > 0) & (raw > 500))
        labels, prediction = raw[usable] <= 500, pred[usable] <= np.log10(500)
        old = (old_train if split == "train" else old_heldout[split])[space][predictor]
        assert int(row["n_observations"]) == len(frame)
        assert int(row["n_exact_observations"]) == int(exact.sum())
        assert int(row["n_positive_observations"]) == int(labels.sum())
        assert int(row["n_usable_observations"]) == int(usable.sum())
        expected = {
            "unchanged_rmse": np.sqrt(np.mean((target[exact] - pred[exact]) ** 2)),
            "new_accuracy": accuracy_score(labels, prediction),
            "new_auroc": roc_auc_score(labels, -pred[usable]),
            "new_auprc": average_precision_score(labels, -pred[usable]),
            "balanced_accuracy": balanced_accuracy_score(labels, prediction),
        }
        for name, value in expected.items():
            assert np.isclose(float(row[name]), value, rtol=1e-12, atol=1e-12), (key, name)
        for name in ("accuracy", "auroc", "auprc"):
            assert float(row["old_" + name]) == old["binding_500nM"][name]
            if split != "train" or space == "raw":
                assert np.isclose(float(row["new_" + name]), old["binding_500nM"][name], atol=1e-12)
        assert float(row["old_pearson"]) == old["pearson"]
        if np.ptp(pred[exact]) == 0:
            assert row["new_pearson"] == "" and abs(old["pearson"]) < 1e-14
            n_constant += 1
        else:
            assert np.isclose(
                float(row["new_pearson"]), np.corrcoef(target[exact], pred[exact])[0, 1], atol=1e-12
            )
        if space == "effective" and predictor == ("selected" if split == "train" else "presto"):
            report = canonical[split]
            for name in ("accuracy", "auroc", "auprc"):
                assert np.isclose(
                    report["threshold_500nm_" + name], expected["new_" + name], atol=1e-12
                )
            assert report["threshold_500nm_n_physical_target_observations"] == int(usable.sum())
            assert report["threshold_500nm_n_transformed_target_observations"] == 0
    assert n_constant == 6 and sum(map(len, tables.values())) == 2614
    assert audit["unchanged_historical_decision"] == json.loads(files["decision.json"])
    assert audit["unchanged_historical_decision"]["validation_quality_screen_passed"] is False
    return {
        "verified": True,
        "publication": payload["release"],
        "n_public_assets": len(paths),
        "n_archive_files": 90,
        "n_observations": 2614,
        "n_comparisons": len(verified_pairs),
        "n_boundary_observations": 4,
        "n_undefined_correlations": 6,
        "unchanged_failed_quality_decision": True,
        "verifier": pin(__file__),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "datacache")
        },
        "public_downloads": {
            name: {"path": str(path), **pin(path)} for name, path in paths.items()
        },
    }


if __name__ == "__main__":
    import sys

    result = verify(sys.argv[1])
    (FAMILY / "results/public-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "public_downloads"}, indent=2
        )
    )
