"""Fixed training-only response controls using qualified frozen MHC scores."""

import hashlib
import importlib.util
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
RECIPE = ROOT / "data/hap1_response_recipes/2026-10-24_0730_v1"
PREVIOUS = ROOT / "experiments/2026-10-24_2018_codex_hap1-context-ablation"
OLD = ROOT / "experiments/2026-10-24_0730_codex_hap1-response-baselines/results/fits"
ALLELES = ["HLA-A*02:01", "HLA-B*40:01", "HLA-C*03:04"]
VARIANTS = [
    "calibrated_precursor",
    "sequence",
    "length_affinity",
    "sequence_affinity",
    "sequence_shuffled_affinity",
    "sequence_affinity_geometry",
]
spec = importlib.util.spec_from_file_location("reference_core", PREVIOUS / "code/core.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def design(rows, variant, scores, geometry):
    mapping = (
        reference.donor_map([reference.key(r) for r in rows])
        if variant == "sequence_shuffled_affinity"
        else {}
    )
    values = []
    for r in rows:
        s, k = r["peptide_sequence"], reference.key(r)
        if variant == "calibrated_precursor":
            features = []
        elif variant == "length_affinity":
            features = [len(s), *scores[s]]
        else:
            features = reference.sequence_features(s).tolist()
            if variant != "sequence":
                donor = json.loads(mapping[k])[0] if mapping else s
                features.extend(scores[donor])
            if variant == "sequence_affinity_geometry":
                features.extend(geometry[k][:4])
        values.append(features)
    return np.array(values), mapping


def main():
    start = time.perf_counter()
    out = EXP / "results/fits"
    out.mkdir(exist_ok=False)
    qualified = json.loads((EXP / "results/predictor/qualification.json").read_text())
    assert qualified["status"] == "passed"
    assert (
        hashlib.sha256((EXP / "results/predictor/scores.parquet").read_bytes()).hexdigest()
        == qualified["scores_sha256"]
    )
    manifest = json.loads((PREVIOUS / "artifact-manifest.json").read_text())
    dependencies = ["code/core.py", "results/fits/context-features.npz"]
    for name in dependencies:
        assert (
            hashlib.sha256((PREVIOUS / name).read_bytes()).hexdigest() == manifest[name]["sha256"]
        )
    shutil.copyfile(PREVIOUS / "code/core.py", EXP / "reproduce/source/reference-core.py")
    shutil.copyfile(__file__, EXP / "reproduce/source/launch.py")
    table = pq.read_table(
        RECIPE / "derived/responses.parquet",
        filters=[("split", "=", "train"), ("precursor_status", "=", "complete")],
    )
    assert table.equals(pq.read_table(OLD / "cohort.parquet"))
    rows = table.to_pylist()
    frame = pd.read_parquet(EXP / "results/predictor/scores.parquet")
    scores = {
        p: np.log10(group.set_index("allele").loc[ALLELES, "prediction"].to_numpy())
        for p, group in frame.groupby("peptide")
    }
    context = np.load(PREVIOUS / "results/fits/context-features.npz")
    geometry = dict(zip(context["keys"], context["features"], strict=True))
    exposure = pd.read_parquet(EXP / "results/predictor/exposure.parquet").set_index("peptide")
    seen = np.array(
        [bool(exposure.loc[r["peptide_sequence"], "any_training_peptide"]) for r in rows]
    )
    fold = np.array(
        [
            int(
                hashlib.sha256(
                    ("hap1_baseline_inner_v1:" + r["component_id"]).encode()
                ).hexdigest(),
                16,
            )
            % 5
            for r in rows
        ]
    )
    y = np.array([r["target_log10_ratio"] for r in rows])
    w = reference.weights(rows)
    conditions = np.array([r["condition_id"] for r in rows])
    predictions = {
        "measurement_id": table["measurement_id"],
        "source_record_id": table["source_record_id"],
        "component_id": table["component_id"],
        "condition_id": table["condition_id"],
        "peptide_sequence": table["peptide_sequence"],
        "fold": fold,
        "weight": w,
        "target_log10_ratio": y,
        "frozen_predictor_training_exposure": seen,
    }
    receipt = {
        "argv": sys.argv,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], text=True
        ),
        "python": sys.executable,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_dependencies": {name: manifest[name] for name in dependencies},
        "predictor_qualification": qualified,
        "environment": {
            k: os.environ.get(k)
            for k in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ]
        },
        "outer_validation_test_scored": False,
    }
    write(EXP / "reproduce/launch.json", receipt)
    summary, strata, checkpoints, donor_maps, reference_errors = {}, [], {}, {}, {}
    for variant in VARIANTS:
        t = time.perf_counter()
        full, cv = np.full(len(rows), np.nan), np.full(len(rows), np.nan)
        checkpoints[variant], donor_maps[variant] = {}, {}
        for part in ["full", "0", "1", "2", "3", "4"]:
            it = np.arange(len(rows)) if part == "full" else np.flatnonzero(fold != int(part))
            ie = np.arange(len(rows)) if part == "full" else np.flatnonzero(fold == int(part))
            tr, ev = [rows[i] for i in it], [rows[i] for i in ie]
            a, td = design(tr, variant, scores, geometry)
            b, ed = design(ev, variant, scores, geometry)
            p, cp = reference.fit(tr, ev, a, b)
            checkpoints[variant][part] = cp
            if td:
                donor_maps[variant][part] = {"fit": td, "eval": ed}
            if part == "full":
                full[:] = p
            else:
                cv[ie] = p
                strata.append(
                    {
                        "variant": variant,
                        "kind": "fold",
                        "label": part,
                        **reference.metrics(y[ie], p, w[ie]),
                    }
                )
        for population, p in [("train", full), ("inner_cv", cv)]:
            assert np.isfinite(p).all()
            predictions[f"{variant}_{population}"] = p
            for cond in sorted(set(conditions)):
                mask = conditions == cond
                strata.append(
                    {
                        "variant": variant,
                        "kind": population + "_condition",
                        "label": cond,
                        **reference.metrics(y[mask], p[mask], w[mask]),
                    }
                )
            for exposed in [False, True]:
                mask = seen == exposed
                if mask.any():
                    strata.append(
                        {
                            "variant": variant,
                            "kind": population + "_exposure",
                            "label": str(exposed),
                            **reference.metrics(y[mask], p[mask], w[mask]),
                        }
                    )
        summary[variant] = {
            "train": reference.metrics(y, full, w),
            "inner_cv": reference.metrics(y, cv, w),
            "seconds": time.perf_counter() - t,
        }
        if variant in {"calibrated_precursor", "sequence"}:
            name = "sequence_0.1" if variant == "sequence" else variant
            errors = []
            for pop, p in [("train", full), ("inner_cv", cv)]:
                old = pq.read_table(OLD / f"{name}-{pop}.parquet")
                assert old["measurement_id"].equals(table["measurement_id"])
                errors.append(float(max(abs(old["prediction_log10_ratio"].to_numpy() - p))))
            assert max(errors) < 1e-10
            reference_errors[variant] = max(errors)
        print(json.dumps({"variant": variant, **summary[variant]}), flush=True)
        assert (
            time.perf_counter() - start < 300
            and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3 < 3
        )
    pq.write_table(pa.table(predictions), out / "predictions.parquet", compression="zstd")
    write(out / "checkpoints.json", checkpoints)
    write(out / "donor-mappings.json", donor_maps)
    write(out / "strata.json", strata)
    get = lambda v: summary[v]["inner_cv"]["rmse"]
    gain = 1 - get("sequence_affinity") / get("sequence")
    shuffle_gain = 1 - get("sequence_affinity") / get("sequence_shuffled_affinity")
    folds = {(s["variant"], s["label"]): s["rmse"] for s in strata if s["kind"] == "fold"}
    n_improved = sum(
        folds["sequence_affinity", str(k)] < folds["sequence", str(k)] for k in range(5)
    )
    result = {
        "metrics": summary,
        "relative_gain": gain,
        "gain_vs_shuffled": shuffle_gain,
        "folds_improved": n_improved,
        "development_gate_pass": gain >= 0.02 and shuffle_gain >= 0.01 and n_improved >= 4,
        "reference_errors": reference_errors,
        "seconds": time.perf_counter() - start,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "outer_validation_test_scored": False,
    }
    write(out / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "metrics"}), flush=True)


if __name__ == "__main__":
    main()
