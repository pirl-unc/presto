"""Qualify and fit the frozen nine-arm HAP1 development comparison."""

import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from core import VARIANTS, context_features, design, fit, key, metrics, weights
from independent_features import fixture_check, reconstruct, source_origin_sets

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
RECIPE = ROOT / "data/hap1_response_recipes/2026-10-24_0730_v1"
OLD = ROOT / "experiments/2026-10-24_0730_codex_hap1-response-baselines/results/fits"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    start = time.perf_counter()
    out = EXP / "results/fits"
    out.mkdir(exist_ok=False)
    for path in (EXP / "code").glob("*.py"):
        shutil.copyfile(path, EXP / "reproduce/source" / path.name)
    receipt = {
        "argv": sys.argv,
        "agent_model": "Codex / GPT-6",
        "python": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], text=True
        ),
        "environment": {
            k: os.environ.get(k)
            for k in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ]
        },
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                RECIPE / "manifest.json",
                RECIPE / "derived/responses.parquet",
                RECIPE / "derived/parents.parquet",
                RECIPE / "source/occurrences.parquet",
                EXP / "reproduce/plan.md",
                *sorted((EXP / "code").glob("*.py")),
            ]
        },
        "outer_validation_test": "not_loaded_or_scored",
        "variants": VARIANTS,
    }
    assert (
        receipt["sources"][str((RECIPE / "manifest.json").relative_to(ROOT))]
        == "a1eec392c06485966e131663f85e7599dcd9e5ff621841eb0d44eeaafe779309"
    )
    write(EXP / "reproduce/launch.json", receipt)
    table = pq.read_table(
        RECIPE / "derived/responses.parquet",
        filters=[("split", "=", "train"), ("precursor_status", "=", "complete")],
    )
    assert table.equals(pq.read_table(OLD / "cohort.parquet"))
    rows = table.to_pylist()
    assert len(rows) == 87260
    assert len({r["peptide_sequence"] for r in rows}) == 6317
    assert len({r["component_id"] for r in rows}) == 1798
    keys = sorted({key(r) for r in rows})
    all_parents = pq.read_table(RECIPE / "derived/parents.parquet").to_pylist()
    needed = {p for k in keys for p in json.loads(k)[1]}
    parents = {
        p["parent_id"]: p["reference_sequence"] for p in all_parents if p["parent_id"] in needed
    }
    reference_parent = {
        rid: p["parent_id"]
        for p in all_parents
        if p["parent_id"] in needed
        for rid in json.loads(p["reference_ids_json"])
    }
    context, lineage = {}, []
    for k in keys:
        s, ids = json.loads(k)
        context[k], origins = context_features(s, ids, parents)
        lineage.extend({"key": k, **origin} for origin in origins)
    independent, expected_origins = reconstruct(keys, parents)
    feature_error = max(float(np.max(abs(context[k] - independent[k]))) for k in keys)
    assert feature_error < 1e-12
    source_ids = {r["source_record_id"] for r in rows}
    canonical = source_origin_sets(
        pq.read_table(RECIPE / "source/occurrences.parquet").to_pylist(),
        reference_parent,
        source_ids,
    )
    for r in rows:
        ids = set(json.loads(r["parent_ids_json"]))
        actual = {v for v in canonical[r["source_record_id"]] if v[0] in ids}
        assert actual == expected_origins[key(r)], (
            r["source_record_id"],
            actual,
            expected_origins[key(r)],
        )
    pq.write_table(
        pa.Table.from_pylist(lineage), out / "feature-lineage.parquet", compression="zstd"
    )
    np.savez_compressed(
        out / "context-features.npz",
        keys=np.array(keys),
        features=np.stack([context[k] for k in keys]),
    )
    qualification = {
        "context_max_error": feature_error,
        "fixtures": fixture_check(context_features),
        "n_unique_keys": len(keys),
        "n_parents": len(parents),
        "n_origins": len(lineage),
        "source_origin_sets_match": True,
    }
    write(out / "feature-qualification.json", qualification)
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
    w = weights(rows)
    y = np.array([r["target_log10_ratio"] for r in rows])
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
    }
    summaries, checkpoints, strata, donor_receipts = {}, {}, [], {}
    fit_start = time.perf_counter()
    for variant in VARIANTS:
        t = time.perf_counter()
        full = np.full(len(rows), np.nan)
        cv = np.full(len(rows), np.nan)
        checkpoints[variant], donor_receipts[variant] = {}, {}
        for part in ["full", "0", "1", "2", "3", "4"]:
            select = np.arange(len(rows)) if part == "full" else np.flatnonzero(fold != int(part))
            evaluate = np.arange(len(rows)) if part == "full" else np.flatnonzero(fold == int(part))
            fitting = [rows[i] for i in select]
            evaluation = [rows[i] for i in evaluate]
            phi, donors_fit = design(fitting, variant, context)
            phie, donors_eval = design(evaluation, variant, context)
            p, cp = fit(fitting, evaluation, phi, phie)
            checkpoints[variant][part] = cp
            if donors_fit:
                donor_receipts[variant][part] = {"fit": donors_fit, "eval": donors_eval}
            if part == "full":
                full[:] = p
            else:
                cv[evaluate] = p
                strata.append(
                    {
                        "variant": variant,
                        "kind": "fold",
                        "label": part,
                        **metrics(y[evaluate], p, w[evaluate]),
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
                        **metrics(y[mask], p[mask], w[mask]),
                    }
                )
        summaries[variant] = {
            "train": metrics(y, full, w),
            "inner_cv": metrics(y, cv, w),
            "seconds": time.perf_counter() - t,
        }
        if variant in {"calibrated_precursor", "sequence"}:
            old_variant = "sequence_0.1" if variant == "sequence" else variant
            errors = []
            for population, p in [("train", full), ("inner_cv", cv)]:
                original = pq.read_table(OLD / f"{old_variant}-{population}.parquet")
                assert original["measurement_id"].equals(table["measurement_id"])
                assert np.array_equal(original["fold"].to_numpy(), fold)
                assert np.max(abs(original["weight"].to_numpy() - w)) < 1e-14
                errors.append(float(np.max(abs(original["prediction_log10_ratio"].to_numpy() - p))))
            assert max(errors) < 1e-10
            qualification[variant + "_reference_max_error"] = max(errors)
            write(out / "feature-qualification.json", qualification)
        print(json.dumps({"variant": variant, **summaries[variant]}), flush=True)
        assert time.perf_counter() - fit_start < 300
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3 < 3
    qualification["status"] = "passed"
    write(out / "feature-qualification.json", qualification)
    pq.write_table(pa.table(predictions), out / "predictions.parquet", compression="zstd")
    write(out / "checkpoints.json", checkpoints)
    write(out / "donor-mappings.json", donor_receipts)
    write(out / "strata.json", strata)
    get = lambda v: summaries[v]["inner_cv"]["rmse"]
    gain = 1 - get("sequence_context") / get("sequence")
    shuffle_gain = 1 - get("sequence_context") / get("sequence_shuffled_context")
    by_fold = {(s["variant"], s["label"]): s["rmse"] for s in strata if s["kind"] == "fold"}
    folds_improved = sum(
        by_fold["sequence_context", str(k)] < by_fold["sequence", str(k)] for k in range(5)
    )
    summary = {
        "metrics": summaries,
        "context_relative_gain": gain,
        "context_gain_vs_shuffled": shuffle_gain,
        "folds_improved": folds_improved,
        "development_gate_pass": gain >= 0.02 and shuffle_gain >= 0.01 and folds_improved >= 4,
        "fit_seconds": time.perf_counter() - fit_start,
        "total_seconds": time.perf_counter() - start,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
        "outer_validation_test_scored": False,
    }
    write(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}), flush=True)


if __name__ == "__main__":
    main()
