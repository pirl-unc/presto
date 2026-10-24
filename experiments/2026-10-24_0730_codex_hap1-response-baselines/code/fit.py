"""Fixed HAP1 biological baselines: outer-training grouped CV only."""

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

AA = "ACDEFGHIKLMNPQRSTVWY"
VARIANTS = [
    "zero",
    "condition",
    "fixed_precursor",
    "calibrated_precursor",
    "sequence_0.001",
    "sequence_0.01",
    "sequence_0.1",
]


def weights(rows):
    counts = Counter((r["condition_id"], r["component_id"], r["peptide_sequence"]) for r in rows)
    peptides = Counter((c, g) for c, g, _ in counts)
    groups = Counter(c for c, _ in peptides)
    n_conditions = len(groups)
    return np.array(
        [
            1
            / (
                n_conditions
                * groups[r["condition_id"]]
                * peptides[r["condition_id"], r["component_id"]]
                * counts[r["condition_id"], r["component_id"], r["peptide_sequence"]]
            )
            for r in rows
        ]
    )


def features(rows):
    return np.array(
        [
            [
                len(s),
                *[s.count(a) / len(s) for a in AA],
                *[float(s[0] == a) for a in AA],
                *[float(s[-1] == a) for a in AA],
            ]
            for s in (r["peptide_sequence"] for r in rows)
        ],
        dtype=float,
    )


def metrics(y, prediction, w):
    w = w / w.sum()
    error = prediction - y
    mse = float(w @ error**2)
    variance = float(w @ (y - w @ y) ** 2)
    varying = bool(np.ptp(prediction) > 1e-14)
    return {
        "n_observations": len(y),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(w @ abs(error)),
        "r2": 1 - mse / variance if variance else None,
        "pearson": float(pearsonr(y, prediction).statistic) if varying and np.ptp(y) else None,
        "spearman": float(spearmanr(y, prediction).statistic) if varying and np.ptp(y) else None,
        "weighted_prediction_sd": float(np.sqrt(w @ (prediction - w @ prediction) ** 2)),
        "weighted_target_sd": float(np.sqrt(variance)),
    }


def fit(train_rows, eval_rows, variant):
    y = np.array([r["target_log10_ratio"] for r in train_rows])
    x = np.array([r["precursor_log10_ratio"] for r in train_rows])
    w = weights(train_rows)
    c = np.array([r["condition_id"] for r in train_rows])
    ce = np.array([r["condition_id"] for r in eval_rows])
    xe = np.array([r["precursor_log10_ratio"] for r in eval_rows])
    conditions = sorted(set(c))
    if set(ce) - set(c):
        raise ValueError("Evaluation condition lacks fitting support")
    means = {
        cond: [
            float(w[c == cond] @ y[c == cond] / w[c == cond].sum()),
            float(w[c == cond] @ x[c == cond] / w[c == cond].sum()),
        ]
        for cond in conditions
    }
    yr = y - np.array([means[cond][0] for cond in c])
    xr = x - np.array([means[cond][1] for cond in c])
    denominator = float(w @ xr**2)
    gamma = max(0.0, float(w @ (xr * yr)) / denominator) if denominator > 0 else 0.0
    gamma = (
        0.0 if variant in {"zero", "condition"} else 1.0 if variant == "fixed_precursor" else gamma
    )
    offsets = {cond: mean_y - gamma * mean_x for cond, (mean_y, mean_x) in means.items()}
    prediction = gamma * xe + np.array([offsets[cond] for cond in ce])
    checkpoint = {
        "variant": variant,
        "gamma": gamma,
        "condition_offsets": offsets,
        "n_fit_observations": len(y),
        "n_fit_components": len({r["component_id"] for r in train_rows}),
    }
    if variant == "zero":
        prediction[:] = 0
        checkpoint["condition_offsets"] = {}
    if variant.startswith("sequence_"):
        penalty = float(variant.split("_")[1])
        phi, phie = features(train_rows), features(eval_rows)
        mean = w @ phi
        sd = np.sqrt(w @ (phi - mean) ** 2)
        mask = sd > 1e-12
        z = (phi[:, mask] - mean[mask]) / sd[mask]
        ze = (phie[:, mask] - mean[mask]) / sd[mask]
        checkpoint.update(
            {
                "feature_mean": mean.tolist(),
                "feature_sd": sd.tolist(),
                "feature_mask": mask.tolist(),
                "penalty": penalty,
                "sequence": {},
            }
        )
        residual = y - gamma * x - np.array([offsets[cond] for cond in c])
        for cond in conditions:
            select, evaluate = c == cond, ce == cond
            wc = w[select] / w[select].sum()
            mu = wc @ z[select]
            design = z[select] - mu
            gram = design.T @ (wc[:, None] * design) + penalty * np.eye(mask.sum())
            rhs = design.T @ (wc * residual[select])
            beta = np.linalg.solve(gram, rhs)
            prediction[evaluate] += (ze[evaluate] - mu) @ beta
            checkpoint["sequence"][cond] = {
                "center": mu.tolist(),
                "beta": beta.tolist(),
                "normal_equation_max_residual": float(np.max(np.abs(gram @ beta - rhs))),
                "n_supported_coordinates": int(np.sum(wc @ design**2 > 1e-12)),
                "n_nonzero_coefficients": int(np.sum(np.abs(beta) > 1e-12)),
            }
    return prediction, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    args = parser.parse_args()
    exp = args.experiment.resolve()
    output = exp / "results/fits"
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, exp / "reproduce/source/fit.py")
    start = time.perf_counter()
    recipe_digest = hashlib.sha256((args.recipe / "manifest.json").read_bytes()).hexdigest()
    verification = json.loads((exp / "results/independent-admission.json").read_text())
    assert (
        verification["status"] == "passed"
        and verification["recipe_manifest_sha256"] == recipe_digest
    )
    # Parquet filter excludes outer validation/test outcomes from the modeling table.
    table = pq.read_table(
        args.recipe / "derived/responses.parquet",
        filters=[("split", "=", "train"), ("precursor_status", "=", "complete")],
    )
    rows = table.to_pylist()
    assert len(rows) and all(r["split"] == "train" for r in rows)
    pq.write_table(table, output / "cohort.parquet", compression="zstd")
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
    w = weights(rows)
    assert abs(w.sum() - 1) < 1e-12
    conditions = np.array([r["condition_id"] for r in rows])
    components = np.array([r["component_id"] for r in rows])
    receipt = {
        "argv": sys.argv,
        "python": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "recipe_manifest_sha256": recipe_digest,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=no"], text=True
        ),
        "environment": {
            k: os.environ.get(k)
            for k in ["PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
        },
        "n_observations": len(rows),
        "n_peptides": len({r["peptide_sequence"] for r in rows}),
        "n_components": len(set(components)),
        "observations_per_fold": Counter(map(str, fold)),
        "outer_validation_test": "never_loaded_or_scored",
    }
    (output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    summaries, checkpoints, strata = {}, {}, []
    for variant in VARIANTS:
        t = time.perf_counter()
        full, checkpoint = fit(rows, rows, variant)
        checkpoints[variant] = {"full": checkpoint, "folds": {}}
        cv = np.full(len(rows), np.nan)
        for k in range(5):
            select = np.flatnonzero(fold != k)
            evaluate = np.flatnonzero(fold == k)
            train_rows = [rows[i] for i in select]
            eval_rows = [rows[i] for i in evaluate]
            prediction, checkpoint = fit(train_rows, eval_rows, variant)
            cv[evaluate] = prediction
            checkpoints[variant]["folds"][str(k)] = checkpoint
            for cond in sorted(set(conditions[evaluate])):
                indices = evaluate[conditions[evaluate] == cond]
                strata.append(
                    {
                        "variant": variant,
                        "population": "inner_validation_fold",
                        "stratum": f"{k}:{cond}",
                        **metrics(y[indices], cv[indices], w[indices]),
                    }
                )
        assert np.isfinite(cv).all()
        summaries[variant] = {
            "train": metrics(y, full, w),
            "inner_cv": metrics(y, cv, w),
            "seconds": time.perf_counter() - t,
        }
        for name, prediction in [("train", full), ("inner_cv", cv)]:
            predictions = pa.table(
                {
                    "measurement_id": table["measurement_id"],
                    "component_id": table["component_id"],
                    "condition_id": table["condition_id"],
                    "peptide_sequence": table["peptide_sequence"],
                    "fold": fold,
                    "target_log10_ratio": y,
                    "weight": w,
                    "prediction_log10_ratio": prediction,
                }
            )
            pq.write_table(predictions, output / f"{variant}-{name}.parquet", compression="zstd")
            for kind, labels in [("condition", conditions), ("component", components)]:
                for label in sorted(set(labels)):
                    indices = labels == label
                    strata.append(
                        {
                            "variant": variant,
                            "population": name,
                            "stratum": f"{kind}:{label}",
                            **metrics(y[indices], prediction[indices], w[indices]),
                        }
                    )
        print(json.dumps({"variant": variant, **summaries[variant]}), flush=True)
    control = min(VARIANTS[:4], key=lambda v: summaries[v]["inner_cv"]["rmse"])
    sequence = min(VARIANTS[4:], key=lambda v: summaries[v]["inner_cv"]["rmse"])
    improvement = (
        1 - summaries[sequence]["inner_cv"]["rmse"] / summaries[control]["inner_cv"]["rmse"]
    )
    spreads = {
        r["stratum"]: r["weighted_prediction_sd"]
        for r in strata
        if r["variant"] == sequence
        and r["population"] == "inner_cv"
        and r["stratum"].startswith("condition:")
    }
    result = {
        "metrics": summaries,
        "best_control": control,
        "best_sequence": sequence,
        "relative_inner_cv_rmse_improvement": improvement,
        "sequence_within_condition_sd": spreads,
        "development_gate_pass": bool(improvement >= 0.02 and min(spreads.values()) > 0.001),
        "outer_validation_test_scored": False,
        "seconds": time.perf_counter() - start,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    (output / "checkpoints.json").write_text(json.dumps(checkpoints, sort_keys=True) + "\n")
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "strata.json").write_text(json.dumps(strata, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)
    assert result["seconds"] <= 300 and result["peak_gib"] <= 2


if __name__ == "__main__":
    main()
