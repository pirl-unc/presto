"""Independent coefficient, prediction, weight and metric closure for fixed baselines."""

import argparse
import hashlib
import json
import math
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.linalg import lstsq
from scipy.stats import pearsonr, spearmanr


def weighted(rows):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for index, row in enumerate(rows):
        groups[row["condition_id"]][row["component_id"]][row["peptide_sequence"]].append(index)
    weight = np.zeros(len(rows))
    for components in groups.values():
        for peptides in components.values():
            for indices in peptides.values():
                weight[indices] = 1 / (len(groups) * len(components) * len(peptides) * len(indices))
    assert math.isclose(weight.sum(), 1)
    return weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    root = args.experiment / "results/fits"
    rows = pq.read_table(root / "cohort.parquet").to_pylist()
    assert all(r["split"] == "train" and r["precursor_status"] == "complete" for r in rows)
    checkpoints = json.loads((root / "checkpoints.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    y = np.array([r["target_log10_ratio"] for r in rows])
    x = np.array([r["precursor_log10_ratio"] for r in rows])
    c = np.array([r["condition_id"] for r in rows])
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
    w = weighted(rows)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    columns = []
    for row in rows:
        s = row["peptide_sequence"]
        counts = np.zeros(20)
        for a in s:
            counts[alphabet.index(a)] += 1
        columns.append(
            np.concatenate(
                (
                    [len(s)],
                    counts / len(s),
                    np.eye(20)[alphabet.index(s[0])],
                    np.eye(20)[alphabet.index(s[-1])],
                )
            )
        )
    phi = np.stack(columns)
    max_prediction_error, max_coefficient_error = 0.0, 0.0
    n_predictions = 0
    for variant, stored in checkpoints.items():
        for fit_name, checkpoint in [("full", stored["full"]), *stored["folds"].items()]:
            indices = (
                np.arange(len(rows))
                if fit_name == "full"
                else np.flatnonzero(fold != int(fit_name))
            )
            evaluate = (
                np.arange(len(rows))
                if fit_name == "full"
                else np.flatnonzero(fold == int(fit_name))
            )
            train_rows = [rows[i] for i in indices]
            wt = weighted(train_rows)
            yt, xt, ct = y[indices], x[indices], c[indices]
            offsets = {}
            xr, yr = xt.copy(), yt.copy()
            for condition in sorted(set(ct)):
                mask = ct == condition
                mean_x = np.average(xt[mask], weights=wt[mask])
                mean_y = np.average(yt[mask], weights=wt[mask])
                xr[mask] -= mean_x
                yr[mask] -= mean_y
            # One-dimensional nonnegative least squares through an explicit design.
            a, b = np.sqrt(wt) * xr, np.sqrt(wt) * yr
            gamma = max(0.0, float(lstsq(a[:, None], b)[0][0])) if np.linalg.norm(a) else 0.0
            if variant in {"zero", "condition"}:
                gamma = 0.0
            if variant == "fixed_precursor":
                gamma = 1.0
            assert abs(gamma - checkpoint["gamma"]) < 1e-10
            pred = gamma * x[evaluate]
            residual = yt - gamma * xt
            for condition in sorted(set(ct)):
                mask = ct == condition
                offset = float(np.average(residual[mask], weights=wt[mask]))
                offsets[condition] = offset
                residual[mask] -= offset
                pred[c[evaluate] == condition] += offset
                if variant != "zero":
                    assert abs(offset - checkpoint["condition_offsets"][condition]) < 1e-10
            if variant == "zero":
                pred[:] = 0
            if variant.startswith("sequence"):
                mean = np.average(phi[indices], weights=wt, axis=0)
                sd = np.sqrt(np.average((phi[indices] - mean) ** 2, weights=wt, axis=0))
                active = sd > 1e-12
                np.testing.assert_allclose(mean, checkpoint["feature_mean"], atol=1e-10)
                np.testing.assert_allclose(sd, checkpoint["feature_sd"], atol=1e-10)
                assert active.tolist() == checkpoint["feature_mask"]
                train = (phi[indices][:, active] - mean[active]) / sd[active]
                evaluation = (phi[evaluate][:, active] - mean[active]) / sd[active]
                for condition in sorted(set(ct)):
                    mask = ct == condition
                    wc = wt[mask] / wt[mask].sum()
                    center = np.average(train[mask], weights=wc, axis=0)
                    design = train[mask] - center
                    augmented = np.vstack(
                        (
                            np.sqrt(wc)[:, None] * design,
                            np.sqrt(checkpoint["penalty"]) * np.eye(active.sum()),
                        )
                    )
                    target = np.concatenate((np.sqrt(wc) * residual[mask], np.zeros(active.sum())))
                    beta = lstsq(augmented, target)[0]
                    saved = checkpoint["sequence"][condition]
                    coefficient_error = float(np.max(abs(beta - saved["beta"])))
                    max_coefficient_error = max(max_coefficient_error, coefficient_error)
                    assert coefficient_error < 1e-9
                    np.testing.assert_allclose(center, saved["center"], atol=1e-10)
                    use = c[evaluate] == condition
                    pred[use] += (evaluation[use] - center) @ beta
            population = "train" if fit_name == "full" else "inner_cv"
            saved = pq.read_table(root / f"{variant}-{population}.parquet").to_pydict()
            np.testing.assert_allclose(saved["target_log10_ratio"], y, atol=0)
            np.testing.assert_allclose(saved["weight"], w, atol=1e-15)
            assert saved["fold"] == fold.tolist()
            assert saved["measurement_id"] == [r["measurement_id"] for r in rows]
            difference = np.max(abs(pred - np.array(saved["prediction_log10_ratio"])[evaluate]))
            assert difference < 1e-9
            max_prediction_error = max(max_prediction_error, float(difference))
            n_predictions += len(pred)
        for population in ["train", "inner_cv"]:
            table = pq.read_table(root / f"{variant}-{population}.parquet").to_pydict()
            p = np.array(table["prediction_log10_ratio"])
            residual = p - y
            mse = float(np.average(residual**2, weights=w))
            variance = float(np.average((y - np.average(y, weights=w)) ** 2, weights=w))
            expected = {
                "mse": mse,
                "rmse": math.sqrt(mse),
                "mae": float(np.average(abs(residual), weights=w)),
                "r2": 1 - mse / variance,
                "pearson": float(pearsonr(y, p).statistic) if np.ptp(p) > 1e-14 else None,
                "spearman": float(spearmanr(y, p).statistic) if np.ptp(p) > 1e-14 else None,
            }
            for name, value in expected.items():
                actual = summary["metrics"][variant][population][name]
                assert (
                    value is None
                    and actual is None
                    or value is not None
                    and abs(value - actual) < 1e-10
                )
    best_control = min(
        [v for v in checkpoints if not v.startswith("sequence")],
        key=lambda v: summary["metrics"][v]["inner_cv"]["rmse"],
    )
    best_sequence = min(
        [v for v in checkpoints if v.startswith("sequence")],
        key=lambda v: summary["metrics"][v]["inner_cv"]["rmse"],
    )
    improvement = (
        1
        - summary["metrics"][best_sequence]["inner_cv"]["rmse"]
        / summary["metrics"][best_control]["inner_cv"]["rmse"]
    )
    prediction = np.array(
        pq.read_table(root / f"{best_sequence}-inner_cv.parquet")["prediction_log10_ratio"]
    )
    min_sd = min(
        np.sqrt(
            np.average(
                (prediction[c == cond] - np.average(prediction[c == cond], weights=w[c == cond]))
                ** 2,
                weights=w[c == cond],
            )
        )
        for cond in set(c)
    )
    assert summary["development_gate_pass"] == bool(improvement >= 0.02 and min_sd > 0.001)
    assert not summary["outer_validation_test_scored"]
    result = {
        "status": "passed",
        "n_predictions_reconstructed": n_predictions,
        "max_prediction_error": max_prediction_error,
        "max_coefficient_error": max_coefficient_error,
        "best_control": best_control,
        "best_sequence": best_sequence,
        "relative_improvement": improvement,
        "development_gate_pass": summary["development_gate_pass"],
        "seconds": time.perf_counter() - start,
        "peak_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**3,
    }
    (args.experiment / "results/independent-fit-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
