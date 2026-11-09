"""Independent NumPy/SciPy prediction, ridge and archived-byte metric reconstruction."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import lstsq, svd
from scipy.special import logsumexp
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]


def measure(frame):
    w = frame.weight.to_numpy()
    w = w / w.sum()
    y = frame.target_centered_log10.to_numpy()
    z = frame.prediction_centered_log10.to_numpy()
    ymean = np.average(y, weights=w)
    variance = np.average((y - ymean) ** 2, weights=w)
    mse = np.average((y - z) ** 2, weights=w)
    q = frame.predicted_fraction.to_numpy()
    p = frame.target_fraction.to_numpy()
    # Peptide weights are group mass divided by group size; KL is not peptide-averaged.
    counts = frame.groupby("group_id", sort=False).group_id.transform("size").to_numpy()
    kl = float(np.sum(w * counts * q * (np.log(q) - np.log(p))))
    variable = np.ptp(z) > 1e-14
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.average(abs(y - z), weights=w)),
        "r2": float(1 - mse / variance),
        "pearson": float(pearsonr(y, z).statistic) if variable else None,
        "spearman": float(spearmanr(y, z).statistic) if variable else None,
        "reverse_kl": kl,
        "prediction_sd": float(np.sqrt(np.average((z - np.average(z, weights=w)) ** 2, weights=w))),
        "target_sd": float(np.sqrt(variance)),
    }


def main():
    seal = json.loads((ROOT / "results/prediction-seal.json").read_text())
    for name, expected in seal.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    rows = json.loads((ROOT / "results/covariates.json").read_text())
    weights = np.array([r["weight"] for r in rows])
    raw = []
    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    for row in rows:
        s = row["peptide_sequence"]
        counts = Counter(s)
        raw.append(
            [math.log(len(s))]
            + [counts[a] / len(s) for a in alphabet]
            + [float(s[0] == a) for a in alphabet]
            + [float(s[-1] == a) for a in alphabet]
        )
    raw = np.array(raw)
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    all_groups = {}
    for i, r in enumerate(rows):
        all_groups.setdefault(r["group_id"], []).append(i)
    fitting = json.loads((ROOT / "results/models.json").read_text())
    designs = {}
    maximum_ridge_error = 0.0
    for family, basis in fitting["bases"].items():
        x = raw[:, basis["indices"]]
        mean = np.average(x[train], axis=0, weights=weights[train])
        scale = np.sqrt(np.average((x[train] - mean) ** 2, axis=0, weights=weights[train]))
        scale[scale <= 1e-14] = 1
        np.testing.assert_allclose(mean, basis["mean"], rtol=0, atol=1e-12)
        np.testing.assert_allclose(scale, basis["scale"], rtol=0, atol=1e-12)
        x = (x - mean) / scale
        for indices in all_groups.values():
            x[indices] -= x[indices].mean(0)
        _, singular, vt = svd(
            np.sqrt(weights[train, None]) * x[train], full_matrices=False, lapack_driver="gesvd"
        )
        supported = vt[singular > singular[0] * 1e-10].T
        projection = np.array(basis["projection"])
        assert supported.shape == projection.shape
        np.testing.assert_allclose(
            projection @ projection.T, supported @ supported.T, rtol=0, atol=1e-10
        )
        designs[family] = x @ projection
    labels = {
        split: json.loads((ROOT / f"results/{split}-labels.json").read_text())
        for split in ("train", "val", "test")
    }
    y = np.array([labels["train"][rows[i]["measurement_id"]]["y"] for i in train])
    saved = np.load(ROOT / "results/covariate-predictions.npz")
    assert np.count_nonzero(saved["uniform"]) == 0
    maximum_prediction = 0.0
    for name, model in fitting["models"].items():
        x = designs[model["basis"]]
        beta = np.array(model["beta"])
        if "alpha" in model:
            matrix = np.vstack(
                (
                    x[train] * np.sqrt(weights[train, None]),
                    math.sqrt(model["alpha"] / len(beta)) * np.eye(len(beta)),
                )
            )
            rhs = np.r_[y * np.sqrt(weights[train]), np.zeros(len(beta))]
            independent = lstsq(matrix, rhs, lapack_driver="gelsd")[0]
            maximum_ridge_error = max(maximum_ridge_error, float(abs(independent - beta).max()))
            assert abs(independent - beta).max() <= 1e-10
        error = float(abs(x @ beta - saved[name]).max())
        maximum_prediction = max(maximum_prediction, error)
        assert error <= 1e-10
    frame = pd.read_csv(ROOT / "results/predictions.csv.gz", float_precision="round_trip")
    all_metrics = json.loads((ROOT / "results/metrics.json").read_text())
    maximum_metric = 0.0
    for record in all_metrics:
        split, name = record["split"], record["model"]
        sub = frame[(frame.split == split) & (frame.model == name)]
        indices = np.array([i for i, r in enumerate(rows) if r["split"] == split])
        local = [rows[i] for i in indices]
        assert list(sub.measurement_id) == [r["measurement_id"] for r in local]
        np.testing.assert_array_equal(sub.weight, [r["weight"] for r in local])
        for column, key in [("target_centered_log10", "y"), ("target_fraction", "p")]:
            np.testing.assert_array_equal(
                sub[column], [labels[split][r["measurement_id"]][key] for r in local]
            )
        predicted = saved[name][indices].copy()
        lookup = {r["measurement_id"]: i for i, r in enumerate(local)}
        q = np.zeros(len(local))
        for _, group in sub.groupby("group_id", sort=False):
            ii = np.array([lookup[key] for key in group.measurement_id])
            predicted[ii] -= predicted[ii].mean()
            logits = math.log(10) * predicted[ii]
            q[ii] = np.exp(logits - logsumexp(logits))
        np.testing.assert_allclose(predicted, sub.prediction_centered_log10, rtol=0, atol=1e-12)
        np.testing.assert_allclose(q, sub.predicted_fraction, rtol=0, atol=1e-12)
        actual = measure(sub)
        for key, value in actual.items():
            if value is None:
                assert record[key] is None
            else:
                error = abs(value - record[key])
                maximum_metric = max(maximum_metric, error)
                assert error <= 1e-10, (split, name, key, error)
    strata = json.loads((ROOT / "results/strata.json").read_text())
    indexed = {
        (split, model): sub for (split, model), sub in frame.groupby(["split", "model"], sort=False)
    }
    for record in strata:
        sub = indexed[record["split"], record["model"]]
        sub = sub[sub[record["axis"]] == record["value"]]
        assert len(sub) == record["n_observations"]
        actual = measure(sub)
        for key, value in actual.items():
            if value is None:
                assert record[key] is None
            else:
                assert abs(value - record[key]) <= 1e-10, (record["axis"], key)
    result = {
        "passed": True,
        "n_predictions": len(frame),
        "n_metric_records": len(all_metrics),
        "n_strata": len(strata),
        "maximum_prediction_error": maximum_prediction,
        "maximum_ridge_coefficient_error": maximum_ridge_error,
        "maximum_overall_metric_error": maximum_metric,
        "optimizer_trajectories": "retained_not_independently_replayed",
        "stratum_ranks": "all_metrics_replayed_from_exact_archived_prediction_bytes",
    }
    (ROOT / "results/independent-prediction-replay.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
