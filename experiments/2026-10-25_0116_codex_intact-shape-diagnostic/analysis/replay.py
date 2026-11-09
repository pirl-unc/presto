"""Independent NumPy forward, finite differences and convex KKT replay."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
prior = Path(config["prior_family"])
rows = pq.read_table(prior / "results/data/train/covariates.parquet").to_pylist()
labels = pq.read_table(prior / "results/data/train/outcomes.parquet").to_pylist()
y = np.array([r["target"] for r in labels])
w = np.array([r["weight"] for r in rows])
sequences = sorted({r["peptide_sequence"] for r in rows})
index = np.array([sequences.index(r["peptide_sequence"]) for r in rows])
parent = rows[0]["parent_sequence"]
amino_acids = sorted(set(parent[:-1]))
aa_index = np.array([amino_acids.index(a) for a in parent[:-1]])
time = np.array([r["time_minutes"] for r in rows])
left = np.array([r["start_1based"] - 1 for r in rows])
right = np.array([r["end_1based"] for r in rows])
mass = np.bincount(index, weights=w)


def independent(theta, kind):
    category = 1e-7 + (1 - 1e-7) * expit(theta[:-1])
    rates = np.full(len(parent) - 1, category[0]) if kind == "uniform" else category[aa_index]
    tau = .001 + (120 - .001) * expit(theta[-1])
    exposure = time + tau
    result = np.zeros(len(rows))
    for j, (start, end, u) in enumerate(zip(left, right, exposure)):
        outside = rates[:start].sum() + rates[end - 1:].sum()
        value = u * outside
        if start:
            value += math.log(-math.expm1(-u * rates[start - 1]))
        if end < len(parent):
            value += math.log(-math.expm1(-u * rates[end - 1]))
        result[j] = value / math.log(10)
    offsets = np.bincount(index, weights=w * (y - result)) / mass
    prediction = result + offsets[index]
    mse = float(w @ ((prediction - y) ** 2))
    penalty = .01 * np.mean((np.log(category) - np.log(category).mean()) ** 2) if kind == "residue" else 0.
    return mse + penalty, mse, prediction, offsets


summary = json.loads((ROOT / "results/summary.json").read_text())
max_prediction_error = max_gradient_error = max_objective_error = 0.
records = []
for fit in summary["fits"]:
    path = ROOT / "results" / fit["condition"]
    checkpoint = np.load(path / "optimizer-history.npz")
    predictions = pq.read_table(path / "predictions.parquet").to_pylist()
    assert [r["assignment_id"] for r in predictions] == [r["assignment_id"] for r in rows]
    theta = checkpoint["final"]
    objective, mse, prediction, offsets = independent(theta, fit["kind"])
    error = max(float(np.max(np.abs(prediction - [r["prediction"] for r in predictions]))),
                float(np.max(np.abs(offsets - checkpoint["offsets"]))))
    assert error <= 1e-10
    assert abs(mse - fit["mse"]) <= 1e-12
    assert abs(objective - fit["objective"]) <= 1e-12
    n = len(theta)
    for evaluation in checkpoint["evaluations"]:
        reconstructed = independent(evaluation[4:4 + n], fit["kind"])[0]
        max_objective_error = max(max_objective_error, abs(reconstructed - evaluation[1]))
    finite = np.zeros(n)
    for j in range(n):
        delta = np.zeros(n)
        delta[j] = 1e-4
        finite[j] = (independent(theta + delta, fit["kind"])[0] - independent(theta - delta, fit["kind"])[0]) / 2e-4
    gradient_error = float(np.max(np.abs(finite - checkpoint["gradient"])))
    assert gradient_error <= 1e-6
    max_prediction_error = max(max_prediction_error, error)
    max_gradient_error = max(max_gradient_error, gradient_error)
    records.append({"condition": fit["condition"], "n_objective_evaluations": len(checkpoint["evaluations"]),
                    "max_prediction_error": error, "max_final_gradient_error": gradient_error})
assert max_objective_error <= 1e-10

shape_rows = pq.read_table(ROOT / "results/shape-predictions.parquet").to_pylist()
curves = json.loads((ROOT / "results/shape-curves.json").read_text())
max_kkt = 0.
for curve in curves:
    times = np.array(curve["times"])
    means = np.array(curve["replicate_means"])
    weights = np.array(curve["time_weight"])
    weights /= weights.sum()
    # Construct inequalities without importing the production shape solver.
    increasing = np.zeros((len(times) - 1, len(times)))
    for j in range(len(times) - 1):
        increasing[j, j] = -1 / (times[j + 1] - times[j])
        increasing[j, j + 1] = -increasing[j, j]
    for name, info in (("increasing", "monotone_kkt"), ("increasing_concave", "concave_kkt")):
        a = increasing if name == "increasing" else np.vstack([increasing, increasing[:-1] - increasing[1:]])
        z = np.array(curve[name])
        lam = np.array(curve[info]["multipliers"])
        residual = max(float(np.max(np.abs(weights * (z - means) - a.T @ lam))),
                       float(np.max(np.abs(lam * (a @ z)))), max(0., float(-np.min(a @ z))),
                       max(0., float(-np.min(lam))))
        assert residual <= 1e-9
        max_kkt = max(max_kkt, residual)
    subset = [j for j, r in enumerate(rows) if r["peptide_sequence"] == curve["peptide_sequence"]]
    for t, mean in zip(times, means):
        cells = [j for j in subset if time[j] == t]
        assert abs(mean - np.average(y[cells], weights=w[cells])) <= 1e-12
    for j in subset:
        for name in ("increasing", "increasing_concave"):
            assert abs(shape_rows[j][name] - curve[name][list(times).index(time[j])]) <= 1e-12
for name, expected in summary["shape_mse"].items():
    p = np.array([r[name] for r in shape_rows])
    assert abs(float(w @ ((p - y) ** 2)) - expected) <= 1e-12
for kind, decision in summary["decisions"].items():
    fits = [f for f in summary["fits"] if f["kind"] == kind]
    gain = max(100 * (1 - f["mse"] / f["original_adam_mse"]) for f in fits)
    fraction = (summary["shape_mse"]["increasing_concave"] - summary["shape_mse"]["free"]) / fits[0]["original_adam_mse"]
    assert decision["optimizer_material_at_5pct"] == (gain >= 5)
    assert decision["shape_material_at_10pct"] == (fraction >= .1)
for name, expected in config["inputs"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
result = {"passed": True, "conditions": records, "n_training_predictions": 6 * len(rows),
          "max_prediction_error": max_prediction_error, "max_objective_replay_error": max_objective_error,
          "max_final_finite_difference_error": max_gradient_error, "max_shape_kkt_residual": max_kkt,
          "selection_thresholds_replayed": True, "heldout_outcomes_read": False,
          "limitations": "L-BFGS internal line-search/update implementation is not independently replayed; every recorded objective and final output are."}
(ROOT / "results/independent-replay.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
