"""Predeclared nested-development metrics; no original outer-test access."""

import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import pearsonr, spearmanr

from core import ROOT, dump, sequences, weights


def metrics(rows):
    y = np.array([r["target"] for r in rows])
    prediction = np.array([r["prediction"] for r in rows])
    w = weights(rows)
    mse = float(w @ (prediction - y) ** 2)
    variance = float(w @ (y - w @ y) ** 2)
    nonconstant = len(rows) >= 2 and np.ptp(y) > 0 and np.ptp(prediction) > 0
    return {"n_observations": len(rows), "n_sequences": len(sequences(rows)),
            "mse": mse, "rmse": math_sqrt(mse), "mae": float(w @ np.abs(prediction - y)),
            "r2": 1 - mse / variance if variance > 0 else None,
            "pearson": float(pearsonr(y, prediction).statistic) if nonconstant else None,
            "spearman": float(spearmanr(y, prediction).statistic) if nonconstant else None,
            "prediction_sd": math_sqrt(float(w @ (prediction - w @ prediction) ** 2)),
            "target_sd": math_sqrt(variance)}


def math_sqrt(value):
    return float(np.sqrt(value))


assert json.loads((ROOT / "results/fit-resource.json").read_text())["passed"]
fitting = json.loads((ROOT / "results/fit-summary.json").read_text())
for name, expected in fitting["output_hashes"].items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
rows = pq.read_table(ROOT / "results/predictions.parquet").to_pylist()
arms = ["persistence", "shared_slope", "length", "sequence", "physical_uniform",
        "physical_residue_penalized", "physical_residue_unpenalized"]
overall, strata = [], []
validation = {}
for arm in arms:
    cv = [r for r in rows if r["arm"] == arm and r["role"] == "validation"]
    assert len({r["assignment_id"] for r in cv}) == len(cv)
    assert all(r["outer_fit"] == r["outer_fold"] for r in cv)
    validation[arm] = cv
    overall.append({"arm": arm, "scope": "pooled_outer_development", "outer_fold": -1, **metrics(cv)})
    for fold in range(5):
        for role in ("training", "validation"):
            subset = [r for r in rows if r["arm"] == arm and r["role"] == role and r["outer_fit"] == fold]
            overall.append({"arm": arm, "scope": role, "outer_fold": fold, **metrics(subset)})
    for sequence in sequences(cv):
        subset = [r for r in cv if r["peptide_sequence"] == sequence]
        strata.append({"arm": arm, "peptide_sequence": sequence, **metrics(subset)})

lookup = {r["arm"]: r for r in overall if r["scope"] == "pooled_outer_development"}
rng = np.random.default_rng(31415)
bootstrap = {}
draws = {}
comparisons = (("sequence_vs_length", "sequence", "length"),
               ("unpenalized_vs_penalized", "physical_residue_unpenalized", "physical_residue_penalized"))
for name, candidate, baseline in comparisons:
    group_errors = []
    base_by_id = {r["assignment_id"]: r for r in validation[baseline]}
    for sequence in sequences(validation[candidate]):
        group = [r for r in validation[candidate] if r["peptide_sequence"] == sequence]
        w = weights(group)
        group_errors.append([w @ np.array([(r["prediction"] - r["target"]) ** 2 for r in group]),
                             w @ np.array([(base_by_id[r["assignment_id"]]["prediction"] - r["target"]) ** 2 for r in group])])
    error = np.array(group_errors)
    gains = []
    for _ in range(1000):
        sampled = error[rng.integers(len(error), size=len(error))].mean(0)
        gains.append(100 * (1 - np.sqrt(sampled[0] / sampled[1])))
    draws[name] = np.array(gains)
    bootstrap[name] = {"unit": "peptide_within_one_parent_not_independent_specimen", "resamples": 1000,
                       "gain_percent_95_interval": np.quantile(gains, [.025, .975]).tolist()}
np.savez_compressed(ROOT / "results/bootstrap.npz", **draws)
sequence_gain = 100 * (1 - lookup["sequence"]["rmse"] / lookup["length"]["rmse"])
physical_gain = 100 * (1 - lookup["physical_residue_unpenalized"]["rmse"] / lookup["physical_residue_penalized"]["rmse"])
sequence_metrics = lookup["sequence"]
sequence_gates = {"rmse_gain_at_least_5pct": sequence_gain >= 5,
                  "positive_pearson": sequence_metrics["pearson"] is not None and sequence_metrics["pearson"] > 0,
                  "positive_spearman": sequence_metrics["spearman"] is not None and sequence_metrics["spearman"] > 0,
                  "prediction_spread": sequence_metrics["prediction_sd"] >= .1 * sequence_metrics["target_sd"]}
physical_success = all(r["optimizer_success"] for r in validation["physical_residue_unpenalized"])
result = {"scope": "nested_development_CV_on_previously_inspected_training",
          "new_blind_biological_test": False, "all_metrics": overall, "bootstrap": bootstrap,
          "sequence_gain_percent": sequence_gain, "sequence_gates": sequence_gates,
          "sequence_gate_passed": all(sequence_gates.values()), "unpenalized_gain_percent": physical_gain,
          "unpenalized_improvement_gate_passed": physical_gain >= 5,
          "unpenalized_selected_optimizers_successful": physical_success,
          "APC_concentration_validated": False}
pq.write_table(pa.Table.from_pylist(overall), ROOT / "results/metrics.parquet")
pq.write_table(pa.Table.from_pylist(strata), ROOT / "results/strata.parquet")
dump("results/evaluation.json", result)
print(json.dumps({k: v for k, v in result.items() if k != "all_metrics"}), flush=True)
