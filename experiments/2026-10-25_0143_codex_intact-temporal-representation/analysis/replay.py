"""Independent fold-local least squares, physical outputs and metric replay."""

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "reproduce/launch.json").read_text())
rows = pq.read_table(ROOT / "results/changes.parquet").to_pylist()
predicted = pq.read_table(ROOT / "results/predictions.parquet").to_pylist()
inner_predictions = pq.read_table(ROOT / "results/inner-predictions.parquet").to_pylist()
direct = json.loads((ROOT / "results/direct-fits.json").read_text())
physical = json.loads((ROOT / "results/physical-fits.json").read_text())
evaluation = json.loads((ROOT / "results/evaluation.json").read_text())
AA = "ACDEFGHIKLMNPQRSTVWY"


def balanced(subset):
    seq = sorted({r["peptide_sequence"] for r in subset})
    result = np.zeros(len(subset))
    for sequence in seq:
        times = sorted({r["time_minutes"] for r in subset if r["peptide_sequence"] == sequence})
        for time in times:
            indexes = [i for i, r in enumerate(subset) if r["peptide_sequence"] == sequence and r["time_minutes"] == time]
            result[indexes] = 1 / (len(seq) * len(times) * len(indexes))
    assert abs(result.sum() - 1) < 1e-12
    return result


def raw(sequence, representation):
    counts = Counter(sequence)
    result = [np.log(len(sequence))]
    if representation == "sequence":
        result.extend(counts[aa] / len(sequence) for aa in AA)
        result.extend(int(sequence.startswith(aa)) for aa in AA)
        result.extend(int(sequence.endswith(aa)) for aa in AA)
    return np.array(result)


def independent_design(subset, transform):
    values = np.stack([raw(r["peptide_sequence"], transform["representation"]) for r in subset])
    z = (values - transform["center"]) / transform["scale"]
    z[:, transform["constant_columns"]] = 0
    projected = z @ np.array(transform["basis"])
    elapsed = np.array([r["time_minutes"] - r["baseline_time"] for r in subset]) / 80
    return np.column_stack([elapsed, elapsed[:, None] * projected])


def independent_physical(subset, parameters, kind, alpha):
    parent = subset[0]["parent_sequence"]
    amino_acids = sorted(set(parent[:-1]))
    categories = 1e-7 + (1 - 1e-7) * expit(parameters[:-1])
    rates = np.repeat(categories[0], len(parent) - 1) if kind == "uniform" else np.array([categories[amino_acids.index(a)] for a in parent[:-1]])
    tau = .001 + (120 - .001) * expit(parameters[-1])
    left = np.array([r["start_1based"] - 1 for r in subset])
    right = np.array([r["end_1based"] for r in subset])
    t = np.array([r["time_minutes"] for r in subset]) + tau
    baseline = np.array([r["baseline_time"] for r in subset]) + tau
    prefix = np.r_[0., np.cumsum(rates)]
    values = (t - baseline) * (prefix[left] + prefix[-1] - prefix[right - 1])
    for selected, bond in ((left > 0, left - 1), (right < len(parent), right - 1)):
        rate = rates[bond[selected]]
        values[selected] += np.log(-np.expm1(-t[selected] * rate)) - np.log(-np.expm1(-baseline[selected] * rate))
    values /= math.log(10)
    y = np.array([r["target"] for r in subset])
    mse = float(balanced(subset) @ (values - y) ** 2)
    penalty = alpha * np.mean((np.log(categories) - np.log(categories).mean()) ** 2) if kind == "residue" else 0.
    return values, mse + penalty, mse


# Independent baseline/label/fold reconstruction from original training files.
prior = Path(config["prior_family"])
original = pq.read_table(prior / "results/data/train/covariates.parquet").to_pylist()
labels = {r["assignment_id"]: r["target"] for r in pq.read_table(prior / "results/data/train/outcomes.parquet").to_pylist()}
original_by_sequence = defaultdict(list)
for row in original:
    original_by_sequence[row["peptide_sequence"]].append(row)
for row in rows:
    old = original_by_sequence[row["peptide_sequence"]]
    first = min(r["time_minutes"] for r in old)
    baseline_ids = [r["assignment_id"] for r in old if r["time_minutes"] == first]
    baseline = sum(labels[i] for i in baseline_ids) / len(baseline_ids)
    assert set(baseline_ids) == set(row["baseline_assignment_ids"])
    assert row["baseline_time"] == first < row["time_minutes"] <= 80
    assert abs(row["baseline_target"] - baseline) < 1e-12
    assert row["elapsed"] == (row["time_minutes"] - first) / 80
    assert abs(row["target"] - (labels[row["assignment_id"]] - baseline)) < 1e-12
    for name, n_folds in (("outer", 5), ("inner", 3)):
        digest = hashlib.sha256((config[name + "_salt"] + row["peptide_sequence"]).encode()).digest()
        assert int.from_bytes(digest, "big") % n_folds == row[name + "_fold"]

arms = {"persistence", "shared_slope", "length", "sequence", "physical_uniform",
        "physical_residue_penalized", "physical_residue_unpenalized"}
assert {r["arm"] for r in predicted} == arms
by_id = {r["assignment_id"]: r for r in rows}
assert len(by_id) == len(rows)
assert len(predicted) == 35 * len(rows)
assert len(direct) == 135 and len(physical) == 45
for record in inner_predictions:
    source = by_id[record["assignment_id"]]
    assert source["outer_fold"] != record["outer_fit"]
    assert source["inner_fold"] == record["inner_fit"]
    assert all(record[k] == v for k, v in source.items() if k != "weight")
for fold in range(5):
    for arm in arms:
        subset = [r for r in predicted if r["outer_fit"] == fold and r["arm"] == arm]
        assert len(subset) == len(rows) and {r["assignment_id"] for r in subset} == set(by_id)
        for record in subset:
            source = by_id[record["assignment_id"]]
            assert record["role"] == ("validation" if source["outer_fold"] == fold else "training")
            assert all(record[k] == v for k, v in source.items() if k != "weight")
            assert record["predicted_source_target"] == record["prediction"] + source["baseline_target"]
            if arm == "persistence":
                assert record["prediction"] == 0
        for role in ("training", "validation"):
            partition = [r for r in subset if r["role"] == role]
            assert np.max(np.abs(balanced(partition) - [r["weight"] for r in partition])) < 1e-12

max_ridge_error = 0.
for fitted in direct:
    training = [r for r in rows if r["outer_fold"] != fitted["outer_fold"]]
    if fitted["arm"] == "shared_slope":
        assert sorted({r["peptide_sequence"] for r in training}) == fitted["training_sequences"]
        w = balanced(training)
        x = np.array([r["elapsed"] for r in training])
        y = np.array([r["target"] for r in training])
        slope = (w * x) @ y / ((w * x) @ x)
        assert abs(slope - fitted["slope"]) < 1e-12
        targets = [r for r in predicted if r["outer_fit"] == fitted["outer_fold"] and r["arm"] == "shared_slope"]
        assert max(abs(slope * r["elapsed"] - r["prediction"]) for r in targets) < 1e-12
        continue
    if fitted["role"] == "inner":
        training = [r for r in training if r["inner_fold"] != fitted["inner_fold"]]
        targets = [r for r in inner_predictions if r["outer_fit"] == fitted["outer_fold"] and
                   r["inner_fit"] == fitted["inner_fold"] and r["arm"] == fitted["arm"] and r["alpha"] == fitted["alpha"]]
        expected_ids = {r["assignment_id"] for r in rows if r["outer_fold"] != fitted["outer_fold"] and r["inner_fold"] == fitted["inner_fold"]}
        assert {r["assignment_id"] for r in targets} == expected_ids and len(targets) == len(expected_ids)
        assert np.max(np.abs(balanced(targets) - [r["weight"] for r in targets])) < 1e-12
    else:
        targets = [r for r in predicted if r["outer_fit"] == fitted["outer_fold"] and r["arm"] == fitted["arm"]]
    transform = fitted["transform"]
    assert sorted({r["peptide_sequence"] for r in training}) == transform["training_sequences"]
    if fitted["role"] == "inner":
        assert not set(transform["training_sequences"]) & {r["peptide_sequence"] for r in targets}
    x = np.stack([raw(sequence, fitted["arm"]) for sequence in transform["training_sequences"]])
    assert np.max(np.abs(x.mean(0) - transform["center"])) < 1e-12
    constant = np.ptp(x, axis=0) == 0
    scale = np.std(x, axis=0)
    scale[(scale == 0) | constant] = 1
    assert np.array_equal(constant, transform["constant_columns"])
    assert np.max(np.abs(scale - transform["scale"])) < 1e-12
    standardized = (x - x.mean(0)) / scale
    standardized[:, constant] = 0
    _, singular, right = np.linalg.svd(standardized, full_matrices=False)
    retained = singular**2 / len(x) > max(singular[0]**2 / len(x), 1.) * 1e-10
    independent_basis = right[retained].T
    stored_basis = np.array(transform["basis"])
    assert int(retained.sum()) == transform["rank"]
    assert np.max(np.abs(independent_basis @ independent_basis.T - stored_basis @ stored_basis.T)) < 1e-8
    a = independent_design(training, transform)
    w = balanced(training)
    penalty = np.zeros(a.shape[1])
    if transform["rank"]:
        penalty[1:] = fitted["alpha"] / transform["rank"]
    augmented = np.vstack([a * np.sqrt(w)[:, None], np.diag(np.sqrt(penalty))])
    target = np.r_[np.sqrt(w) * [r["target"] for r in training], np.zeros(a.shape[1])]
    coefficient = np.linalg.lstsq(augmented, target, rcond=None)[0]
    values = independent_design(targets, transform) @ coefficient
    error = float(np.max(np.abs(values - [r["prediction"] for r in targets])))
    max_ridge_error = max(max_ridge_error, error)
    assert error < 1e-8

# Reproduce alpha selection from the complete inner-validation prediction sets.
for fitted in [r for r in direct if r.get("role") == "selected_outer"]:
    scores = {}
    for alpha in config["alpha_grid"]:
        subset = [r for r in inner_predictions if r["outer_fit"] == fitted["outer_fold"] and
                  r["arm"] == fitted["arm"] and r["alpha"] == alpha]
        scores[alpha] = float(balanced(subset) @ np.array([(r["prediction"] - r["target"]) ** 2 for r in subset]))
    assert min(config["alpha_grid"], key=lambda a: scores[a]) == fitted["alpha"]
    assert max(abs(scores[a] - fitted["inner_mse"][str(a)]) for a in scores) < 1e-12

max_physical_error = max_objective_error = max_gradient_error = 0.
for fitted in physical:
    training = [r for r in rows if r["outer_fold"] != fitted["outer_fold"]]
    assert sorted({r["peptide_sequence"] for r in training}) == fitted["training_sequences"]
    path = ROOT / f"results/physical/fold{fitted['outer_fold']}-{fitted['arm']}-start{fitted['start_number']}"
    history = np.load(path / "optimizer-history.npz")
    n = len(history["final"])
    spec = config["starts"][fitted["start_number"]]
    rate = (spec["rate"] - 1e-7) / (1 - 1e-7)
    tau = (spec["tau"] - .001) / (120 - .001)
    expected_initial = [math.log(rate / (1 - rate))] * (n - 1) + [math.log(tau / (1 - tau))]
    assert np.max(np.abs(history["initial"] - expected_initial)) < 1e-12
    stored = pq.read_table(path / "training-predictions.parquet").to_pylist()
    assert [r["assignment_id"] for r in training] == [r["assignment_id"] for r in stored]
    values, objective, mse = independent_physical(training, history["final"], fitted["kind"], fitted["alpha"])
    error = float(np.max(np.abs(values - [r["prediction"] for r in stored])))
    max_physical_error = max(max_physical_error, error)
    assert error < 1e-10 and abs(mse - fitted["mse"]) < 1e-10
    assert abs(objective - fitted["objective"]) < 1e-10
    for recorded in history["evaluations"]:
        value = independent_physical(training, recorded[4:4 + n], fitted["kind"], fitted["alpha"])[1]
        max_objective_error = max(max_objective_error, abs(value - recorded[1]))
    finite = np.zeros(n)
    for j in range(n):
        step = np.zeros(n); step[j] = 1e-4
        plus = independent_physical(training, history["final"] + step, fitted["kind"], fitted["alpha"])[1]
        minus = independent_physical(training, history["final"] - step, fitted["kind"], fitted["alpha"])[1]
        finite[j] = (plus - minus) / 2e-4
    max_gradient_error = max(max_gradient_error, float(np.max(np.abs(finite - history["gradient"]))))
    assert np.max(np.abs(history["final"] - fitted["raw_parameters"])) == 0
    assert np.max(np.abs(history["gradient"] - fitted["raw_gradient"])) == 0
    assert np.max(np.abs(np.abs(history["evaluations"][:, 4 + n:]).sum(0) - fitted["gradient_absolute_totals"])) < 1e-10
    assert bool(np.all(history["final"] != history["initial"])) == fitted["all_parameters_changed"]
assert max_objective_error < 1e-8 and max_gradient_error < 1e-6

for fold in range(5):
    for arm in ("physical_uniform", "physical_residue_penalized", "physical_residue_unpenalized"):
        candidates = [r for r in physical if r["outer_fold"] == fold and r["arm"] == arm]
        selected = min(candidates, key=lambda r: r["objective"])
        targets = [r for r in predicted if r["outer_fit"] == fold and r["arm"] == arm]
        assert all(r["selected_start"] == selected["start_number"] for r in targets)
        values = independent_physical(targets, np.array(selected["raw_parameters"]), selected["kind"], selected["alpha"])[0]
        error = float(np.max(np.abs(values - [r["prediction"] for r in targets])))
        max_physical_error = max(max_physical_error, error)
        assert error < 1e-10
        assert all(r["optimizer_success"] == selected["success"] for r in targets)


def correlation(a, b):
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    a = a - np.mean(a); b = b - np.mean(b)
    return float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))


max_metric_error = 0.
strata = pq.read_table(ROOT / "results/strata.parquet").to_pylist()
assert pq.read_table(ROOT / "results/metrics.parquet").to_pylist() == evaluation["all_metrics"]
for record in evaluation["all_metrics"] + strata:
    subset = [r for r in predicted if r["arm"] == record["arm"] and
              (("peptide_sequence" in record and r["role"] == "validation" and r["peptide_sequence"] == record["peptide_sequence"]) or
               (record.get("scope") == "pooled_outer_development" and r["role"] == "validation") or
               (r["role"] == record.get("scope") and r["outer_fit"] == record["outer_fold"]))]
    w = balanced(subset)
    y = np.array([r["target"] for r in subset]); p = np.array([r["prediction"] for r in subset])
    mse = float(np.sum(w * (y - p) ** 2)); variance = float(np.sum(w * (y - np.sum(w * y)) ** 2))
    values = {"mse": mse, "rmse": np.sqrt(mse), "mae": np.sum(w * np.abs(y - p)),
              "r2": 1 - mse / variance if variance else None,
              "pearson": correlation(y, p), "spearman": correlation(rankdata(y), rankdata(p)),
              "prediction_sd": np.sqrt(np.sum(w * (p - np.sum(w * p)) ** 2)), "target_sd": np.sqrt(variance)}
    assert len(subset) == record["n_observations"]
    assert len({r["peptide_sequence"] for r in subset}) == record["n_sequences"]
    for key, value in values.items():
        if value is None:
            assert record[key] is None
        else:
            error = abs(float(value) - record[key]); max_metric_error = max(max_metric_error, error)
            assert error < 1e-10, (record["arm"], key, error)

# Independently aggregate per-peptide errors, replay paired draws and fixed gates.
rng = np.random.default_rng(31415)
bootstrap_draws = np.load(ROOT / "results/bootstrap.npz")
max_bootstrap_error = 0.
for name, candidate, baseline in (("sequence_vs_length", "sequence", "length"),
                                  ("unpenalized_vs_penalized", "physical_residue_unpenalized", "physical_residue_penalized")):
    groups = defaultdict(dict)
    for record in predicted:
        if record["role"] == "validation" and record["arm"] in (candidate, baseline):
            groups[record["peptide_sequence"]].setdefault(record["arm"], []).append(record)
    per_peptide = []
    for sequence in sorted(groups):
        errors = []
        for arm in (candidate, baseline):
            records = groups[sequence][arm]
            by_time = defaultdict(list)
            for record in records:
                by_time[record["time_minutes"]].append((record["prediction"] - record["target"])**2)
            errors.append(np.mean([np.mean(v) for v in by_time.values()]))
        per_peptide.append(errors)
    array = np.array(per_peptide)
    draws = []
    for _ in range(1000):
        selected = rng.integers(len(array), size=len(array))
        a, b = array[selected].mean(0)
        draws.append(100 * (1 - math.sqrt(a / b)))
    error = float(np.max(np.abs(draws - bootstrap_draws[name])))
    max_bootstrap_error = max(max_bootstrap_error, error)
    assert error < 1e-10
    assert np.max(np.abs(np.quantile(draws, [.025, .975]) - evaluation["bootstrap"][name]["gain_percent_95_interval"])) < 1e-10
lookup = {r["arm"]: r for r in evaluation["all_metrics"] if r["scope"] == "pooled_outer_development"}
seq = lookup["sequence"]
sequence_gain = 100 * (1 - seq["rmse"] / lookup["length"]["rmse"])
physical_gain = 100 * (1 - lookup["physical_residue_unpenalized"]["rmse"] / lookup["physical_residue_penalized"]["rmse"])
assert abs(sequence_gain - evaluation["sequence_gain_percent"]) < 1e-12
assert abs(physical_gain - evaluation["unpenalized_gain_percent"]) < 1e-12
gates = {"rmse_gain_at_least_5pct": sequence_gain >= 5,
         "positive_pearson": seq["pearson"] is not None and seq["pearson"] > 0,
         "positive_spearman": seq["spearman"] is not None and seq["spearman"] > 0,
         "prediction_spread": seq["prediction_sd"] >= .1 * seq["target_sd"]}
assert gates == evaluation["sequence_gates"]
assert all(gates.values()) == evaluation["sequence_gate_passed"]
assert (physical_gain >= 5) == evaluation["unpenalized_improvement_gate_passed"]
assert all(r["optimizer_success"] for r in predicted if r["arm"] == "physical_residue_unpenalized") == evaluation["unpenalized_selected_optimizers_successful"]
for name, expected in config["inputs"].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
result = {"passed": True, "n_physical_starts": len(physical), "n_direct_records": len(direct),
          "max_ridge_prediction_error": max_ridge_error, "max_physical_prediction_error": max_physical_error,
          "max_recorded_objective_error": max_objective_error, "max_final_gradient_error": max_gradient_error,
          "max_metric_error": max_metric_error, "fold_local_training_and_initializers_verified": True,
          "max_bootstrap_error": max_bootstrap_error, "n_metrics_verified": len(evaluation["all_metrics"]) + len(strata),
          "baseline_predictions_and_selection_verified": True,
          "original_outer_outcomes_read": False,
          "scope_limit": "No independent replay of L-BFGS internal updates; forward objectives, final gradients, overall/stratified metrics, paired bootstrap and gates are checked."}
(ROOT / "results/independent-replay.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
