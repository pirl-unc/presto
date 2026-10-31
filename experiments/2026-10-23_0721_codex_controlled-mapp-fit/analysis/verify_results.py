"""Independent NumPy/scalar model, Adam, source, metric and bootstrap replay."""

import gzip
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
started = time.perf_counter()
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.set_default_dtype(torch.float64)


def close(a, b, tolerance=1e-10):
    assert np.allclose(a, b, rtol=tolerance, atol=tolerance), (a, b)


def ranks(values):
    result = np.zeros(len(values))
    order = sorted(range(len(values)), key=lambda i: values[i])
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        for i in order[start:end]:
            result[i] = (start + end - 1) / 2
        start = end
    return result


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    left, right = a - a.mean(), b - b.mean()
    return float(left @ right / np.sqrt((left @ left) * (right @ right)))


def raw_features(rows):
    output = []
    for row in rows:
        seq = row["peptide_sequence"]
        output.append(
            [
                math.log(len(seq)),
                *(
                    math.fsum(1 for residue in seq if residue in set(group)) / len(seq)
                    for group in ("AVILMFYW", "KR", "DE")
                ),
            ]
        )
    return np.asarray(output)


def arrays(rows, mean, scale):
    x = (raw_features(rows) - mean) / scale
    u = np.array([r["untreated_mean_intensity"] for r in rows])
    e = np.array([r["epoxomicin_mean_intensity"] for r in rows])
    return x, u, e


def forward(weights, x, u):
    logits = x @ weights + np.log(u / u.sum())
    return logits - np.logaddexp.reduce(logits)


def gradient(weights, x, u, e):
    logp = forward(weights, x, u)
    p, cost = np.exp(logp), logp - np.log(e / e.sum())
    jacobian = x - p @ x
    return jacobian.T @ (p * cost) + CFG["policy_l2_half_weight"] * weights


rows = json.loads((Path(CFG["contrast_output"]) / "examples.json").read_text())
source_rows = {r["peptide_row_id"]: r for r in rows}
models = json.loads((R / "models.json").read_text())
all_metrics = json.loads((R / "metrics.json").read_text())
with gzip.open(R / "predictions.jsonl.gz", "rt") as stream:
    predictions = [json.loads(line) for line in stream]
assert len(predictions) == 870 and len(all_metrics) == 30 and len(models) == 10
train = [r for r in rows if r["split"] == "train"]
raw = raw_features(train)
mean, scale = raw.mean(0), np.maximum(raw.std(0), 1e-8)
scaling = json.loads((R / "feature-scaling.json").read_text())
close(mean, scaling["mean"])
close(scale, scaling["population_std"])
assert np.linalg.matrix_rank((raw - mean) / scale) == scaling["training_rank"] == 4
x, u, e = arrays(train, mean, scale)
beta = math.log(math.fsum(e) / math.fsum(u))
max_replay_error = 0.0
work = {}
for model in models:
    if model["kind"] != "policy":
        continue
    variant = model["variant"]
    directory = R / "runs" / variant
    checkpoints = {
        step: torch.load(directory / f"checkpoint-{step}.pt", weights_only=False)
        for step in CFG["probe_updates"]
    }
    weights = np.zeros(4)
    m, v = np.zeros(4), np.zeros(4)
    generator = torch.Generator().manual_seed(model["seed"])
    count = model["sample_count"]
    history = {
        h["update"]: h for h in json.loads((directory / "history.json").read_text())
    }
    for step in range(CFG["steps"] + 1):
        if step:
            logp = forward(weights, x, u)
            if count is None:
                grad = gradient(weights, x, u, e)
            else:
                p = np.exp(logp)
                indices = torch.multinomial(
                    torch.from_numpy(p), count, replacement=True, generator=generator
                ).numpy()
                costs = logp[indices] - np.log(e[indices] / e.sum())
                advantages = costs - (costs.sum() - costs) / (count - 1)
                grad = (x[indices] - p @ x).T @ advantages / count + CFG[
                    "policy_l2_half_weight"
                ] * weights
            b1, b2 = CFG["adam"]["betas"]
            m, v = b1 * m + (1 - b1) * grad, b2 * v + (1 - b2) * grad**2
            weights -= (
                CFG["adam"]["lr"]
                * (m / (1 - b1**step))
                / (np.sqrt(v / (1 - b2**step)) + CFG["adam"]["eps"])
            )
        if step in history:
            logp = forward(weights, x, u)
            kl = np.exp(logp) @ (logp - np.log(e / e.sum()))
            penalty = CFG["policy_l2_half_weight"] / 2 * weights @ weights
            close(history[step]["train_reverse_kl"], kl)
            close(history[step]["l2_penalty"], penalty)
            close(history[step]["objective"], kl + penalty)
            close(history[step]["weights"], weights)
        if step in checkpoints:
            checkpoint = checkpoints[step]
            expected = checkpoint["weights"].numpy()
            max_replay_error = max(
                max_replay_error, float(np.max(np.abs(expected - weights)))
            )
            close(weights, expected)
            close(checkpoint["beta"], beta)
            close(checkpoint["feature_mean"], mean)
            close(checkpoint["feature_scale"], scale)
            assert checkpoint["update"] == step
            if step:
                state = checkpoint["optimizer"]["state"][0]
                close(state["exp_avg"].numpy(), m)
                close(state["exp_avg_sq"].numpy(), v)
                assert state["step"].item() == step
                assert torch.equal(checkpoint["sampling_rng"], generator.get_state())
    close(model["weights"], weights)
    close(model["beta"], beta)
    assert model["all_four_weights_changed"] and all(model["gradient_covered_weights"])
    probes = json.loads((directory / "gradient-probes.json").read_text())
    for probe in probes:
        reference = gradient(checkpoints[probe["update"]]["weights"].numpy(), x, u, e)
        close(probe["exact_gradient"], reference)
        samples = np.asarray(probe["sampled_gradients"])
        close(
            probe["mean_absolute_l2_error"], np.linalg.norm(samples.mean(0) - reference)
        )
        close(
            probe["mean_single_draw_squared_gradient_error"],
            np.mean(np.sum((samples - reference) ** 2, axis=1)),
        )
        close(probe["total_coordinate_variance"], samples.var(0, ddof=1).sum())
    assert model["training_policy_rows"] == 59 * 2000
    assert model["training_target_terms"] == (59 if count is None else count) * 2000
    assert model["history_policy_rows"] == 59 * 101
    assert model["probe_policy_rows"] == 59 * 64 * len(probes)
    # Original counters omit the four full exact-gradient reference calls.
    work[variant] = {
        "training_policy_rows": model["training_policy_rows"],
        "training_target_terms": model["training_target_terms"],
        "history_policy_rows": model["history_policy_rows"],
        "sampled_probe_policy_rows": model["probe_policy_rows"],
        "sampled_probe_target_terms": model["probe_target_terms"],
        "exact_probe_reference_policy_rows": 59 * 4,
        "exact_probe_reference_target_terms": 59 * 4,
        "total_training_and_diagnostic_policy_rows": model["training_policy_rows"]
        + model["history_policy_rows"]
        + model["probe_policy_rows"]
        + 59 * 4,
    }
    print("Independent Adam/checkpoint replay", variant, flush=True)
ridge = next(m for m in models if m["kind"] == "ridge")
design = np.column_stack([np.ones(len(train)), x])
augment = np.diag([0.0] + [math.sqrt(len(train) * CFG["ridge_mse_l2_weight"])] * 4)
coefs = np.linalg.lstsq(
    np.vstack([design, augment]), np.r_[np.log(e / u), np.zeros(5)], rcond=None
)[0]
close([ridge["intercept"], *ridge["weights"]], coefs)
max_prediction_error = 0.0
for model in models:
    for split in ("train", "val", "test"):
        source = [r for r in rows if r["split"] == split]
        xx, uu, ee = arrays(source, mean, scale)
        saved = [
            r
            for r in predictions
            if r["variant"] == model["variant"] and r["split"] == split
        ]
        assert len(saved) == len(source)
        for row in saved:
            for key, value in source_rows[row["peptide_row_id"]].items():
                assert row[key] == value
        constant = bool(np.ptp(xx @ np.array(model["weights"])) == 0)
        if model["kind"] == "ridge":
            delta = model["intercept"] + xx @ np.array(model["weights"])
            predicted_ln = np.log(uu) + delta
            logp = predicted_ln - np.logaddexp.reduce(predicted_ln)
        else:
            logp = forward(np.array(model["weights"]), xx, uu)
            predicted_ln = math.log(uu.sum()) + model["beta"] + logp
            delta = (
                np.full(len(uu), model["beta"])
                if constant
                else predicted_ln - np.log(uu)
            )
        target_logp = np.log(ee / ee.sum())
        values = {
            "predicted_log_probability": logp,
            "target_log_probability": target_logp,
            "predicted_log10_intensity": predicted_ln / math.log(10),
            "target_log10_intensity": np.log10(ee),
            "predicted_log10_fold_change": delta / math.log(10),
            "target_log10_fold_change": np.log(ee / uu) / math.log(10),
            "predicted_treated_mean": np.exp(predicted_ln),
            "untreated_mean_input": uu,
            "treated_mean_target": ee,
        }
        for name, numbers in values.items():
            close([r[name] for r in saved], numbers)
            if "log" in name:
                max_prediction_error = max(
                    max_prediction_error,
                    float(np.max(np.abs(numbers - [r[name] for r in saved]))),
                )
        metric = next(
            r
            for r in all_metrics
            if r["variant"] == model["variant"] and r["split"] == split
        )
        close(metric["reverse_kl"], np.exp(logp) @ (logp - target_logp))
        close(metric["forward_kl"], np.exp(target_logp) @ (target_logp - logp))
        close(
            metric["total_variation"],
            np.abs(np.exp(logp) - np.exp(target_logp)).sum() / 2,
        )
        for kind in ("intensity", "fold_change"):
            pred, target = (
                values["predicted_log10_" + kind],
                values["target_log10_" + kind],
            )
            result = metric["log10_" + kind]
            squared = math.fsum(
                (float(a) - float(b)) ** 2 for a, b in zip(pred, target)
            ) / len(pred)
            close(result["mse"], squared)
            close(result["rmse"], math.sqrt(squared))
            close(
                result["mae"],
                math.fsum(abs(float(a) - float(b)) for a, b in zip(pred, target))
                / len(pred),
            )
            if kind == "fold_change" and constant:
                assert (
                    result["pearson"] is result["spearman"] is None
                    and result["prediction_sd"] == 0
                )
            else:
                close(result["pearson"], correlation(pred, target))
                close(result["spearman"], correlation(ranks(pred), ranks(target)))
bootstrap = json.loads((R / "test-component-bootstrap.json").read_text())
groups = bootstrap["components"]
draws = np.random.default_rng(20261023).integers(
    0, len(groups), size=(1000, len(groups))
)
assert np.array_equal(draws, bootstrap["bootstrap_indices"])
test = [r for r in predictions if r["split"] == "test"]
baseline = {r["peptide_row_id"]: r for r in test if r["variant"] == "common_shift"}
for variant, result in bootstrap["variants"].items():
    records = [r for r in test if r["variant"] == variant]
    d = {}
    for row in records:
        ref = baseline[row["peptide_row_id"]]
        improvement = (
            ref["predicted_log10_fold_change"] - ref["target_log10_fold_change"]
        ) ** 2 - (
            row["predicted_log10_fold_change"] - row["target_log10_fold_change"]
        ) ** 2
        d.setdefault(row["component_id"], []).append(improvement)
    actual = []
    for indices in draws:
        values = [value for index in indices for value in d[groups[index]]]
        actual.append(math.fsum(values) / len(values))
    close(actual, result["draws"])
    close(np.quantile(actual, [0.025, 0.975]), result["bootstrap_95pct_interval"])
summary = json.loads((R / "training-summary.json").read_text())
for variant, expected in summary["predictive_gates"].items():
    candidate = [
        m
        for m in all_metrics
        if m["variant"] == variant and m["split"] in ("val", "test")
    ]
    assert expected == all(
        m["log10_fold_change"]["rmse"]
        < next(
            b
            for b in all_metrics
            if b["variant"] == "common_shift" and b["split"] == m["split"]
        )["log10_fold_change"]["rmse"]
        and m["log10_fold_change"]["prediction_sd"] > 1e-8
        for m in candidate
    )
result = {
    "passed": True,
    "n_independent_optimizer_updates": 14000,
    "n_checkpoints_replayed": 28,
    "n_ridge_augmented_lstsq_replays": 1,
    "n_predictions_verified": len(predictions),
    "n_split_metric_sets_verified": len(all_metrics),
    "maximum_checkpoint_weight_error": max_replay_error,
    "maximum_log_prediction_error": max_prediction_error,
    "bootstrap_draws_recomputed_per_variant": 1000,
    "original_source_membership_and_scaling_verified": True,
    "all_model_selection_gates_verified": True,
    "wall_seconds": time.perf_counter() - started,
    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
}
(R / "independent-results.json").write_text(json.dumps(result, indent=2) + "\n")
(R / "work-accounting.json").write_text(
    json.dumps(
        {
            "note": "Original model receipts counted sampled probe rows but omitted four exact-reference gradients. This adds236 policy/target rows per fit without changing fits, predictions or recorded elapsed time.",
            "variants": work,
            "terminal_evaluation_rows_all_models_splits": 870,
        },
        indent=2,
    )
    + "\n"
)
print(json.dumps(result, sort_keys=True))
