"""Frozen exact/sampled conditional-response comparison; one terminal evaluation."""

import gzip
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from model import FEATURE_NAMES, exact_loss, features, log_policy, sampled_loss
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
RESULTS = ROOT / "results"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_data(rows, mean=None, scale=None):
    raw = features([r["peptide_sequence"] for r in rows])
    if mean is None:
        mean, scale = raw.mean(0), np.maximum(raw.std(0), 1e-8)
    x = (raw - mean) / scale
    u = np.array([r["untreated_mean_intensity"] for r in rows])
    e = np.array([r["epoxomicin_mean_intensity"] for r in rows])
    assert np.isfinite(x).all() and (u > 0).all() and (e > 0).all()
    return {
        "rows": rows,
        "raw_features": raw,
        "x": x,
        "u": u,
        "e": e,
        "q0": u / u.sum(),
        "q1": e / e.sum(),
        "mean": mean,
        "scale": scale,
    }


def tensor_data(data):
    return tuple(
        torch.tensor(a, dtype=torch.float64)
        for a in (data["x"], np.log(data["q0"]), np.log(data["q1"]))
    )


def gradient_probe(w, inputs, counts, checkpoint):
    started = time.perf_counter()
    exact = (
        torch.autograd.grad(exact_loss(w, *inputs, CFG["policy_l2_half_weight"]), w)[0]
        .detach()
        .numpy()
    )
    results = []
    for count in counts:
        generator = torch.Generator().manual_seed(9300000 + checkpoint * 1000 + count)
        estimates = []
        for _ in range(CFG["probe_groups"]):
            loss = sampled_loss(
                w, *inputs, count, generator, CFG["policy_l2_half_weight"]
            )
            estimates.append(torch.autograd.grad(loss, w)[0].detach().numpy())
        estimates = np.array(estimates)
        average = estimates.mean(0)
        norm = np.linalg.norm(exact)
        results.append(
            {
                "update": checkpoint,
                "sample_count": count,
                "n_groups": len(estimates),
                "exact_gradient": exact.tolist(),
                "sampled_gradients": estimates.tolist(),
                "mean_absolute_l2_error": float(np.linalg.norm(average - exact)),
                "mean_relative_l2_error": float(np.linalg.norm(average - exact) / norm)
                if norm > 1e-12
                else None,
                "exact_gradient_norm": float(norm),
                "mean_gradient_cosine": float(
                    average @ exact / (np.linalg.norm(average) * norm)
                )
                if norm > 1e-12 and np.linalg.norm(average) > 1e-12
                else None,
                "mean_single_draw_squared_gradient_error": float(
                    np.mean(np.sum((estimates - exact) ** 2, axis=1))
                ),
                "total_coordinate_variance": float(estimates.var(0, ddof=1).sum()),
                "target_terms": count * len(estimates),
                "policy_rows": len(inputs[0]) * len(estimates),
                "note": "Relative error is unstable near an optimum; retain absolute error, exact norm and all estimates.",
            }
        )
    return results, time.perf_counter() - started


def fit_policy(data, variant, seed, count):
    directory = RESULTS / "runs" / variant
    directory.mkdir(parents=True, exist_ok=False)
    inputs = tensor_data(data)
    weights = torch.zeros(4, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([weights], **CFG["adam"])
    generator = torch.Generator().manual_seed(seed)
    beta = float(np.log(data["e"].sum() / data["u"].sum()))
    start = time.perf_counter()
    probes, history, visited = [], [], np.zeros(4, dtype=bool)
    fit_seconds = probe_seconds = history_seconds = 0.0
    for update in range(CFG["steps"] + 1):
        if update:
            tick = time.perf_counter()
            optimizer.zero_grad()
            if count is None:
                loss = exact_loss(weights, *inputs, CFG["policy_l2_half_weight"])
            else:
                loss = sampled_loss(
                    weights, *inputs, count, generator, CFG["policy_l2_half_weight"]
                )
            assert torch.isfinite(loss)
            loss.backward()
            assert weights.grad is not None and torch.isfinite(weights.grad).all()
            visited |= weights.grad.detach().numpy() != 0
            optimizer.step()
            fit_seconds += time.perf_counter() - tick
        if update % 20 == 0:
            tick = time.perf_counter()
            with torch.no_grad():
                logp = log_policy(weights, inputs[0], inputs[1])
                kl = (logp.exp() * (logp - inputs[2])).sum().item()
                penalty = (
                    CFG["policy_l2_half_weight"] / 2 * weights.square().sum().item()
                )
            history.append(
                {
                    "update": update,
                    "train_reverse_kl": kl,
                    "l2_penalty": penalty,
                    "objective": kl + penalty,
                    "weights": weights.detach().tolist(),
                }
            )
            history_seconds += time.perf_counter() - tick
        if update in CFG["probe_updates"]:
            torch.save(
                {
                    "weights": weights.detach(),
                    "beta": beta,
                    "feature_mean": data["mean"],
                    "feature_scale": data["scale"],
                    "optimizer": optimizer.state_dict(),
                    "sampling_rng": generator.get_state(),
                    "update": update,
                    "variant": variant,
                    "seed": seed,
                    "sample_count": count,
                },
                directory / f"checkpoint-{update}.pt",
            )
            values, seconds = gradient_probe(
                weights,
                inputs,
                CFG["sample_counts"] if count is None else [count],
                update,
            )
            probes.extend(values)
            probe_seconds += seconds
    assert visited.all(), visited
    model = {
        "kind": "policy",
        "variant": variant,
        "seed": seed,
        "sample_count": count,
        "weights": weights.detach().tolist(),
        "beta": beta,
        "feature_mean": data["mean"].tolist(),
        "feature_scale": data["scale"].tolist(),
        "feature_names": list(FEATURE_NAMES),
        "n_trainable_coefficients": 5,
        "gradient_covered_weights": visited.tolist(),
        "scale_fit": "closed_form_train_log_total_ratio",
        "all_four_weights_changed": bool((weights.detach().numpy() != 0).all()),
        "optimizer_seconds": fit_seconds,
        "probe_seconds": probe_seconds,
        "history_eval_seconds": history_seconds,
        "total_fit_wall_seconds": time.perf_counter() - start,
        "training_policy_rows": len(data["rows"]) * CFG["steps"],
        "training_target_terms": (len(data["rows"]) if count is None else count)
        * CFG["steps"],
        "history_policy_rows": len(data["rows"]) * len(history),
        "probe_policy_rows": sum(p["policy_rows"] for p in probes),
        "probe_target_terms": sum(p["target_terms"] for p in probes),
        "checkpoint_selection": "final_update_2000;no_validation_selection",
    }
    save(directory / "model.json", model)
    save(directory / "history.json", history)
    save(directory / "gradient-probes.json", probes)
    print(
        "Fit complete",
        variant,
        "training KL",
        history[-1]["train_reverse_kl"],
        flush=True,
    )
    return model


def regression_metrics(predicted, target, *, constant=False):
    varying = not constant and np.ptp(predicted) > 0 and np.ptp(target) > 0
    return {
        "mse": float(np.mean((predicted - target) ** 2)),
        "rmse": float(np.sqrt(np.mean((predicted - target) ** 2))),
        "mae": float(np.mean(np.abs(predicted - target))),
        "pearson": float(np.corrcoef(predicted, target)[0, 1]) if varying else None,
        "spearman": float(spearmanr(predicted, target).statistic) if varying else None,
        "prediction_sd": 0.0 if constant else float(predicted.std()),
        "target_sd": float(target.std()),
        "constant_prediction": constant,
    }


def evaluate(model, data, split):
    kind, x, u, e = model["kind"], data["x"], data["u"], data["e"]
    if kind == "ridge":
        delta = model["intercept"] + x @ np.array(model["weights"])
        predicted_log = np.log(u) + delta
        logp = predicted_log - np.logaddexp.reduce(predicted_log)
        constant = bool(np.ptp(x @ np.array(model["weights"])) == 0)
    else:
        scores = np.log(data["q0"]) + x @ np.array(model["weights"])
        logp = scores - np.logaddexp.reduce(scores)
        predicted_log = np.log(u.sum()) + model["beta"] + logp
        constant = bool(np.ptp(x @ np.array(model["weights"])) == 0)
        delta = (
            np.full(len(u), model["beta"]) if constant else predicted_log - np.log(u)
        )
    logq = np.log(data["q1"])
    predicted_delta, observed_delta = delta / np.log(10), np.log(e / u) / np.log(10)
    predicted_log10, observed_log10 = predicted_log / np.log(10), np.log10(e)
    metrics = {
        "variant": model["variant"],
        "split": split,
        "n_peptides": len(u),
        "n_components": len({r["component_id"] for r in data["rows"]}),
        "reverse_kl": float(np.exp(logp) @ (logp - logq)),
        "forward_kl": float(data["q1"] @ (logq - logp)),
        "total_variation": float(np.abs(np.exp(logp) - data["q1"]).sum() / 2),
        "log10_intensity": regression_metrics(predicted_log10, observed_log10),
        "log10_fold_change": regression_metrics(
            predicted_delta, observed_delta, constant=constant
        ),
        "predicted_total_intensity": float(np.exp(predicted_log).sum()),
        "observed_total_intensity": float(e.sum()),
        "observed_untreated_total_input": float(u.sum()),
    }
    if kind != "ridge":
        penalty = (
            CFG["policy_l2_half_weight"] / 2 * float(np.square(model["weights"]).sum())
        )
        metrics.update(
            {
                "l2_penalty": penalty,
                "composition_objective": metrics["reverse_kl"] + penalty,
                "log_total_ratio_squared_error": float(
                    (
                        np.log(metrics["predicted_total_intensity"] / u.sum())
                        - np.log(e.sum() / u.sum())
                    )
                    ** 2
                ),
            }
        )
    records = []
    for i, row in enumerate(data["rows"]):
        records.append(
            {
                "variant": model["variant"],
                "split": split,
                **row,
                "features_standardized": x[i].tolist(),
                "untreated_mean_input": float(u[i]),
                "treated_mean_target": float(e[i]),
                "predicted_treated_mean": float(np.exp(predicted_log[i])),
                "target_log10_intensity": float(observed_log10[i]),
                "predicted_log10_intensity": float(predicted_log10[i]),
                "target_log10_fold_change": float(observed_delta[i]),
                "predicted_log10_fold_change": float(predicted_delta[i]),
                "target_log_probability": float(logq[i]),
                "predicted_log_probability": float(logp[i]),
                "baseline_log_probability_input": float(np.log(data["q0"][i])),
                "fold_change_prediction_constant": constant,
            }
        )
    return metrics, records


def bootstrap(predictions):
    test = [r for r in predictions if r["split"] == "test"]
    base = {r["peptide_row_id"]: r for r in test if r["variant"] == "common_shift"}
    components = sorted({r["component_id"] for r in base.values()})
    rng = np.random.default_rng(CFG["bootstrap"]["seed"])
    indices = rng.integers(
        0, len(components), size=(CFG["bootstrap"]["draws"], len(components))
    )
    result = {
        "components": components,
        "bootstrap_indices": indices.tolist(),
        "seed": CFG["bootstrap"]["seed"],
        "scope": "conditional_on_this_study_and_its_acquisitions;parent_component_resampling_only",
        "variants": {},
    }
    for variant in sorted({r["variant"] for r in test}):
        records = [r for r in test if r["variant"] == variant]
        deltas = {}
        for r in records:
            b = base[r["peptide_row_id"]]
            delta = (
                b["predicted_log10_fold_change"] - b["target_log10_fold_change"]
            ) ** 2 - (
                r["predicted_log10_fold_change"] - r["target_log10_fold_change"]
            ) ** 2
            deltas.setdefault(r["component_id"], []).append(delta)
        sums = np.array([sum(deltas[c]) for c in components])
        counts = np.array([len(deltas[c]) for c in components])
        draws = sums[indices].sum(1) / counts[indices].sum(1)
        result["variants"][variant] = {
            "test_mse_improvement_vs_common": float(sums.sum() / counts.sum()),
            "bootstrap_95pct_interval": np.quantile(draws, [0.025, 0.975]).tolist(),
            "bootstrap_fraction_positive": float((draws > 0).mean()),
            "draws": draws.tolist(),
        }
    return result


def main():
    started = time.perf_counter()
    assert json.loads((RESULTS / "math-checks.json").read_text())["passed"]
    assert json.loads((RESULTS / "independent-admission.json").read_text())["passed"]
    freeze = json.loads((ROOT / "reproduce/training-launch.json").read_text())
    for name, expected in freeze["code_sha256"].items():
        assert digest(ROOT / "code" / name) == expected, name
    assert (
        digest(Path(CFG["contrast_output"]) / "manifest.json")
        == freeze["contrast_manifest_sha256"]
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    rows = json.loads((Path(CFG["contrast_output"]) / "examples.json").read_text())
    train = make_data([r for r in rows if r["split"] == "train"])
    assert len(train["rows"]) == 59 and np.linalg.matrix_rank(train["x"]) == 4
    save(
        RESULTS / "feature-scaling.json",
        {
            "features": list(FEATURE_NAMES),
            "mean": train["mean"].tolist(),
            "population_std": train["scale"].tolist(),
            "training_rank": int(np.linalg.matrix_rank(train["x"])),
            "singular_values": np.linalg.svd(train["x"], compute_uv=False).tolist(),
            "training_only": True,
        },
    )
    models = []
    models.append(fit_policy(train, "exact", 20261023, None))
    for count in CFG["sample_counts"]:
        for seed in CFG["seeds"]:
            models.append(
                fit_policy(train, f"reinforce{count}_seed{seed}", seed, count)
            )
    beta = float(np.log(train["e"].sum() / train["u"].sum()))
    for name, value in (("no_change", 0.0), ("common_shift", beta)):
        models.append(
            {
                "kind": "control",
                "variant": name,
                "weights": [0.0] * 4,
                "beta": value,
                "scale_fit": "train_total_ratio" if value else "none",
            }
        )
    tick = time.perf_counter()
    design = np.column_stack((np.ones(len(train["rows"])), train["x"]))
    target = np.log(train["e"] / train["u"])
    penalty = np.diag([0.0] + [len(target) * CFG["ridge_mse_l2_weight"]] * 4)
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    models.append(
        {
            "kind": "ridge",
            "variant": "ridge",
            "intercept": float(coef[0]),
            "weights": coef[1:].tolist(),
            "objective": "mean_squared_ln_fold_change_plus_0.1_sum_squared_slopes;intercept_unpenalized",
            "optimizer_seconds": time.perf_counter() - tick,
        }
    )
    save(RESULTS / "models.json", models)
    # All models are now fixed. Validation/test targets are used only below.
    metrics, predictions = [], []
    for split in ("train", "val", "test"):
        data = (
            train
            if split == "train"
            else make_data(
                [r for r in rows if r["split"] == split], train["mean"], train["scale"]
            )
        )
        for model in models:
            metric, records = evaluate(model, data, split)
            metrics.append(metric)
            predictions.extend(records)
    save(RESULTS / "metrics.json", metrics)
    with gzip.open(RESULTS / "predictions.jsonl.gz", "wt") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    save(RESULTS / "test-component-bootstrap.json", bootstrap(predictions))
    common = {
        r["split"]: r["log10_fold_change"]["rmse"]
        for r in metrics
        if r["variant"] == "common_shift"
    }
    gates = {}
    for model in models:
        selected = [
            r
            for r in metrics
            if r["variant"] == model["variant"] and r["split"] in ("val", "test")
        ]
        gates[model["variant"]] = all(
            r["log10_fold_change"]["rmse"] < common[r["split"]]
            and r["log10_fold_change"]["prediction_sd"] > 1e-8
            for r in selected
        )
    summary = {
        "passed_execution": True,
        "n_policy_fits": 7,
        "n_closed_form_ridge_fits": 1,
        "n_controls": 2,
        "n_predictions": len(predictions),
        "predictive_gates": gates,
        "decision": "conditional_response_candidate_only"
        if any(gates.values())
        else "no_predictive_candidate;advance_known_substrate_kinetics_without_test_tuning",
        "scope": "selected captured_MS_response_given_observed_untreated_baseline;not_unobserved_products_or_concentration",
        "optimizer_updates": 7 * CFG["steps"],
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": "CPU",
            "torch_threads": torch.get_num_threads(),
            "dtype": "float64",
        },
    }
    save(RESULTS / "training-summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
