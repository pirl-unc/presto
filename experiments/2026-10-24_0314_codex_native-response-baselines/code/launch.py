"""Frozen, local, exact/censored biological response baselines."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import optimize, stats

FAMILY = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as f:
        return json.load(f)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sequence_features(groups):
    names = ["log_length"] + ["fraction:" + a for a in AA]
    names += [side + ":" + a for side in ("first", "last") for a in AA]
    names += ["n_terminal_acetylation", "oxidation_count"]
    matrix = []
    for g in groups:
        s = g["peptide_sequence"]
        assert s and set(s) <= set(AA)
        mods = Counter(m["name"] for m in g["modifications"])
        matrix.append(
            [math.log(len(s))]
            + [s.count(a) / len(s) for a in AA]
            + [float(s[index] == a) for index in (0, -1) for a in AA]
            + [float(mods["N_terminal_acetylation"] > 0), float(mods["oxidation"])]
        )
    return np.asarray(matrix, dtype=np.float64), names


def design(groups, train, group_weights, variant, rank_threshold):
    conditions = sorted({g["condition_id"] for g, t in zip(groups, train) if t})
    compounds = sorted({g["condition"][2] for g, t in zip(groups, train) if t})
    raw = np.array([[float(g["condition_id"] == c) for c in conditions] for g in groups])
    names = ["condition:" + c for c in conditions]
    seq, seq_names = sequence_features(groups)
    if variant in {"sequence_shared", "sequence_compound"}:
        raw = np.column_stack([raw, seq])
        names += seq_names
    if variant == "sequence_compound":
        for c in compounds:
            indicator = np.array([float(g["condition"][2] == c) for g in groups])
            raw = np.column_stack([raw, seq * indicator[:, None]])
            names += ["compound:" + c + "*" + name for name in seq_names]
    weights = group_weights[train] / group_weights[train].sum()
    mean = weights @ raw[train]
    variance = weights @ (raw[train] - mean) ** 2
    keep = variance > 1e-24
    scale = np.sqrt(variance[keep])
    standardized = (raw[:, keep] - mean[keep]) / scale
    _, singular, vt = np.linalg.svd(
        np.sqrt(weights[:, None]) * standardized[train], full_matrices=False
    )
    rank = int(np.sum(singular > singular[0] * rank_threshold))
    projection = vt[:rank].T
    x = np.column_stack([np.ones(len(groups)), standardized @ projection])
    metadata = {
        "all_feature_names": names,
        "retained_feature_names": np.array(names)[keep].tolist(),
        "n_raw_columns": len(names),
        "n_nonconstant_columns": int(keep.sum()),
        "n_supported_feature_directions": rank,
        "n_parameters": rank + 1,
        "conditions": conditions,
        "compounds": compounds,
        "unseen_validation_condition_groups": sum(
            not t and g["condition_id"] not in conditions for g, t in zip(groups, train)
        ),
        "unseen_validation_compound_groups": sum(
            not t and g["condition"][2] not in compounds for g, t in zip(groups, train)
        ),
        "removed_constant_feature_names": np.array(names)[~keep].tolist(),
    }
    arrays = {
        "mean": mean,
        "keep": keep,
        "scale": scale,
        "projection": projection,
        "singular_values": singular,
    }
    return x, metadata, arrays


def residuals(prediction, target, censor):
    delta = prediction - target
    return np.where(
        censor == -1, np.maximum(delta, 0), np.where(censor == 1, np.minimum(delta, 0), delta)
    )


def objective(theta, x, y, censor, weight, penalty):
    residual = residuals(x @ theta, y, censor)
    loss = float(weight @ residual**2 + penalty * (theta[1:] @ theta[1:]))
    gradient = 2 * (x.T @ (weight * residual))
    gradient[1:] += 2 * penalty * theta[1:]
    return loss, gradient


def metric_rows(records):
    p = np.array([r["prediction_log10_ratio"] for r in records])
    y = np.array([r["target_log10_ratio"] for r in records])
    w = np.array([r["weight"] for r in records])
    c = np.array([r["censor_code"] for r in records])
    exact, bounded = c == 0, c != 0
    out = {
        "n_observations": len(records),
        "n_response_groups": len({r["response_group_id"] for r in records}),
        "n_components": len({r["component_id"] for r in records}),
        "weight_sum": float(w.sum()),
        "data_loss": float(w @ residuals(p, y, c) ** 2 / w.sum()),
        "n_exact_observations": int(exact.sum()),
        "n_bounded_observations": int(bounded.sum()),
        "prediction_sd_pooled": float(p.std()),
        "prediction_min": float(p.min()),
        "prediction_max": float(p.max()),
    }
    if exact.any():
        pe, ye, we = p[exact], y[exact], w[exact] / w[exact].sum()
        mse = float(we @ (pe - ye) ** 2)
        target_mean = float(we @ ye)
        variance = float(we @ (ye - target_mean) ** 2)
        varying = len(pe) > 1 and np.ptp(pe) > 0 and np.ptp(ye) > 0
        direction = ye != 0
        out.update(
            exact_rmse=math.sqrt(mse),
            exact_mae=float(we @ abs(pe - ye)),
            exact_r2=None if variance == 0 else 1 - mse / variance,
            exact_pearson_pooled=float(stats.pearsonr(pe, ye).statistic) if varying else None,
            exact_spearman_pooled=float(stats.spearmanr(pe, ye).statistic) if varying else None,
            exact_prediction_sd_pooled=float(pe.std()),
            exact_target_sd_pooled=float(ye.std()),
            exact_direction_accuracy=None
            if not direction.any()
            else float(
                np.sum(we[direction] * (np.sign(pe[direction]) == np.sign(ye[direction])))
                / we[direction].sum()
            ),
        )
    if bounded.any():
        satisfied = np.where(c[bounded] == -1, p[bounded] < y[bounded], p[bounded] > y[bounded])
        wb = w[bounded] / w[bounded].sum()
        out.update(
            bound_satisfaction_weighted=float(wb @ satisfied),
            bound_satisfaction_pooled=float(satisfied.mean()),
            bound_violation_rmse=math.sqrt(
                float(wb @ residuals(p[bounded], y[bounded], c[bounded]) ** 2)
            ),
        )
        for code, name in ((-1, "upper"), (1, "lower")):
            mask = c == code
            if mask.any():
                good = p[mask] < y[mask] if code == -1 else p[mask] > y[mask]
                out[name + "_bound_satisfaction_weighted"] = float(w[mask] @ good / w[mask].sum())
    return out


def gradient_check(theta, x, y, c, w, penalty):
    rng = np.random.default_rng(703)
    state = theta + 0.01 * rng.normal(size=len(theta))
    loss, grad = objective(state, x, y, c, w, penalty)
    results = []
    for _ in range(3):
        direction = rng.normal(size=len(theta))
        direction /= np.linalg.norm(direction)
        step = 1e-6
        numerical = (
            objective(state + step * direction, x, y, c, w, penalty)[0]
            - objective(state - step * direction, x, y, c, w, penalty)[0]
        ) / (2 * step)
        analytic = float(grad @ direction)
        error = abs(numerical - analytic)
        assert error < 1e-7, (numerical, analytic)
        results.append({"numerical": numerical, "analytic": analytic, "absolute_error": error})
    return {"state_loss": loss, "directions": results}


def main():
    start = time.perf_counter()
    config = read(FAMILY / "reproduce/launch.json")
    assert not (FAMILY / "results/summary.json").exists(), "Do not overwrite runs"
    execution = read(FAMILY / "reproduce/execution.json")
    assert execution["launcher_sha256"] == sha(__file__)
    recipe = Path(config["recipe_directory"])
    assert sha(recipe / "manifest.json") == config["recipe_manifest_sha256"]
    receipt = read(config["admission_receipt"])
    assert (
        receipt["status"] == "passed"
        and receipt["recipe_manifest_sha256"] == config["recipe_manifest_sha256"]
    )
    assert sha(config["admission_receipt"]) == execution["admission_receipt_sha256"]
    manifest = read(recipe / "manifest.json")
    assert sha(recipe / "examples.json.gz") == manifest["files"]["examples.json.gz"]["sha256"]
    groups = [g for g in read(recipe / "examples.json.gz") if g["split"] in {"train", "val"}]
    train = np.array([g["split"] == "train" for g in groups])
    group_weights = np.array([g["group_weight"] for g in groups])
    observations = []
    indices = []
    for i, g in enumerate(groups):
        for o in g["observations"]:
            row = {
                k: g[k]
                for k in (
                    "response_group_id",
                    "condition_id",
                    "condition",
                    "component_id",
                    "split",
                    "peptide_sequence",
                    "modifications",
                    "full_parent_model_supported",
                )
            }
            row.update(o)
            row["censor_code"] = {None: 0, "lt": -1, "gt": 1}[o["inequality"]]
            observations.append(row)
            indices.append(i)
    indices = np.array(indices)
    obs_train = train[indices]
    y = np.array([o["target_log10_ratio"] for o in observations])
    c = np.array([o["censor_code"] for o in observations])
    w = np.array([o["weight"] for o in observations])
    assert math.isclose(w[obs_train].sum(), 1, abs_tol=1e-12)
    assert math.isclose(w[~obs_train].sum(), 1, abs_tol=1e-12)
    # Explicit upper/lower-bound arithmetic qualification, independent of real outcomes.
    toy = residuals(np.array([0.0, 2.0, 0.0, 2.0]), np.ones(4), np.array([0, -1, 1, 1]))
    assert np.array_equal(toy, [-1.0, 1.0, -1.0, 0.0])
    assert math.isclose(np.array([0.1, 0.2, 0.3, 0.4]) @ toy**2, 0.6)
    all_metrics, all_strata, fit_records = [], [], []
    exact_train = obs_train & (c == 0)
    exact_mean = float(w[exact_train] @ y[exact_train] / w[exact_train].sum())
    constant = optimize.brentq(
        lambda value: float(w[obs_train] @ residuals(value, y[obs_train], c[obs_train])),
        float(y[obs_train].min() - 1),
        float(y[obs_train].max() + 1),
        xtol=1e-14,
    )
    for variant in config["variants"]:
        arm_start = time.perf_counter()
        fit = {"variant": variant, "pretraining": "none", "test_evaluated": False}
        if variant in {"no_change", "exact_train_mean", "censored_constant"}:
            value = {
                "no_change": 0.0,
                "exact_train_mean": exact_mean,
                "censored_constant": constant,
            }[variant]
            prediction = np.full(len(observations), value)
            fit.update(
                constant=value,
                n_parameters=int(variant != "no_change"),
                converged=True,
                gradient_checks="analytical baseline",
                training_objective=float(
                    w[obs_train] @ residuals(value, y[obs_train], c[obs_train]) ** 2
                ),
            )
        else:
            x, features, arrays = design(
                groups, train, group_weights, variant, config["relative_rank_threshold"]
            )
            xo = x[indices]
            xt, yt, ct, wt = xo[obs_train], y[obs_train], c[obs_train], w[obs_train]
            theta0 = np.zeros(x.shape[1])
            initial_loss, initial_grad = objective(theta0, xt, yt, ct, wt, config["penalty"])

            def callback(_, started_at=arm_start, arm=variant):
                if time.perf_counter() - started_at > config["seconds_per_fit_cap"]:
                    raise TimeoutError(arm)

            result = optimize.minimize(
                objective,
                theta0,
                args=(xt, yt, ct, wt, config["penalty"]),
                method="L-BFGS-B",
                jac=True,
                callback=callback,
                options={
                    "maxiter": config["maxiter"],
                    "gtol": config["gtol"],
                    "ftol": config["ftol"],
                },
            )
            prediction = xo @ result.x
            final_loss, final_grad = objective(result.x, xt, yt, ct, wt, config["penalty"])
            checks = gradient_check(result.x, xt, yt, ct, wt, config["penalty"])
            fit.update(
                features=features,
                converged=bool(result.success),
                convergence_message=str(result.message),
                iterations=int(result.nit),
                objective_evaluations=int(result.nfev),
                initial_objective=initial_loss,
                training_objective=final_loss,
                initial_gradient_inf=float(np.max(abs(initial_grad))),
                final_gradient_inf=float(np.max(abs(final_grad))),
                n_parameters=len(result.x),
                n_parameters_changed=int(np.sum(abs(result.x) > 1e-12)),
                coefficient_norm=float(np.linalg.norm(result.x[1:])),
                gradient_checks=checks,
            )
            np.savez_compressed(
                FAMILY / "results" / (variant + "-model.npz"), theta=result.x, **arrays
            )
        rows = [
            {**o, "variant": variant, "prediction_log10_ratio": float(p)}
            for o, p in zip(observations, prediction)
        ]
        assert all(math.isfinite(r["prediction_log10_ratio"]) for r in rows)
        with gzip.open(FAMILY / "results" / (variant + "-predictions.jsonl.gz"), "wt") as f:
            for row in rows:
                f.write(json.dumps(row, allow_nan=False) + "\n")
        for split in ("train", "val"):
            selected = [r for r in rows if r["split"] == split]
            metrics = {"variant": variant, "split": split, **metric_rows(selected)}
            all_metrics.append(metrics)
            for field in ("condition_id", "component_id"):
                for value in sorted({r[field] for r in selected}):
                    all_strata.append(
                        {
                            "variant": variant,
                            "split": split,
                            "stratum": field,
                            "stratum_id": value,
                            **metric_rows([r for r in selected if r[field] == value]),
                        }
                    )
        fit["seconds"] = time.perf_counter() - arm_start
        fit["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        fit["within_resource_caps"] = (
            fit["seconds"] < config["seconds_per_fit_cap"]
            and fit["peak_rss_bytes"] < config["rss_bytes_cap"]
        )
        fit_records.append(fit)
        write(FAMILY / "results" / (variant + "-fit.json"), fit)
        print(
            json.dumps(
                {
                    "variant": variant,
                    "converged": fit["converged"],
                    "seconds": fit["seconds"],
                    "parameters": fit["n_parameters"],
                    "validation": all_metrics[-1],
                }
            ),
            flush=True,
        )
        assert fit["within_resource_caps"]
    validation = {r["variant"]: r for r in all_metrics if r["split"] == "val"}
    comparisons = {}
    for name in ("sequence_shared", "sequence_compound"):
        m = validation[name]
        comparisons[name] = {
            "beats_condition_and_no_change_primary": m["data_loss"]
            < min(validation["condition"]["data_loss"], validation["no_change"]["data_loss"]),
            "beats_condition_exact_rmse": m["exact_rmse"] < validation["condition"]["exact_rmse"],
            "relative_primary_improvement_vs_condition": 1
            - m["data_loss"] / validation["condition"]["data_loss"],
        }
    summary = {
        "recipe_manifest_sha256": config["recipe_manifest_sha256"],
        "metrics": all_metrics,
        "comparisons": comparisons,
        "fits": fit_records,
        "test_evaluated": False,
        "test_reason": config["test_policy"],
        "seconds": time.perf_counter() - start,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "biological_scope": "native_extract_relative_treatment_response;not_APC_or_absolute_concentration",
    }
    assert (
        summary["seconds"] < config["seconds_family_cap"]
        and summary["peak_rss_bytes"] < config["rss_bytes_cap"]
    )
    write(FAMILY / "results/summary.json", summary)
    write(FAMILY / "results/stratified-metrics.json", all_strata)
    columns = sorted(set().union(*(r.keys() for r in all_metrics)))
    with (FAMILY / "results/metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_metrics)
    print(
        json.dumps({"completed": True, "seconds": summary["seconds"], "comparisons": comparisons}),
        flush=True,
    )


if __name__ == "__main__":
    main()
