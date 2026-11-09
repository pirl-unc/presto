"""All four prespecified weighted linear models and descriptive metrics."""

from collections import Counter
import math

import numpy as np
from scipy.stats import rankdata

from common import ids_digest

ARMS = ("constant", "unit_slope", "shared_slope", "method_slope")


def design(rows, groups, methods, arm):
    if arm not in ARMS:
        raise ValueError("Unknown arm")
    extra = 1 if arm == "shared_slope" else len(methods) if arm == "method_slope" else 0
    matrix = np.zeros((len(rows), len(groups) + extra))
    for index, row in enumerate(rows):
        if not math.isfinite(row["x"]) or not math.isfinite(row["y"]):
            raise ValueError("Nonfinite input")
        matrix[index, groups.index(row["group"])] = 1
        if arm == "shared_slope":
            matrix[index, -1] = row["x"]
        elif arm == "method_slope":
            matrix[index, len(groups) + methods.index(row["method"])] = row["x"]
    return matrix


def fit(rows, arm):
    if not rows or len({r["candidate_id"] for r in rows}) != len(rows):
        raise ValueError("Empty or duplicate training membership")
    groups = sorted({r["group"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    for group in groups:
        if len({r["x"] for r in rows if r["group"] == group}) < 2:
            raise ValueError("Training group needs two distinct levels")
    matrix = design(rows, groups, methods, arm)
    counts = Counter(r["group"] for r in rows)
    weights = np.array([1 / (len(groups) * counts[r["group"]]) for r in rows])
    offset = np.array([r["x"] if arm == "unit_slope" else 0 for r in rows])
    target = np.array([r["y"] for r in rows])
    coef, _, rank, singular = np.linalg.lstsq(
        matrix * np.sqrt(weights[:, None]), (target - offset) * np.sqrt(weights), rcond=None
    )
    if rank != matrix.shape[1]:
        raise ValueError("Unidentified model")
    return {
        "arm": arm,
        "groups": groups,
        "methods": methods,
        "coefficients": coef.tolist(),
        "rank": int(rank),
        "singular_values": singular.tolist(),
        "train_ids": [r["candidate_id"] for r in rows],
        "train_ids_sha256": ids_digest(r["candidate_id"] for r in rows),
        "weights": weights.tolist(),
        "train_loss": float(np.sum(weights * (matrix @ coef + offset - target) ** 2)),
    }


def predict(model, rows):
    matrix = design(rows, model["groups"], model["methods"], model["arm"])
    result = matrix @ np.array(model["coefficients"])
    if model["arm"] == "unit_slope":
        result += np.array([r["x"] for r in rows])
    return result.tolist()


def fit_fold(rows, arm, held_level):
    train = [r for r in rows if r["nominal_level_id"] != held_level]
    test = [r for r in rows if r["nominal_level_id"] == held_level]
    if not train or not test or {r["group"] for r in train} != {r["group"] for r in rows}:
        raise ValueError("Invalid held-level support")
    model = fit(train, arm)
    model["held_level"] = held_level
    return model, test, predict(model, test)


def correlation(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if len(y) < 2 or np.std(y) < 1e-12 or np.std(p) < 1e-12:
        return None
    y, p = y - y.mean(), p - p.mean()
    return float(y @ p / np.sqrt((y @ y) * (p @ p)))


def metrics(rows):
    y = np.array([r["y"] for r in rows])
    p = np.array([r["prediction"] for r in rows])
    group_ids = sorted({r["group"] for r in rows})
    groups = [[i for i, r in enumerate(rows) if r["group"] == g] for g in group_ids]
    errors = p - y
    yc, pc = y.copy(), p.copy()
    for ids in groups:
        yc[ids] -= y[ids].mean()
        pc[ids] -= p[ids].mean()
    macro_mse = float(np.mean([np.mean(errors[ids] ** 2) for ids in groups]))
    return {
        "n": len(rows),
        "n_groups": len(groups),
        "mse": float(np.mean(errors**2)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
        "equal_group_mse": macro_mse,
        "equal_group_rmse": math.sqrt(macro_mse),
        "equal_group_mae": float(np.mean([np.mean(np.abs(errors[ids])) for ids in groups])),
        "pearson": correlation(y, p),
        "spearman": correlation(rankdata(y), rankdata(p)),
        "target_sd": float(np.std(y)),
        "prediction_sd": float(np.std(p)),
        "centered_rmse": float(np.sqrt(np.mean((pc - yc) ** 2))),
        "centered_pearson": correlation(yc, pc),
        "centered_spearman": correlation(rankdata(yc), rankdata(pc)),
        "centered_target_sd": float(np.std(yc)),
        "centered_prediction_sd": float(np.std(pc)),
        "centered_spread_ratio": float(np.std(pc) / np.std(yc)) if np.std(yc) else None,
        "undefined_correlation_policy": "null for fewer than two or SD below 1e-12",
    }


def summarize(predictions, models):
    by_arm = {}
    for arm in ARMS:
        rows = [r for r in predictions if r["arm"] == arm]
        by_arm[arm] = {"overall": metrics(rows)}
        for field in ("method", "standard_id", "nominal_level_id"):
            by_arm[arm][field] = {
                key: metrics([r for r in rows if r[field] == key])
                for key in sorted({r[field] for r in rows})
            }
    base = by_arm["constant"]
    shared = by_arm["shared_slope"]
    gain = 1 - shared["overall"]["equal_group_rmse"] / base["overall"]["equal_group_rmse"]
    slopes = [m["coefficients"][-1] for m in models if m["arm"] == "shared_slope"]
    gates = {
        "at_least_10_percent_macro_rmse_gain": gain >= 0.10,
        "all_five_training_slopes_positive": len(slopes) == 5 and min(slopes) > 0,
        "centered_prediction_spread_at_least_25_percent": (
            shared["overall"]["centered_spread_ratio"] is not None
            and shared["overall"]["centered_spread_ratio"] >= 0.25
        ),
        "neither_method_worse": all(
            shared["method"][key]["equal_group_rmse"] <= base["method"][key]["equal_group_rmse"]
            for key in base["method"]
        ),
    }
    return {
        "by_arm": by_arm,
        "shared_slope_gain": gain,
        "fold_shared_slopes": slopes,
        "gates": gates,
        "all_gates_pass": all(gates.values()),
    }
