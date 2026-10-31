"""Scalar regression and metric replay without importing the fitting implementation."""

from collections import defaultdict
import json
import math
import statistics as stats

from common import ROOT, ids_digest, read, write

CHECKS = 0
MAX_ERROR = 0.0


def close(a, b, tolerance=2e-10):
    global CHECKS, MAX_ERROR
    if a is None or b is None:
        assert a is b
        return
    error = abs(a - b)
    assert error <= tolerance, (a, b, error)
    MAX_ERROR = max(MAX_ERROR, error)
    CHECKS += 1


def scalar_fit(rows, arm):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    means = {
        g: (stats.mean(r["x"] for r in v), stats.mean(r["y"] for r in v)) for g, v in groups.items()
    }
    slope_keys = ["all"] if arm == "shared_slope" else sorted({r["method"] for r in rows})
    slopes = {}
    for key in slope_keys:
        if arm in ("constant", "unit_slope"):
            slopes[key] = float(arm == "unit_slope")
            continue
        numerator, denominator = 0.0, 0.0
        for g, entries in groups.items():
            if arm == "method_slope" and entries[0]["method"] != key:
                continue
            mx, my = means[g]
            numerator += stats.mean((r["x"] - mx) * (r["y"] - my) for r in entries)
            denominator += stats.mean((r["x"] - mx) ** 2 for r in entries)
        assert denominator > 0
        slopes[key] = numerator / denominator
    slope_for = {
        g: slopes["all" if arm == "shared_slope" else entries[0]["method"]]
        for g, entries in groups.items()
    }
    intercepts = {g: my - slope_for[g] * mx for g, (mx, my) in means.items()}
    return intercepts, slope_for, slopes


def corr(y, p):
    if len(y) < 2 or stats.pstdev(y) < 1e-12 or stats.pstdev(p) < 1e-12:
        return None
    my, mp = stats.mean(y), stats.mean(p)
    return sum((a - my) * (b - mp) for a, b in zip(y, p, strict=True)) / math.sqrt(
        sum((a - my) ** 2 for a in y) * sum((b - mp) ** 2 for b in p)
    )


def ranks(values):
    positions = defaultdict(list)
    for index, value in enumerate(sorted(values), 1):
        positions[value].append(index)
    return [stats.mean(positions[value]) for value in values]


def verify_metrics(rows, recorded):
    y, p = [r["y"] for r in rows], [r["prediction"] for r in rows]
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["group"]].append(index)
    errors = [b - a for a, b in zip(y, p, strict=True)]
    yc, pc = y.copy(), p.copy()
    for ids in groups.values():
        my, mp = stats.mean(y[i] for i in ids), stats.mean(p[i] for i in ids)
        for i in ids:
            yc[i], pc[i] = y[i] - my, p[i] - mp
    mse = stats.mean(e * e for e in errors)
    macro_mse = stats.mean(stats.mean(errors[i] ** 2 for i in ids) for ids in groups.values())
    expected = {
        "n": len(rows),
        "n_groups": len(groups),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": stats.mean(abs(e) for e in errors),
        "equal_group_mse": macro_mse,
        "equal_group_rmse": math.sqrt(macro_mse),
        "equal_group_mae": stats.mean(
            stats.mean(abs(errors[i]) for i in ids) for ids in groups.values()
        ),
        "pearson": corr(y, p),
        "spearman": corr(ranks(y), ranks(p)),
        "target_sd": stats.pstdev(y),
        "prediction_sd": stats.pstdev(p),
        "centered_rmse": math.sqrt(stats.mean((b - a) ** 2 for a, b in zip(yc, pc, strict=True))),
        "centered_pearson": corr(yc, pc),
        "centered_spearman": corr(ranks(yc), ranks(pc)),
        "centered_target_sd": stats.pstdev(yc),
        "centered_prediction_sd": stats.pstdev(pc),
        "centered_spread_ratio": stats.pstdev(pc) / stats.pstdev(yc) if stats.pstdev(yc) else None,
    }
    # Centering mathematically equal values can change their raw floating-point rank.
    # Recompute rank metrics on the exact stored operands used by the report below.
    for key, value in expected.items():
        if key != "centered_spearman":
            close(value, recorded[key])
    return expected["centered_spearman"]


def main():
    rows = read(ROOT / "results/admitted.json")
    models = read(ROOT / "results/fold-models.json")
    full_models = read(ROOT / "results/full-models.json")
    predictions = read(ROOT / "results/development-predictions.json")
    full_predictions = read(ROOT / "results/descriptive-full-predictions.json")
    assert len(models) == 20 and len(full_models) == 4
    assert len(predictions) == len(full_predictions) == 4 * len(rows)
    assert len({(r["arm"], r["candidate_id"]) for r in predictions}) == len(predictions)
    for model in models + full_models:
        train = [r for r in rows if r["nominal_level_id"] != model.get("held_level")]
        assert model["train_ids"] == [r["candidate_id"] for r in train]
        assert model["train_ids_sha256"] == ids_digest(model["train_ids"])
        intercepts, slope_for, slopes = scalar_fit(train, model["arm"])
        expected_coef = [intercepts[g] for g in model["groups"]]
        if model["arm"] == "shared_slope":
            expected_coef += [slopes["all"]]
        if model["arm"] == "method_slope":
            expected_coef += [slopes[m] for m in model["methods"]]
        assert model["rank"] == len(expected_coef) == len(model["coefficients"])
        for a, b in zip(expected_coef, model["coefficients"], strict=True):
            close(a, b)
        counts = {g: sum(r["group"] == g for r in train) for g in intercepts}
        loss = 0.0
        for r, weight in zip(train, model["weights"], strict=True):
            close(weight, 1 / (len(counts) * counts[r["group"]]))
            loss += weight * (intercepts[r["group"]] + slope_for[r["group"]] * r["x"] - r["y"]) ** 2
        close(loss, model["train_loss"])
        subset = [
            r
            for r in (predictions if "held_level" in model else full_predictions)
            if r["arm"] == model["arm"]
            and ("held_level" not in model or r["held_level"] == model["held_level"])
        ]
        for r in subset:
            if "held_level" in model:
                assert r["candidate_id"] not in model["train_ids"]
                assert r["nominal_level_id"] == model["held_level"]
                assert r["train_ids_sha256"] == model["train_ids_sha256"]
            close(intercepts[r["group"]] + slope_for[r["group"]] * r["x"], r["prediction"])
    summary = read(ROOT / "results/summary.json")
    rank_sensitivity = []
    for arm, reports in summary["by_arm"].items():
        selected = [r for r in predictions if r["arm"] == arm]
        alternate = verify_metrics(selected, reports["overall"])
        rank_sensitivity.append(
            {
                "arm": arm,
                "scope": "overall",
                "scalar_centered_spearman": alternate,
                "reported_centered_spearman": reports["overall"]["centered_spearman"],
            }
        )
        for field in ("method", "standard_id", "nominal_level_id"):
            for key, report in reports[field].items():
                alternate = verify_metrics([r for r in selected if r[field] == key], report)
                rank_sensitivity.append(
                    {
                        "arm": arm,
                        "scope": field + ":" + key,
                        "scalar_centered_spearman": alternate,
                        "reported_centered_spearman": report["centered_spearman"],
                    }
                )
    # Exact report computation is separately checked from independently reconstructed fits.
    # Only centered means use NumPy to recover its archived arithmetic convention.
    import numpy as np

    for entry in rank_sensitivity:
        selected = [r for r in predictions if r["arm"] == entry["arm"]]
        if entry["scope"] != "overall":
            field, key = entry["scope"].split(":", 1)
            selected = [r for r in selected if r[field] == key]
        y = np.array([r["y"] for r in selected])
        p = np.array([r["prediction"] for r in selected])
        for g in {r["group"] for r in selected}:
            ids = [i for i, r in enumerate(selected) if r["group"] == g]
            y[ids] -= y[ids].mean()
            p[ids] -= p[ids].mean()
        close(corr(ranks(y.tolist()), ranks(p.tolist())), entry["reported_centered_spearman"])
    base, shared = summary["by_arm"]["constant"], summary["by_arm"]["shared_slope"]
    gain = 1 - shared["overall"]["equal_group_rmse"] / base["overall"]["equal_group_rmse"]
    close(gain, summary["shared_slope_gain"])
    slopes = [m["coefficients"][-1] for m in models if m["arm"] == "shared_slope"]
    assert summary["fold_shared_slopes"] == slopes
    gates = {
        "at_least_10_percent_macro_rmse_gain": gain >= 0.1,
        "all_five_training_slopes_positive": len(slopes) == 5 and min(slopes) > 0,
        "centered_prediction_spread_at_least_25_percent": shared["overall"]["centered_spread_ratio"]
        >= 0.25,
        "neither_method_worse": all(
            shared["method"][m]["equal_group_rmse"] <= base["method"][m]["equal_group_rmse"]
            for m in base["method"]
        ),
    }
    assert gates == summary["gates"] and summary["all_gates_pass"] == all(gates.values())
    result = {
        "passed": True,
        "fits_replayed": 24,
        "predictions_replayed": len(predictions) + len(full_predictions),
        "numeric_checks": CHECKS,
        "maximum_absolute_error": MAX_ERROR,
        "coefficient_prediction_metric_tolerance": 2e-10,
        "rank_arithmetic_sensitivity": rank_sensitivity,
        "scientific_scope": "conditional nominal-dose development only",
        "full_goal_achieved": False,
    }
    write(ROOT / "results/verification.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k != "rank_arithmetic_sensitivity"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
