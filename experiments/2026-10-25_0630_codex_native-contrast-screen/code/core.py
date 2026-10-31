"""Small antisymmetric linear response models and explicit weighted evaluation."""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from features import raw_features

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())


def dump(path, value):
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def inner_fold(group):
    token = (CFG["inner_namespace"] + "|" + group).encode()
    return int(hashlib.sha256(token).hexdigest(), 16) % 3


def weights(rows):
    w = np.array([r["weight"] for r in rows], dtype=np.float64)
    assert len(w) and np.all(np.isfinite(w)) and np.all(w > 0)
    return w / w.sum()


def differences(rows, kind):
    return np.array(
        [
            np.subtract(
                raw_features(r["peptide_sequence"], kind),
                raw_features(r["reference_peptide_sequence"], kind),
            )
            for r in rows
        ],
        dtype=np.float64,
    )


def fit(rows, kind, alpha):
    assert rows and all(r["split"] == "train" for r in rows)
    x, w = differences(rows, kind), weights(rows)
    y = np.array([r["target_log10_ratio_change"] for r in rows], dtype=np.float64)
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and alpha > 0
    scales = np.sqrt(np.sum(w[:, None] * x**2, axis=0))
    active = np.flatnonzero(scales > 0)
    design = x[:, active] / scales[active]
    matrix = design.T @ (w[:, None] * design) + alpha * np.eye(len(active))
    rhs = design.T @ (w * y)
    beta = np.linalg.solve(matrix, rhs)
    values = design @ beta
    coefficients = np.zeros(x.shape[1])
    coefficients[active] = beta
    mse = float(w @ (values - y) ** 2)
    return dict(
        representation=kind,
        alpha=float(alpha),
        scales=scales.tolist(),
        active_coordinates=active.tolist(),
        coefficients=coefficients.tolist(),
        n_raw_features=x.shape[1],
        n_solved_coordinates=len(active),
        design_rank=int(np.linalg.matrix_rank(design)) if len(active) else 0,
        n_nonzero_coefficients=int(np.count_nonzero(beta)),
        stationarity_max_abs=float(np.max(np.abs(matrix @ beta - rhs), initial=0)),
        training_mse=mse,
        penalty=float(alpha * (beta @ beta)),
        training_objective=mse + float(alpha * (beta @ beta)),
        training_example_ids=[r["example_id"] for r in rows],
        training_components=sorted({r["dependency_group_id"] for r in rows}),
    )


def predict(rows, model):
    if model["representation"] == "zero":
        return np.zeros(len(rows))
    active = np.array(model["active_coordinates"], dtype=int)
    x = differences(rows, model["representation"])
    return (x[:, active] / np.array(model["scales"])[active]) @ np.array(
        model["coefficients"]
    )[active]


def select(rows, kind):
    assert rows and all(r["split"] == "train" for r in rows)
    selections, predictions = [], []
    for alpha in CFG["alpha_grid"]:
        losses = []
        for fold in range(3):
            train = [r for r in rows if inner_fold(r["dependency_group_id"]) != fold]
            check = [r for r in rows if inner_fold(r["dependency_group_id"]) == fold]
            assert train and check, "Empty fixed inner fold; do not change the hash"
            assert not {r["dependency_group_id"] for r in train} & {
                r["dependency_group_id"] for r in check
            }
            model = fit(train, kind, alpha)
            values = predict(check, model)
            y = np.array([r["target_log10_ratio_change"] for r in check])
            loss = float(weights(check) @ (values - y) ** 2)
            losses.append(loss)
            selections.append(
                dict(kind=kind, alpha=alpha, fold=fold, loss=loss, fitted=model)
            )
            predictions.extend(
                dict(
                    example_id=r["example_id"],
                    kind=kind,
                    alpha=alpha,
                    fold=fold,
                    target=float(t),
                    prediction=float(p),
                    weight=float(w),
                )
                for r, t, p, w in zip(check, y, values, weights(check), strict=True)
            )
    means = {
        alpha: float(np.mean([r["loss"] for r in selections if r["alpha"] == alpha]))
        for alpha in CFG["alpha_grid"]
    }
    alpha = min(CFG["alpha_grid"], key=lambda a: (means[a], -a))
    return fit(rows, kind, alpha), selections, predictions, means


def correlation(a, b, w):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    ca, cb = a - w @ a, b - w @ b
    return float((w @ (ca * cb)) / np.sqrt((w @ ca**2) * (w @ cb**2)))


def metrics(rows, values):
    y, p, w = (
        np.array([r["target_log10_ratio_change"] for r in rows]),
        np.asarray(values),
        weights(rows),
    )
    assert y.shape == p.shape and np.all(np.isfinite(p))
    mse = float(w @ (p - y) ** 2)
    nonzero = y != 0
    target_sd = float(np.sqrt(w @ (y - w @ y) ** 2))
    prediction_sd = float(np.sqrt(w @ (p - w @ p) ** 2))
    uniform = np.full(len(y), 1 / len(y))
    return dict(
        n_contrasts=len(rows),
        n_components=len({r["dependency_group_id"] for r in rows}),
        loss=mse,
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(w @ np.abs(p - y)),
        pearson=correlation(y, p, w),
        spearman=correlation(rankdata(y), rankdata(p), w),
        unweighted_pearson=correlation(y, p, uniform),
        unweighted_spearman=correlation(rankdata(y), rankdata(p), uniform),
        correlation_undefined_reason="constant target or prediction"
        if np.ptp(y) == 0 or np.ptp(p) == 0
        else None,
        target_mean=float(w @ y),
        prediction_mean=float(w @ p),
        target_sd=target_sd,
        prediction_sd=prediction_sd,
        prediction_target_sd_ratio=prediction_sd / target_sd if target_sd else None,
        signed_change_accuracy=float(
            w[nonzero] @ (np.sign(p[nonzero]) == np.sign(y[nonzero])) / w[nonzero].sum()
        )
        if np.any(nonzero)
        else None,
        signed_change_weight=float(w[nonzero].sum()),
        n_zero_targets=int((~nonzero).sum()),
        zero_target_weight=float(w[~nonzero].sum()),
    )


def bootstrap(rows, predictions):
    groups = sorted({r["dependency_group_id"] for r in rows})
    y, w = np.array([r["target_log10_ratio_change"] for r in rows]), weights(rows)
    errors = {}
    for kind, values in predictions.items():
        errors[kind] = []
        for group in groups:
            mask = np.array([r["dependency_group_id"] == group for r in rows])
            errors[kind].append(
                float(
                    w[mask] @ (np.asarray(values)[mask] - y[mask]) ** 2 / w[mask].sum()
                )
            )
    rng = np.random.Generator(np.random.PCG64(CFG["bootstrap_seed"]))
    sampled = rng.integers(0, len(groups), size=(CFG["bootstrap_draws"], len(groups)))
    rmses = {
        kind: np.sqrt(np.asarray(error)[sampled].mean(axis=1))
        for kind, error in errors.items()
    }
    assert all(np.all(rmses[ref] > 0) for ref in ("zero", "length")), (
        "Undefined bootstrap relative gain"
    )
    gains = {ref: 1 - rmses["sequence"] / rmses[ref] for ref in ("zero", "length")}
    interval = {
        ref: np.quantile(value, [0.025, 0.975], method="linear").tolist()
        for ref, value in gains.items()
    }
    return groups, errors, sampled, rmses, gains, interval


def decision(scores, intervals):
    result = {}
    for part in ("val", "test"):
        gains = {
            ref: 1 - scores[part]["sequence"]["rmse"] / scores[part][ref]["rmse"]
            if scores[part][ref]["rmse"]
            else None
            for ref in ("zero", "length")
        }
        spread = scores[part]["sequence"]["prediction_target_sd_ratio"]
        result[part] = dict(
            gains=gains,
            spread_ratio=spread,
            gain_and_spread_pass=all(
                g is not None and g >= 0.05 for g in gains.values()
            )
            and spread is not None
            and spread >= 0.25,
            paired_gain_intervals=intervals[part],
        )
    result["advance_physical_comparison"] = result["val"]["gain_and_spread_pass"]
    result["test_transfer_supported"] = (
        result["advance_physical_comparison"]
        and result["test"]["gain_and_spread_pass"]
        and all(intervals["test"][ref][0] > 0 for ref in ("zero", "length"))
    )
    result["complete_APC_concentration_model"] = False
    return result
