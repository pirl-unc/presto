"""Reconstruct all fits and held-out outputs through a separate SVD/statistics path."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "reproduce/launch.json").read_text())
AA = "ACDEFGHIKLMNPQRSTVWY"


def features(sequence, kind):
    result = [np.log(len(sequence))]
    if kind == "sequence":
        indices = np.array([AA.index(a) for a in sequence])
        result.extend((np.bincount(indices, minlength=20) / len(sequence)).tolist())
        result.extend(np.eye(20)[indices[0]].tolist())
        result.extend(np.eye(20)[indices[-1]].tolist())
    return np.asarray(result)


def w_for(rows):
    values = np.asarray([r["weight"] for r in rows])
    return values / np.sum(values)


def matrix(rows, kind):
    return np.stack(
        [
            features(r["peptide_sequence"], kind)
            - features(r["reference_peptide_sequence"], kind)
            for r in rows
        ]
    )


def fold(group):
    return (
        int.from_bytes(
            hashlib.sha256((CFG["inner_namespace"] + "|" + group).encode()).digest(),
            "big",
        )
        % 3
    )


def independent_fit(rows, kind, alpha):
    x, w = matrix(rows, kind), w_for(rows)
    y = np.asarray([r["target_log10_ratio_change"] for r in rows])
    scale = np.sqrt(np.average(x * x, axis=0, weights=w))
    active = np.where(scale != 0)[0]
    design = x[:, active] / scale[active]
    augmented = np.vstack(
        [np.sqrt(w[:, None]) * design, np.sqrt(alpha) * np.eye(len(active))]
    )
    rhs = np.r_[np.sqrt(w) * y, np.zeros(len(active))]
    beta = np.linalg.lstsq(augmented, rhs, rcond=None)[0]
    return scale, active, beta


def prediction(rows, kind, fitted):
    if kind == "zero":
        return np.zeros(len(rows))
    scale, active, beta = fitted
    return np.einsum("ij,j->i", matrix(rows, kind)[:, active] / scale[active], beta)


def check(a, b, label="value"):
    np.testing.assert_allclose(a, b, rtol=2e-9, atol=2e-10, err_msg=label)


def corr(a, b, weights):
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    covariance = np.cov(a, b, aweights=weights, ddof=0)
    return float(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1]))


def statistics(rows, predicted):
    actual = np.asarray([r["target_log10_ratio_change"] for r in rows])
    w = w_for(rows)
    loss = float(np.average(np.square(actual - predicted), weights=w))
    target_mean, predicted_mean = (
        float(np.average(actual, weights=w)),
        float(np.average(predicted, weights=w)),
    )
    target_sd = float(np.sqrt(np.average((actual - target_mean) ** 2, weights=w)))
    predicted_sd = float(
        np.sqrt(np.average((predicted - predicted_mean) ** 2, weights=w))
    )
    mask = actual != 0
    ranks_y, ranks_p = rankdata(actual), rankdata(predicted)
    return dict(
        n_contrasts=len(rows),
        n_components=len({r["dependency_group_id"] for r in rows}),
        loss=loss,
        mse=loss,
        rmse=float(np.sqrt(loss)),
        mae=float(np.average(np.abs(actual - predicted), weights=w)),
        pearson=corr(actual, predicted, w),
        spearman=corr(ranks_y, ranks_p, w),
        unweighted_pearson=corr(actual, predicted, np.ones(len(rows))),
        unweighted_spearman=corr(ranks_y, ranks_p, np.ones(len(rows))),
        correlation_undefined_reason="constant target or prediction"
        if len(set(actual)) < 2 or len(set(predicted)) < 2
        else None,
        target_mean=target_mean,
        prediction_mean=predicted_mean,
        target_sd=target_sd,
        prediction_sd=predicted_sd,
        prediction_target_sd_ratio=predicted_sd / target_sd if target_sd else None,
        signed_change_accuracy=float(
            np.average(
                np.sign(predicted[mask]) == np.sign(actual[mask]), weights=w[mask]
            )
        )
        if np.any(mask)
        else None,
        signed_change_weight=float(w[mask].sum()),
        n_zero_targets=int((~mask).sum()),
        zero_target_weight=float(w[~mask].sum()),
    )


def compare_dict(expected, observed):
    assert expected.keys() == observed.keys()
    for key, value in expected.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            check(value, observed[key], key)
        else:
            assert value == observed[key], key


package = Path(CFG["package"])
train = pq.read_table(
    package / "typed/examples.parquet", filters=[("split", "=", "train")]
).to_pylist()
saved_models = json.loads((ROOT / "results/models.json").read_text())
saved_selection = json.loads((ROOT / "results/inner-selection.json").read_text())
saved_inner = pq.read_table(ROOT / "results/inner-predictions.parquet").to_pylist()
models, recomputed_inner = {"zero": None}, {}
n_coefficient_sets = 0
for kind in ("length", "sequence"):
    loss_by_alpha = {}
    for alpha in CFG["alpha_grid"]:
        losses = []
        for heldout in range(3):
            training = [r for r in train if fold(r["dependency_group_id"]) != heldout]
            query = [r for r in train if fold(r["dependency_group_id"]) == heldout]
            fitted = independent_fit(training, kind, alpha)
            y = np.asarray([r["target_log10_ratio_change"] for r in query])
            p = prediction(query, kind, fitted)
            loss = float(np.average(np.square(p - y), weights=w_for(query)))
            losses.append(loss)
            record = next(
                r
                for r in saved_selection[kind]["folds"]
                if r["alpha"] == alpha and r["fold"] == heldout
            )
            check(loss, record["loss"])
            check(fitted[0], record["fitted"]["scales"])
            assert fitted[1].tolist() == record["fitted"]["active_coordinates"]
            check(fitted[2], np.asarray(record["fitted"]["coefficients"])[fitted[1]])
            assert record["fitted"]["training_example_ids"] == [
                r["example_id"] for r in training
            ]
            for row, value, weight, target in zip(
                query, p, w_for(query), y, strict=True
            ):
                recomputed_inner[kind, alpha, heldout, row["example_id"]] = (
                    value,
                    weight,
                    target,
                )
            n_coefficient_sets += 1
        loss_by_alpha[alpha] = float(np.mean(losses))
        check(loss_by_alpha[alpha], saved_selection[kind]["mean_losses"][str(alpha)])
    chosen = min(CFG["alpha_grid"], key=lambda alpha: (loss_by_alpha[alpha], -alpha))
    assert chosen == saved_models[kind]["alpha"]
    models[kind] = independent_fit(train, kind, chosen)
    scale, active, beta = models[kind]
    check(scale, saved_models[kind]["scales"])
    assert active.tolist() == saved_models[kind]["active_coordinates"]
    check(beta, np.asarray(saved_models[kind]["coefficients"])[active])
    n_coefficient_sets += 1
assert len(saved_inner) == len(recomputed_inner)
for row in saved_inner:
    value, weight, target = recomputed_inner.pop(
        (row["kind"], row["alpha"], row["fold"], row["example_id"])
    )
    check(value, row["prediction"])
    check(weight, row["weight"])
    assert target == row["target"]
assert not recomputed_inner

saved_metrics = json.loads((ROOT / "results/heldout-metrics.json").read_text())
all_metrics, all_intervals, n_predictions, max_prediction_difference = {}, {}, 0, 0.0
for part in ("train", "val", "test"):
    rows = (
        train
        if part == "train"
        else pq.read_table(
            package / "typed/examples.parquet", filters=[("split", "=", part)]
        ).to_pylist()
    )
    filename = (
        "training-predictions.parquet"
        if part == "train"
        else part + "-predictions.parquet"
    )
    saved = pq.read_table(ROOT / "results" / filename).to_pylist()
    assert len(saved) == 3 * len(rows)
    predictions = {
        kind: prediction(rows, kind, model) for kind, model in models.items()
    }
    by_example = {r["example_id"]: (i, r) for i, r in enumerate(rows)}
    for row in saved:
        index, source = by_example[row["example_id"]]
        assert {k: row[k] for k in source} == source
        expected = predictions[row["condition"]][index]
        check(expected, row["prediction"])
        max_prediction_difference = max(
            max_prediction_difference, abs(expected - row["prediction"])
        )
        n_predictions += 1
    scores = {kind: statistics(rows, p) for kind, p in predictions.items()}
    if part == "train":
        old = json.loads((ROOT / "results/fit-summary.json").read_text())[
            "training_metrics"
        ]
        for kind in models:
            compare_dict(scores[kind], old[kind])
        continue
    all_metrics[part] = scores
    for kind in models:
        compare_dict(scores[kind], saved_metrics[part][kind])
    groups = sorted({r["dependency_group_id"] for r in rows})
    errors = {}
    for kind, p in predictions.items():
        errors[kind] = np.array(
            [
                np.average(
                    [
                        (p[i] - row["target_log10_ratio_change"]) ** 2
                        for i, row in enumerate(rows)
                        if row["dependency_group_id"] == group
                    ],
                    weights=[
                        row["weight"]
                        for row in rows
                        if row["dependency_group_id"] == group
                    ],
                )
                for group in groups
            ]
        )
    saved_errors = json.loads(
        (ROOT / "results" / (part + "-component-errors.json")).read_text()
    )
    assert groups == saved_errors["components"]
    for kind in models:
        check(errors[kind], saved_errors["conditional_mse"][kind])
    rng = np.random.Generator(np.random.PCG64(CFG["bootstrap_seed"]))
    indices = np.stack(
        [
            rng.choice(len(groups), size=len(groups), replace=True)
            for _ in range(CFG["bootstrap_draws"])
        ]
    )
    archive = np.load(ROOT / "results" / (part + "-bootstrap.npz"))
    assert np.array_equal(indices, archive["sampled_component_indices"])
    rmses = {
        kind: np.sqrt(error[indices].sum(axis=1) / len(groups))
        for kind, error in errors.items()
    }
    for kind in models:
        check(rmses[kind], archive["rmse_" + kind])
    all_intervals[part] = {}
    for ref in ("zero", "length"):
        gain = (rmses[ref] - rmses["sequence"]) / rmses[ref]
        check(gain, archive["gain_vs_" + ref])
        all_intervals[part][ref] = np.percentile(
            gain, [2.5, 97.5], method="linear"
        ).tolist()

reported_decision = json.loads((ROOT / "results/decision.json").read_text())
qualified = {}
for part in ("val", "test"):
    gains = {
        ref: (all_metrics[part][ref]["rmse"] - all_metrics[part]["sequence"]["rmse"])
        / all_metrics[part][ref]["rmse"]
        for ref in ("zero", "length")
    }
    spread = all_metrics[part]["sequence"]["prediction_target_sd_ratio"]
    qualified[part] = (
        all(gain >= 0.05 for gain in gains.values())
        and spread is not None
        and spread >= 0.25
    )
    for ref in gains:
        check(gains[ref], reported_decision[part]["gains"][ref])
        check(
            all_intervals[part][ref],
            reported_decision[part]["paired_gain_intervals"][ref],
        )
    check(spread, reported_decision[part]["spread_ratio"])
    assert qualified[part] == reported_decision[part]["gain_and_spread_pass"]
assert reported_decision["advance_physical_comparison"] == qualified["val"]
assert reported_decision["test_transfer_supported"] == (
    qualified["val"]
    and qualified["test"]
    and all(all_intervals["test"][r][0] > 0 for r in ("zero", "length"))
)
assert reported_decision["complete_APC_concentration_model"] is False
result = dict(
    passed=True,
    n_svd_coefficient_sets=n_coefficient_sets,
    n_inner_predictions=len(saved_inner),
    n_predictions=n_predictions,
    max_prediction_abs_difference=float(max_prediction_difference),
    all_metrics_and_bootstrap_draws_verified=True,
    final_decision=reported_decision,
    numerical_tolerance=dict(rtol=2e-9, atol=2e-10),
)
with (ROOT / "results/independent-replay.json").open("x") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(
    "Independent SVD, predictions, metrics and component-bootstrap replay passed",
    flush=True,
)
