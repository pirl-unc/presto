"""Paired component uncertainty and fixed all-pair decision."""

import numpy as np

from core import CFG, weights

PRIMARY = "all_pairs:boundary"
REFERENCES = [
    "all_pairs:zero",
    "all_pairs:length",
    "all_pairs:sequence",
    "reference:boundary",
]


def bootstrap(rows, predictions):
    groups = sorted({r["dependency_group_id"] for r in rows})
    y, w = np.array([r["target_log10_ratio_change"] for r in rows]), weights(rows)
    errors = {}
    for key, values in predictions.items():
        errors[key] = []
        for group in groups:
            mask = np.array([r["dependency_group_id"] == group for r in rows])
            errors[key].append(
                float(
                    w[mask] @ (np.asarray(values)[mask] - y[mask]) ** 2 / w[mask].sum()
                )
            )
    indices = np.random.Generator(np.random.PCG64(CFG["bootstrap_seed"])).integers(
        0, len(groups), (CFG["bootstrap_draws"], len(groups))
    )
    rmses = {
        key: np.sqrt(np.array(error)[indices].mean(1)) for key, error in errors.items()
    }
    assert all(np.all(rmses[key] > 0) for key in REFERENCES)
    gains = {key: 1 - rmses[PRIMARY] / rmses[key] for key in REFERENCES}
    intervals = {
        key: np.quantile(value, [0.025, 0.975], method="linear").tolist()
        for key, value in gains.items()
    }
    return groups, errors, indices, rmses, gains, intervals


def decision(pooled, folds, intervals):
    gains = {
        key: 1 - pooled[PRIMARY]["rmse"] / pooled[key]["rmse"] for key in REFERENCES
    }
    spread = pooled[PRIMARY]["prediction_target_sd_ratio"]
    improved = sum(
        folds[str(f)][PRIMARY]["rmse"] < folds[str(f)]["reference:boundary"]["rmse"]
        for f in range(5)
    )
    gates = dict(
        gain_over_all_matched_controls=all(v >= 0.05 for v in gains.values()),
        spread=spread is not None and spread >= 0.25,
        four_of_five_folds=improved >= 4,
        positive_matched_bootstrap_lower_bound=intervals["reference:boundary"][0] > 0,
    )
    passed = all(gates.values())
    return dict(
        gains=gains,
        prediction_target_sd_ratio=spread,
        improved_folds_vs_reference_objective=improved,
        intervals=intervals,
        gates=gates,
        prioritize_independent_confirmation=passed,
        stop_response_only_feature_optimizer_expansion=not passed,
        complete_APC_concentration_model=False,
        evidence_scope="Training-component development; pairs are dependent, outer cohorts remain closed.",
    )
