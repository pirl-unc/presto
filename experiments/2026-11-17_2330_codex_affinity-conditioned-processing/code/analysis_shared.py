"""Previously verified paired multiplicity bootstrap, shared by this family."""
import numpy as np


def weighted_auc_draws(target, scores, row_weights):
    """Exact AUROC of expanded rows using multiplicities and half-credit ties."""
    order = np.argsort(scores, kind="stable")
    sorted_scores = np.asarray(scores)[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    y = np.asarray(target)[order]
    weights = row_weights[:, order]
    positive = np.add.reduceat(weights * y, starts, axis=1)
    negative = np.add.reduceat(weights * (1 - y), starts, axis=1)
    return (positive * (np.cumsum(negative, axis=1) - 0.5 * negative)).sum(axis=1) / (
        positive.sum(axis=1) * negative.sum(axis=1)
    )


def complete_pairs(a):
    """A retained source pair ID alone does not establish a paired cohort."""
    if a.empty or a.pair.isna().any():
        return False
    counts = a.groupby("pair").target.agg(["count", "sum", "nunique"])
    return bool(((counts["count"] == 2) & (counts["sum"] == 1) & (counts["nunique"] == 2)).all())


def paired_draws(a, cluster_draws, cluster_names):
    """Match the original paired cluster bootstrap without expanding rows."""
    lookup = {name: i for i, name in enumerate(cluster_names)}
    counts = np.stack([np.bincount(draw, minlength=len(cluster_names)) for draw in cluster_draws])
    row_weights = counts[:, [lookup[v] for v in a.bootstrap_group]]
    auc_delta = weighted_auc_draws(a.target, a.prediction_real, row_weights) - weighted_auc_draws(
        a.target, a.prediction_other, row_weights
    )
    if not complete_pairs(a):
        return np.column_stack([auc_delta, np.full(len(auc_delta), np.nan)])
    pairs = a.pivot(
        index="pair", columns="target", values=["prediction_real", "prediction_other"]
    ).dropna()
    groups = a.pivot(index="pair", columns="target", values="bootstrap_group").loc[pairs.index]
    assert (groups[0.0] == groups[1.0]).all()
    pair_weights = counts[:, [lookup[v] for v in groups[1.0]]]
    wins = {
        k: ((pairs[k][1.0] > pairs[k][0.0]) + 0.5 * (pairs[k][1.0] == pairs[k][0.0])).to_numpy()
        for k in ("prediction_real", "prediction_other")
    }
    pair_delta = (pair_weights * (wins["prediction_real"] - wins["prediction_other"])).sum(
        axis=1
    ) / pair_weights.sum(axis=1)
    return np.column_stack([auc_delta, pair_delta])
