"""Cluster resampling of saved predictions without expanding repeated observations."""

import hashlib
import numpy as np


def ordering(values):
    order = np.argsort(values, kind='stable')
    ordered = np.asarray(values)[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    return order, starts, np.diff(np.r_[starts, len(order)])


def ranks(weights, arrangement):
    order, starts, sizes = arrangement
    totals = np.add.reduceat(weights[order], starts)
    centers = np.cumsum(totals) - (totals - 1) / 2
    result = np.empty(len(weights), dtype=float)
    result[order] = np.repeat(centers, sizes)
    return result


def correlation(y, p, weights):
    total = weights.sum()
    if total <= 1:
        return None
    yy = y - np.dot(weights, y) / total
    pp = p - np.dot(weights, p) / total
    denominator = np.sqrt(np.dot(weights, yy * yy) * np.dot(weights, pp * pp))
    return float(np.dot(weights, yy * pp) / denominator) if denominator > 0 else None


class Metric:
    def __init__(self, targets, predictions, *, binary=False):
        self.y = np.asarray(targets, dtype=float)
        self.p = np.asarray(predictions, dtype=float)
        assert len(self.y) and self.y.shape == self.p.shape
        assert np.isfinite(self.y).all() and np.isfinite(self.p).all()
        self.binary = binary
        self.p_order = ordering(self.p)
        self.y_order = ordering(self.y)
        if binary:
            assert np.isin(self.y, [0, 1]).all()

    def __call__(self, weights):
        weights = np.asarray(weights, dtype=float)
        assert weights.shape == self.y.shape and (weights >= 0).all()
        total = weights.sum()
        assert total > 0
        if not self.binary:
            return dict(
                spearman=correlation(
                    ranks(weights, self.y_order), ranks(weights, self.p_order), weights
                ),
                pearson=correlation(self.y, self.p, weights),
                rmse=float(np.sqrt(np.dot(weights, (self.y - self.p) ** 2) / total)),
            )
        order, starts, _ = self.p_order
        positive = np.add.reduceat((weights * self.y)[order], starts)
        negative = np.add.reduceat((weights * (1 - self.y))[order], starts)
        positives, negatives = positive.sum(), negative.sum()
        auc = None
        if positives and negatives:
            lower_negative = np.cumsum(negative) - negative
            auc = float(np.dot(positive, lower_negative + negative / 2) / positives / negatives)
        ap = None
        if positives:
            pp, nn = positive[::-1], negative[::-1]
            cumulative = np.cumsum(pp + nn)
            precision = np.divide(
                np.cumsum(pp), cumulative, out=np.zeros_like(cumulative), where=cumulative > 0
            )
            ap = float(np.dot(pp, precision) / positives)
        return dict(auroc=auc, auprc=ap)


def bootstrap(metric, clusters, *, minimum_clusters, token, draws=500):
    names, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    point = metric(np.ones(len(inverse)))
    result = dict(n=len(inverse), clusters=len(names), point=point, requested_draws=draws)
    if len(names) < minimum_clusters:
        return dict(**result, status='insufficient_clusters', intervals=None)
    digest = hashlib.sha256(('20261028|' + token).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
    sampled = {key: [] for key in point}
    for _ in range(draws):
        copies = rng.multinomial(len(names), np.full(len(names), 1 / len(names)))
        values = metric(copies[inverse])
        for key, value in values.items():
            if value is not None and np.isfinite(value):
                sampled[key].append(value)
    intervals = {}
    for key, values in sampled.items():
        intervals[key] = dict(
            defined_draws=len(values),
            percentile_95=np.quantile(values, [0.025, 0.975]).tolist()
            if len(values) >= 0.9 * draws
            else None,
        )
    return dict(**result, status='computed', intervals=intervals)
