"""Independent inference and reductions; no producer model imports or fitting."""

import math
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import rankdata

AA = "ACDEFGHIKLMNPQRSTVWY"


def direct(rows, cp):
    if cp["variant"] == "zero":
        return np.zeros(len(rows))
    values = np.array(
        [
            cp["condition_offsets"][r["condition_id"]] + cp["gamma"] * r["precursor_log10_ratio"]
            for r in rows
        ]
    )
    if "sequence" in cp:
        active = np.flatnonzero(cp["feature_mask"])
        mean = np.asarray(cp["feature_mean"])[active]
        sd = np.asarray(cp["feature_sd"])[active]
        design = []
        for row in rows:
            p = row["peptide_sequence"]
            count = Counter(p)
            design.append(
                [
                    len(p),
                    *[count[a] / len(p) for a in AA],
                    *[int(p.startswith(a)) for a in AA],
                    *[int(p.endswith(a)) for a in AA],
                ]
            )
        design = np.asarray(design)[:, active]
        for condition, params in cp["sequence"].items():
            ids = [i for i, r in enumerate(rows) if r["condition_id"] == condition]
            beta = np.array(params["beta"])
            affine_intercept = -np.sum(beta * (mean / sd + params["center"]))
            values[ids] += design[ids] @ (beta / sd) + affine_intercept
    return values


def physical(rows, origins, parents, cp):
    """Concentration recurrence, independent of the log-path producer."""
    p = cp["parameter"]
    rates = [math.exp(x) for x in (p[:1] * 20 if len(p) in (2, 4) else p[:20])]
    modifiers = [math.exp(p[-3]), 1.0, math.exp(p[-2])] if len(p) in (4, 23) else [1.0] * 3
    scale = 1 / (1 + math.exp(-p[-1]))
    total = defaultdict(lambda: [0.0, 0.0])
    for origin in origins:
        seq = parents[origin["parent_id"]]
        a, b = origin["start0"], origin["end0"]
        assert seq[a:b] == origin["peptide_sequence"]
        wt = ko = previous_wt = previous_ko = 0.0
        for j in range(a + 1):
            length = b - j
            boundary = int(j != 0) + int(b != len(seq))
            birth = (
                1 / length
                if boundary == 0
                else 1 / (length * (length + 1))
                if boundary == 1
                else 2 / (length * (length + 1) * (length + 2))
            )
            band = 0 if length <= 8 else 1 if length <= 16 else 2
            rate = rates[AA.index(seq[j])] * modifiers[band] if length > 1 else 0.0
            wt = (birth + previous_wt * wt) / (1 + rate)
            ko = (birth + previous_ko * ko) / (1 + scale * rate)
            previous_wt, previous_ko = rate, scale * rate
        total[origin["peptide_sequence"]][0] += wt
        total[origin["peptide_sequence"]][1] += ko * 10 ** (
            cp["gamma"] * origin["parent_log10_response"]
        )
    ratios = {p: math.log10(v[1] / v[0]) for p, v in total.items()}
    return np.array([ratios[r["peptide_sequence"]] for r in rows])


def weights(rows):
    tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for i, row in enumerate(rows):
        tree[row["condition_id"]][row["component_id"]][row["peptide_sequence"]].append(i)
    result = np.zeros(len(rows))
    for components in tree.values():
        for peptides in components.values():
            for ids in peptides.values():
                result[ids] = 1 / (len(tree) * len(components) * len(peptides) * len(ids))
    return result


def metrics(y, p, weights):
    w = weights / np.sum(weights)
    error = p - y
    mse = float(np.sum(w * error * error))
    variance = float(np.sum(w * (y - np.sum(w * y)) ** 2)) if np.ptp(y) else 0.0

    def correlation(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    varying = len(y) > 1 and np.ptp(p) > 1e-14 and np.ptp(y) > 0
    return {
        "n_observations": len(y),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(np.sum(w * np.abs(error))),
        "r2": 1 - mse / variance if variance else None,
        "pearson": correlation(y, p) if varying else None,
        "spearman": correlation(rankdata(y), rankdata(p)) if varying else None,
        "prediction_sd": math.sqrt(float(np.sum(w * (p - np.sum(w * p)) ** 2)))
        if np.ptp(p)
        else 0.0,
        "target_sd": math.sqrt(variance),
    }
