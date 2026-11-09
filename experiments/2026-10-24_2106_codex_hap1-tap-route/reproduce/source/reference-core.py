"""Fixed features and condition-specific ridge; no held-out outcome access."""

import hashlib
import json
from collections import Counter

import numpy as np
from scipy.stats import pearsonr, spearmanr

AA = "ACDEFGHIKLMNPQRSTVWY"
VARIANTS = [
    "calibrated_precursor",
    "sequence",
    "length",
    "position_p2_c",
    "position_p1_p2_c",
    "sequence_geometry",
    "sequence_flanks",
    "sequence_context",
    "sequence_shuffled_context",
]


def key(row):
    return json.dumps(
        [row["peptide_sequence"], sorted(set(json.loads(row["parent_ids_json"])))],
        separators=(",", ":"),
    )


def weights(rows):
    counts = Counter((r["condition_id"], r["component_id"], r["peptide_sequence"]) for r in rows)
    peptides = Counter((c, g) for c, g, _ in counts)
    groups = Counter(c for c, _ in peptides)
    return np.array(
        [
            1 / (len(groups) * groups[c] * peptides[c, g] * counts[c, g, s])
            for c, g, s in (
                (r["condition_id"], r["component_id"], r["peptide_sequence"]) for r in rows
            )
        ]
    )


def onehot(symbol, alphabet=AA):
    assert symbol in alphabet
    return [float(symbol == a) for a in alphabet]


def sequence_features(s):
    return np.array([len(s), *[s.count(a) / len(s) for a in AA], *onehot(s[0]), *onehot(s[-1])])


def context_features(s, parent_ids, parents):
    by_parent, lineage = [], []
    for pid in parent_ids:
        parent = parents[pid]
        values = []
        start = parent.find(s)
        while start >= 0:
            end = start + len(s)
            left = parent[start - 1] if start else "^"
            right = parent[end] if end < len(parent) else "^"
            value = [
                np.log1p(start),
                np.log1p(len(parent) - end),
                float(start == 0),
                float(end == len(parent)),
                *onehot(left, AA + "^"),
                *onehot(right, AA + "^"),
            ]
            values.append(value)
            lineage.append(
                {
                    "parent_id": pid,
                    "start": start,
                    "end": end,
                    "parent_length": len(parent),
                    "left": left,
                    "right": right,
                }
            )
            start = parent.find(s, start + 1)
        assert values, (s, pid)
        by_parent.append(np.mean(values, axis=0))
        for item in lineage:
            if item["parent_id"] == pid:
                item["weight"] = 1 / len(parent_ids) / len(values)
    return np.mean(by_parent, axis=0), lineage


def donor_map(keys):
    groups = {}
    for value in sorted(set(keys)):
        groups.setdefault(len(json.loads(value)[0]), []).append(value)
    result = {}
    for group in groups.values():
        ordered = sorted(
            group,
            key=lambda k: hashlib.sha256(("hap1_context_shuffle_v1:" + k).encode()).hexdigest(),
        )
        result.update(zip(ordered, ordered[1:] + ordered[:1], strict=True))
    return result


def design(rows, variant, contexts):
    keys = [key(r) for r in rows]
    donors = donor_map(keys) if variant == "sequence_shuffled_context" else {}
    output = []
    for r, k in zip(rows, keys, strict=True):
        s = r["peptide_sequence"]
        if variant == "calibrated_precursor":
            features = []
        elif variant == "length":
            features = [len(s)]
        elif variant == "position_p2_c":
            features = [len(s), *onehot(s[1]), *onehot(s[-1])]
        elif variant == "position_p1_p2_c":
            features = [len(s), *onehot(s[0]), *onehot(s[1]), *onehot(s[-1])]
        else:
            features = sequence_features(s).tolist()
            if variant == "sequence_geometry":
                features.extend(contexts[k][:4])
            elif variant == "sequence_flanks":
                features.extend(contexts[k][4:])
            elif variant in {"sequence_context", "sequence_shuffled_context"}:
                features.extend(contexts[donors.get(k, k)])
        output.append(features)
    return np.array(output, dtype=float), donors


def fit(rows, eval_rows, phi, phie):
    y = np.array([r["target_log10_ratio"] for r in rows])
    x = np.array([r["precursor_log10_ratio"] for r in rows])
    xe = np.array([r["precursor_log10_ratio"] for r in eval_rows])
    c = np.array([r["condition_id"] for r in rows])
    ce = np.array([r["condition_id"] for r in eval_rows])
    w = weights(rows)
    means = {
        cond: [float(w[c == cond] @ v[c == cond] / w[c == cond].sum()) for v in [y, x]]
        for cond in sorted(set(c))
    }
    yr = y - np.array([means[cond][0] for cond in c])
    xr = x - np.array([means[cond][1] for cond in c])
    gamma = max(0.0, float(w @ (xr * yr)) / float(w @ xr**2)) if w @ xr**2 > 0 else 0.0
    offsets = {cond: my - gamma * mx for cond, (my, mx) in means.items()}
    pred = gamma * xe + np.array([offsets[cond] for cond in ce])
    checkpoint = {"gamma": gamma, "offsets": offsets, "n_fit": len(rows), "penalty": 0.1}
    mean = w @ phi
    sd = np.sqrt(w @ (phi - mean) ** 2)
    mask = sd > 1e-12
    z, ze = (phi[:, mask] - mean[mask]) / sd[mask], (phie[:, mask] - mean[mask]) / sd[mask]
    checkpoint.update(mean=mean.tolist(), sd=sd.tolist(), mask=mask.tolist(), coefficients={})
    residual = y - gamma * x - np.array([offsets[cond] for cond in c])
    for cond in means:
        select, evaluate = c == cond, ce == cond
        wc = w[select] / w[select].sum()
        mu = wc @ z[select]
        a = z[select] - mu
        gram = a.T @ (wc[:, None] * a) + 0.1 * np.eye(mask.sum())
        rhs = a.T @ (wc * residual[select])
        beta = np.linalg.solve(gram, rhs)
        pred[evaluate] += (ze[evaluate] - mu) @ beta
        checkpoint["coefficients"][cond] = {
            "center": mu.tolist(),
            "beta": beta.tolist(),
            "normal_equation_residual": float(np.max(abs(gram @ beta - rhs), initial=0)),
            "supported": int(np.sum(wc @ a**2 > 1e-12)),
            "nonzero": int(np.sum(abs(beta) > 1e-12)),
        }
    assert np.isfinite(pred).all()
    return pred, checkpoint


def metrics(y, p, w):
    w = w / w.sum()
    mse = float(w @ (y - p) ** 2)
    var = float(w @ (y - w @ y) ** 2)
    varying = np.ptp(p) > 1e-14 and np.ptp(y) > 1e-14
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(w @ abs(y - p)),
        "r2": 1 - mse / var if var else None,
        "pearson": float(pearsonr(y, p).statistic) if varying else None,
        "spearman": float(spearmanr(y, p).statistic) if varying else None,
        "prediction_sd": float(np.sqrt(w @ (p - w @ p) ** 2)),
        "target_sd": float(np.sqrt(var)),
        "n_observations": len(y),
    }
