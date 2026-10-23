"""Prior composition metrics with an exact constant-centering guard."""

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr


def correlation(a, b, kind):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
        return None
    value = (pearsonr if kind == "pearson" else spearmanr)(a, b).statistic
    return float(value) if np.isfinite(value) else None


def metrics(predictions):
    if not predictions:
        return {"n_examples": 0}
    groups = {}
    rev, forward, tv, mses, all_pred, all_target = [], [], [], [], [], []
    by_parent = {}
    for row in predictions:
        q = np.asarray(row["logq"], dtype=float)
        t = np.asarray(row["logt"], dtype=float)
        rev.append(float(np.sum(np.exp(q) * (q - t))))
        forward.append(float(np.sum(np.exp(t) * (t - q))))
        tv.append(float(np.abs(np.exp(q) - np.exp(t)).sum() / 2))
        centered_q = (q - q.mean()) / math.log(10)
        if np.ptp(q) == 0:
            centered_q = np.zeros_like(q)
        centered_t = (t - t.mean()) / math.log(10)
        mses.append(float(np.mean((centered_q - centered_t) ** 2)))
        all_pred.extend(centered_q)
        all_target.extend(centered_t)
        by_parent[row["parent_sequence_sha256"], row["condition"]] = (
            centered_q,
            centered_t,
            row["peptide_ids"],
        )
    delta_pred, delta_target, delta_mses = [], [], []
    for parent in sorted({key[0] for key in by_parent}):
        a = by_parent[parent, "UT"]
        b = by_parent[parent, "TNF+IFN"]
        assert a[2] == b[2]
        p = b[0] - a[0]
        t = b[1] - a[1]
        delta_pred.extend(p)
        delta_target.extend(t)
        delta_mses.append(float(np.mean((p - t) ** 2)))
    groups.update(
        {
            "n_examples": len(predictions),
            "n_parents": len(delta_mses),
            "n_peptide_condition_predictions": len(all_pred),
            "reverse_kl": float(np.mean(rev)),
            "forward_kl": float(np.mean(forward)),
            "total_variation": float(np.mean(tv)),
            "centered_log10_rmse": math.sqrt(np.mean(mses)),
            "centered_log10_pearson": correlation(all_pred, all_target, "pearson"),
            "centered_log10_spearman": correlation(all_pred, all_target, "spearman"),
            "prediction_centered_log10_sd": float(np.std(all_pred)),
            "target_centered_log10_sd": float(np.std(all_target)),
            "condition_change_log10_rmse": math.sqrt(np.mean(delta_mses)),
            "condition_change_pearson": correlation(
                delta_pred, delta_target, "pearson"
            ),
            "condition_change_spearman": correlation(
                delta_pred, delta_target, "spearman"
            ),
            "condition_change_prediction_sd": float(np.std(delta_pred)),
            "condition_change_target_sd": float(np.std(delta_target)),
        }
    )
    return groups
