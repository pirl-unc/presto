"""Evaluation metrics for distributional BA heads.

Point metrics (discrimination) and calibration metrics.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch


def point_metrics(
    pred_nM: torch.Tensor,
    true_nM: torch.Tensor,
    mask: torch.Tensor,
    threshold_nM: float = 500.0,
) -> Dict[str, float]:
    """Compute point-prediction metrics on valid samples.

    Returns Spearman, Pearson, RMSE (log10 nM), plus binary classification
    metrics at the given threshold.
    """
    m = mask.bool()
    if m.sum() < 2:
        return {}

    p = pred_nM[m].float()
    t = true_nM[m].float()

    # Log10 space for regression metrics
    p_log = torch.log10(p.clamp(min=1e-3))
    t_log = torch.log10(t.clamp(min=1e-3))

    rmse_log10 = float(((p_log - t_log) ** 2).mean().sqrt())
    pearson = float(_pearson(p_log, t_log))
    spearman = float(_spearman(p_log, t_log))

    # Binary classification at threshold
    pred_bind = (p <= threshold_nM).float()
    true_bind = (t <= threshold_nM).float()

    tp = (pred_bind * true_bind).sum()
    fp = (pred_bind * (1 - true_bind)).sum()
    fn = ((1 - pred_bind) * true_bind).sum()
    tn = ((1 - pred_bind) * (1 - true_bind)).sum()

    accuracy = float((tp + tn) / (tp + tn + fp + fn).clamp(min=1))
    precision = float(tp / (tp + fp).clamp(min=1))
    recall = float(tp / (tp + fn).clamp(min=1))
    f1 = float(2 * precision * recall / max(precision + recall, 1e-8))

    n_pos = float(true_bind.sum())
    n_neg = float((1 - true_bind).sum())
    bal_acc = 0.5 * (recall + float(tn / max(n_neg, 1)))

    # AUROC — simple trapezoidal on sorted predictions
    auroc = float(_auroc(p, true_bind))
    auprc = float(_auprc(p, true_bind))

    return {
        "spearman": spearman,
        "pearson": pearson,
        "rmse_log10": rmse_log10,
        "accuracy": accuracy,
        "balanced_accuracy": bal_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "n_samples": int(m.sum()),
    }


def calibration_metrics(
    probs: torch.Tensor,
    edges: torch.Tensor,
    true_y: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Calibration metrics for distributional heads.

    Args:
        probs: (B, K) predicted bin probabilities.
        edges: (B, K+1) or (K+1,) bin edges.
        true_y: (B,) true values in log1p space.
        mask: (B,) valid-sample mask.

    Returns:
        PIT KS statistic, coverage at 90%, entropy-error SRCC.
    """
    m = mask.bool()
    if m.sum() < 5:
        return {}

    probs_m = probs[m]
    true_m = true_y[m]
    if edges.dim() == 1:
        edges_m = edges.unsqueeze(0).expand(probs_m.shape[0], -1)
    else:
        edges_m = edges[m]

    B, K = probs_m.shape

    # PIT: probability integral transform
    # CDF(y) = sum of probs for bins with right edge <= y, plus partial for containing bin
    cdf = probs_m.cumsum(dim=-1)  # (B, K)
    # For each sample, find which bin contains true_y
    bin_idx = torch.searchsorted(edges_m.contiguous(), true_m.unsqueeze(-1)).squeeze(-1) - 1
    bin_idx = bin_idx.clamp(0, K - 1)

    left = edges_m.gather(1, bin_idx.unsqueeze(-1)).squeeze(-1)
    right = edges_m.gather(1, (bin_idx + 1).clamp(max=K).unsqueeze(-1)).squeeze(-1)
    bw = (right - left).clamp(min=1e-8)
    frac = ((true_m - left) / bw).clamp(0, 1)

    # CDF at left edge of containing bin
    prev_idx = (bin_idx - 1).clamp(min=0)
    cdf_left = cdf.gather(1, prev_idx.unsqueeze(-1)).squeeze(-1)
    cdf_left = torch.where(bin_idx == 0, torch.zeros_like(cdf_left), cdf_left)

    prob_at = probs_m.gather(1, bin_idx.unsqueeze(-1)).squeeze(-1)
    pit = cdf_left + frac * prob_at  # (B,)

    # KS test: max deviation from uniform
    pit_sorted = pit.sort().values
    n = pit_sorted.shape[0]
    uniform = torch.linspace(0, 1, n + 1, device=pit.device)[1:]
    ks_stat = float((pit_sorted - uniform).abs().max())

    # Coverage at 90%: fraction of true values within 90% prediction interval
    # Find bins at 5th and 95th CDF percentiles using searchsorted
    # lo_bin: first bin where cumulative prob >= 0.05 → use its left edge
    # hi_bin: first bin where cumulative prob >= 0.95 → use its right edge
    cdf_contiguous = cdf.contiguous()
    lo_bin = torch.searchsorted(cdf_contiguous, torch.full((B,), 0.05, device=probs_m.device).unsqueeze(-1)).squeeze(-1)
    hi_bin = torch.searchsorted(cdf_contiguous, torch.full((B,), 0.95, device=probs_m.device).unsqueeze(-1)).squeeze(-1)
    lo_bin = lo_bin.clamp(0, K - 1)
    hi_bin = hi_bin.clamp(0, K - 1)
    lo_edge = edges_m.gather(1, lo_bin.unsqueeze(-1)).squeeze(-1)            # left edge of lo_bin
    hi_edge = edges_m.gather(1, (hi_bin + 1).clamp(max=K).unsqueeze(-1)).squeeze(-1)  # right edge of hi_bin
    covered = ((true_m >= lo_edge) & (true_m <= hi_edge)).float()
    coverage_90 = float(covered.mean())

    # Entropy-error SRCC
    entropy = -(probs_m * torch.log(probs_m.clamp(min=1e-10))).sum(dim=-1)
    ev = (probs_m * (edges_m[:, :-1] + edges_m[:, 1:]) * 0.5).sum(dim=-1)
    error = (ev - true_m).abs()
    entropy_error_srcc = float(_spearman(entropy, error))

    return {
        "pit_ks": ks_stat,
        "coverage_90": coverage_90,
        "entropy_error_srcc": entropy_error_srcc,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_c = x - x.mean()
    y_c = y - y.mean()
    num = (x_c * y_c).sum()
    den = (x_c.pow(2).sum() * y_c.pow(2).sum()).sqrt().clamp(min=1e-8)
    return num / den


def _spearman(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return _pearson(_rank(x), _rank(y))


def _rank(t: torch.Tensor) -> torch.Tensor:
    order = t.argsort()
    ranks = torch.empty_like(t)
    ranks[order] = torch.arange(len(t), dtype=t.dtype, device=t.device)
    return ranks


def _auroc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Simple AUROC via sorted thresholds (lower score = predicted positive for binding)."""
    if labels.sum() == 0 or labels.sum() == len(labels):
        return 0.5
    # For binding: lower IC50 = stronger binder = positive
    # We want to discriminate binders from non-binders.
    # Use negative score so higher = more likely positive.
    neg_scores = -scores
    order = neg_scores.argsort(descending=True)
    sorted_labels = labels[order]
    tps = sorted_labels.cumsum(dim=0)
    fps = (1 - sorted_labels).cumsum(dim=0)
    tpr = tps / tps[-1].clamp(min=1)
    fpr = fps / fps[-1].clamp(min=1)
    # Trapezoidal rule
    dfpr = torch.cat([fpr[:1], fpr[1:] - fpr[:-1]])
    return float((tpr * dfpr).sum())


def _auprc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Simple AUPRC via sorted thresholds."""
    if labels.sum() == 0:
        return 0.0
    neg_scores = -scores
    order = neg_scores.argsort(descending=True)
    sorted_labels = labels[order]
    tps = sorted_labels.cumsum(dim=0)
    n_pred = torch.arange(1, len(sorted_labels) + 1, dtype=scores.dtype, device=scores.device)
    prec = tps / n_pred
    rec = tps / tps[-1].clamp(min=1)
    drec = torch.cat([rec[:1], rec[1:] - rec[:-1]])
    return float((prec * drec).sum())
