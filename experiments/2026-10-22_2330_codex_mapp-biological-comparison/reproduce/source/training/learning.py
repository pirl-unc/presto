"""Conditional observed-signal objective; unobserved mass is unidentified."""

import math

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def observed_log_probs(model, sequence, example):
    context = example["context"] if model.context_size else ()
    state = model.prepare(sequence, context)
    starts, lengths, groups = [], [], []
    for i, row in enumerate(example["candidates"]):
        for start in row["start_positions_0based"]:
            starts.append(start)
            lengths.append(len(row["sequence"]))
            groups.append(i)
    raw = model.log_prob(state, starts, lengths)
    membership = torch.tensor(groups, dtype=torch.long, device=raw.device)
    peptide = torch.stack(
        [
            torch.logsumexp(raw[membership == i], dim=0)
            for i in range(len(example["candidates"]))
        ]
    )
    result = peptide.log_softmax(0)
    work = {
        "start_rows": len(state.start_log_probs),
        "conditional_rows": len(set(starts)) * model.width,
        "selected_occurrences": len(starts),
        "selected_candidates": len(example["candidates"]),
    }
    return result, peptide.logsumexp(0), work


def reverse_kl(logq, logt):
    return (logq.exp() * (logq - logt)).sum()


def reinforce_loss(logq, logt, *, count, generator):
    if count < 2:
        raise ValueError("Leave-one-out requires at least two draws")
    indices = torch.multinomial(
        logq.detach().exp(), count, replacement=True, generator=generator
    )
    cost = (logq[indices] - logt[indices]).detach()
    baseline = (cost.sum() - cost) / (count - 1)
    return ((cost - baseline) * logq[indices]).mean(), indices


def estimator_check():
    logits = torch.tensor([-0.7, 0.2, 1.1], dtype=torch.float64, requires_grad=True)
    logq = logits.log_softmax(0)
    logt = torch.tensor([0.2, 0.5, 0.3], dtype=torch.float64).log()
    exact = torch.autograd.grad(reverse_kl(logq, logt), logits, retain_graph=True)[0]
    expectation = torch.zeros_like(logits)
    for i in range(3):
        for j in range(3):
            indices = torch.tensor([i, j])
            cost = (logq[indices] - logt[indices]).detach()
            surrogate = ((cost - cost.flip(0)) * logq[indices]).mean()
            gradient = torch.autograd.grad(surrogate, logits, retain_graph=True)[0]
            expectation += (logq[i] + logq[j]).detach().exp() * gradient
    error = float((exact - expectation).abs().max())
    assert error < 1e-12
    return {
        "max_abs_gradient_error": error,
        "n_exhaustive_pairs": 9,
        "exact_gradient": exact.tolist(),
        "expected_score_gradient": expectation.tolist(),
    }


def correlation(a, b, kind):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
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
