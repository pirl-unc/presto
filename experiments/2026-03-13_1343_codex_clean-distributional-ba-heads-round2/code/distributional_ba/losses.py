"""Loss functions for distributional and regression BA heads.

Standalone functions — heads compose these inside ``compute_loss``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Regression losses
# ---------------------------------------------------------------------------

def mhcflurry_censored_loss(
    pred_bounded: torch.Tensor,
    target_bounded: torch.Tensor,
    qual_flipped: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MSE in MHCflurry bounded [0,1] space with censor awareness.

    In bounded space, larger values = stronger binding, so censor directions
    are flipped relative to nM space (caller must flip quals before passing).

    qual_flipped: -1 means "true < threshold" (stronger), 0 exact, 1 "true > threshold" (weaker).
    """
    mse = (pred_bounded - target_bounded) ** 2
    less_loss = F.relu(pred_bounded - target_bounded) ** 2
    greater_loss = F.relu(target_bounded - pred_bounded) ** 2

    is_exact = (qual_flipped == 0).float()
    is_less = (qual_flipped == -1).float()
    is_greater = (qual_flipped == 1).float()

    loss = is_exact * mse + is_less * less_loss + is_greater * greater_loss
    return _masked_mean(loss, mask)


def log_censored_mse(
    pred_y: torch.Tensor,
    target_y: torch.Tensor,
    qual: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Censored MSE in log1p space.

    qual: -1 means "true < threshold" (lower IC50), 0 exact, 1 "true > threshold" (higher IC50).
    In log1p space, larger values = weaker binding — same direction as nM quals.
    """
    mse = (pred_y - target_y) ** 2
    less_loss = F.relu(pred_y - target_y) ** 2
    greater_loss = F.relu(target_y - pred_y) ** 2

    is_exact = (qual == 0).float()
    is_less = (qual == -1).float()
    is_greater = (qual == 1).float()

    loss = is_exact * mse + is_less * less_loss + is_greater * greater_loss
    return _masked_mean(loss, mask)


# ---------------------------------------------------------------------------
# Distributional losses
# ---------------------------------------------------------------------------

def distributional_cross_entropy(
    logits: torch.Tensor,
    target_vector: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy between predicted logits and a soft target distribution.

    Args:
        logits: (B, K) raw logits.
        target_vector: (B, K) target probabilities (sum ~1 per row).

    Returns:
        (B,) per-example cross-entropy.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_vector * log_probs).sum(dim=-1)


def survival_nll(
    probs: torch.Tensor,
    edges: torch.Tensor,
    threshold_y: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Discrete survival NLL for censored distributional observations.

    For a ">" censored sample (direction=1, true value >= threshold):
      loss = -log P(Y >= threshold)

    For a "<" censored sample (direction=-1, true value <= threshold):
      loss = -log P(Y <= threshold)

    Args:
        probs: (B, K) bin probabilities.
        edges: (B, K+1) or (K+1,) bin edges in target space.
        threshold_y: (B,) censor threshold in target space.
        direction: (B,) +1 for ">", -1 for "<".

    Returns:
        (B,) per-example survival NLL.
    """
    B, K = probs.shape

    # Broadcast edges to (B, K+1) if shared
    if edges.dim() == 1:
        edges = edges.unsqueeze(0).expand(B, -1)

    # Find which bin contains the threshold: k_c such that edges[k_c] <= threshold < edges[k_c+1]
    # searchsorted returns index where threshold would be inserted to keep sorted
    # We want the bin index, so subtract 1 and clamp.
    bin_idx = torch.searchsorted(edges.contiguous(), threshold_y.unsqueeze(-1)).squeeze(-1) - 1
    bin_idx = bin_idx.clamp(0, K - 1)

    # Fraction of the containing bin that is above the threshold
    left_edge = edges.gather(1, bin_idx.unsqueeze(-1)).squeeze(-1)
    right_edge = edges.gather(1, (bin_idx + 1).clamp(max=K).unsqueeze(-1)).squeeze(-1)
    bin_width = (right_edge - left_edge).clamp(min=1e-8)
    frac_above = ((right_edge - threshold_y) / bin_width).clamp(0, 1)
    frac_below = 1.0 - frac_above

    # P(Y >= threshold) = frac_above * probs[k_c] + sum(probs[k_c+1:])
    # Build cumulative sum from right
    cum_right = probs.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])  # (B, K)
    # cum_right[:, k] = sum of probs[:, k:]
    # P(above) = frac_above * probs[k_c] + cum_right[k_c+1]  (or 0 if k_c == K-1)
    prob_at_bin = probs.gather(1, bin_idx.unsqueeze(-1)).squeeze(-1)
    idx_plus1 = (bin_idx + 1).clamp(max=K - 1)
    cum_above_next = cum_right.gather(1, idx_plus1.unsqueeze(-1)).squeeze(-1)
    # If bin_idx == K-1, there's nothing above
    at_last = (bin_idx == K - 1).float()
    prob_above = frac_above * prob_at_bin + (1 - at_last) * cum_above_next

    # P(Y <= threshold) = frac_below * probs[k_c] + sum(probs[:k_c])
    cum_left = probs.cumsum(dim=-1)  # (B, K), cum_left[:, k] = sum probs[:, :k+1]
    idx_minus1 = (bin_idx - 1).clamp(min=0)
    cum_below_prev = cum_left.gather(1, idx_minus1.unsqueeze(-1)).squeeze(-1)
    at_first = (bin_idx == 0).float()
    prob_below = frac_below * prob_at_bin + (1 - at_first) * cum_below_prev

    # Select based on direction
    is_greater = (direction == 1).float()
    is_less = (direction == -1).float()
    survival_prob = is_greater * prob_above + is_less * prob_below

    return -torch.log(survival_prob.clamp(min=1e-8))


# ---------------------------------------------------------------------------
# Gaussian losses
# ---------------------------------------------------------------------------

_LOG_2PI = math.log(2.0 * math.pi)


def gaussian_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Per-example Gaussian NLL: 0.5 * (log(2*pi*sigma^2) + ((y-mu)/sigma)^2).

    Args:
        mu: (B,) predicted mean.
        sigma: (B,) predicted std dev (> 0).
        target: (B,) true value.

    Returns:
        (B,) per-example NLL.
    """
    sigma_safe = sigma.clamp(min=1e-6)
    return 0.5 * (_LOG_2PI + 2.0 * torch.log(sigma_safe) + ((target - mu) / sigma_safe) ** 2)


def censored_gaussian_nll(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    threshold_y: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Censored normal (Tobit) NLL.

    For direction=1 (true >= threshold): loss = -log Phi((mu - threshold) / sigma)
    For direction=-1 (true <= threshold): loss = -log Phi((threshold - mu) / sigma)

    Args:
        mu: (B,) predicted mean.
        sigma: (B,) predicted std dev.
        threshold_y: (B,) censor threshold in target space.
        direction: (B,) +1 for ">", -1 for "<".

    Returns:
        (B,) per-example censored NLL.
    """
    sigma_safe = sigma.clamp(min=1e-6)

    # z = (mu - threshold) / sigma for direction=1
    # z = (threshold - mu) / sigma for direction=-1
    z = direction.float() * (threshold_y - mu) / sigma_safe
    # P(correct side) = Phi(-z) = 1 - Phi(z)
    # Actually: for direction=1, P(Y >= t) = Phi((mu - t)/sigma)
    # z_for_phi = (mu - threshold) / sigma * direction ... let me be precise

    # direction=1: want P(Y >= t) = Phi((mu - t)/sigma)
    # direction=-1: want P(Y <= t) = Phi((t - mu)/sigma)
    # Unified: Phi(direction * (mu - t) / sigma) ... no
    # direction=1: arg = (mu - t)/sigma
    # direction=-1: arg = (t - mu)/sigma = -(mu - t)/sigma
    # So arg = -direction * (t - mu) / sigma = direction * (mu - t) / sigma ... wait
    # direction=1: (mu - t)/sigma ✓
    # direction=-1: (t - mu)/sigma = -1 * (mu - t)/sigma ... so arg = -direction * (mu - t)/sigma?
    # No: direction=-1: arg = (t - mu)/sigma = (-1) * (mu - t)/sigma
    # And we want: direction * (mu - t) / sigma?  For dir=1: (mu-t)/sigma ✓. For dir=-1: -(mu-t)/sigma = (t-mu)/sigma ✓
    # Wait that's wrong. direction * (mu - t) / sigma for dir=-1 gives -1*(mu-t)/sigma = (t-mu)/sigma ✓

    phi_arg = direction.float() * (mu - threshold_y) / sigma_safe
    # Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    log_phi = torch.log((0.5 * (1.0 + torch.erf(phi_arg / math.sqrt(2.0)))).clamp(min=1e-8))
    return -log_phi


# ---------------------------------------------------------------------------
# Quantile losses
# ---------------------------------------------------------------------------

def censored_pinball_loss(
    quantiles: torch.Tensor,
    target: torch.Tensor,
    direction: torch.Tensor,
    tau: torch.Tensor,
) -> torch.Tensor:
    """Censored pinball (quantile) loss.

    For exact samples (direction=0): standard pinball loss.
    For right-censored (direction=1, true >= threshold):
      if q < t: loss = tau * (t - q)  (know true residual >= t-q)
      if q >= t: loss = 0  (can't determine)
    For left-censored (direction=-1, true <= threshold):
      if q > t: loss = (1-tau) * (q - t)
      if q <= t: loss = 0

    Args:
        quantiles: (B, Q) predicted quantiles.
        target: (B,) true/threshold value.
        direction: (B,) qualifier: 0 exact, 1 right-censored, -1 left-censored.
        tau: (Q,) quantile levels.

    Returns:
        (B, Q) per-example per-quantile loss.
    """
    Q = quantiles.shape[1]
    target_exp = target.unsqueeze(-1).expand_as(quantiles)  # (B, Q)
    tau_exp = tau.unsqueeze(0).expand_as(quantiles)          # (B, Q)
    dir_exp = direction.unsqueeze(-1).expand_as(quantiles)   # (B, Q)

    residual = target_exp - quantiles  # positive when under-predicting

    # Standard pinball loss
    pinball = torch.where(
        residual >= 0,
        tau_exp * residual,
        (tau_exp - 1.0) * residual,
    )

    # Right-censored (dir=1): only penalize if q < t
    right_loss = tau_exp * F.relu(target_exp - quantiles)

    # Left-censored (dir=-1): only penalize if q > t
    left_loss = (1.0 - tau_exp) * F.relu(quantiles - target_exp)

    is_exact = (dir_exp == 0).float()
    is_right = (dir_exp == 1).float()
    is_left = (dir_exp == -1).float()

    return is_exact * pinball + is_right * right_loss + is_left * left_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _masked_mean(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of loss over masked (valid) elements."""
    mask_f = mask.float()
    return (loss * mask_f).sum() / mask_f.sum().clamp(min=1.0)
