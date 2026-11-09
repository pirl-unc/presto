"""Exponential substrate entry followed by complete independent-cut partitions.

The process conserves residue mass. Positive quadrature approximates the entry
time distribution; its accuracy must be qualified for the applied rate domain.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from .independent_cut import log_fragment_expectation


@lru_cache(maxsize=16)
def _rule(order):
    if not isinstance(order, int) or order < 2:
        raise ValueError("Quadrature order must be an integer >= 2")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return (nodes + 1) / 2, weights / weights.sum()


@lru_cache(maxsize=16)
def _laguerre_rule(order):
    if not isinstance(order, int) or order < 2:
        raise ValueError("Quadrature order must be an integer >= 2")
    # Golub-Welsch avoids overflow in the polynomial formula at high orders.
    # Tiny tail weights may underflow to zero; the retained weights are positive.
    off = np.arange(1, order, dtype=float)
    jacobi = np.diag(2 * np.arange(order, dtype=float) + 1)
    jacobi += np.diag(off, 1) + np.diag(off, -1)
    nodes, vectors = np.linalg.eigh(jacobi)
    weights = vectors[0] ** 2
    return nodes, weights / weights.sum()


def log_entry_fragment_expectation(
    log_rates, log_entry_rate, starts, ends, times, *, order=64, validate=True
):
    """Log expected copies per parent, with a shared exponential entry clock.

    Conditional entry-age weights are positive and sum to the analytically
    known entered probability. All spans share these weights at a given time,
    so the quadrature mixture retains residue mass, including unentered parent.
    This is differentiable quadrature, not an exact continuous-time integral.
    """
    if validate:
        if log_entry_rate.numel() != 1 or not torch.isfinite(log_entry_rate).all():
            raise ValueError("Supply one finite scalar log entry rate")
        # Reuse coordinate, dtype and positive-time validation.
        log_fragment_expectation(log_rates, starts, ends, times)
    nodes, weights = _rule(order)
    nodes = torch.as_tensor(nodes, dtype=log_rates.dtype, device=log_rates.device)
    weights = torch.as_tensor(weights, dtype=log_rates.dtype, device=log_rates.device)
    ages = times[:, None] * nodes
    entry_hazard = log_entry_rate.exp() * times
    log_entered = torch.log(-torch.expm1(-entry_hazard))
    log_weights = (
        weights.log()[None, :] - entry_hazard[:, None] * (1 - nodes[None, :])
    ).log_softmax(-1) + log_entered[:, None]
    # A parent-level guard keeps a common mixture for every span at each time.
    # Fast entry concentrates density at age~t; fixed age nodes cannot resolve
    # that limit. Positive entry-time Laguerre nodes resolve it directly.
    fast = (entry_hazard >= 64) & (log_entry_rate.exp() > 2 * log_rates.exp().sum())
    if bool(fast.any()):
        ln, lw = _laguerre_rule(order)
        ln = torch.as_tensor(ln, dtype=log_rates.dtype, device=log_rates.device)
        lw = torch.as_tensor(lw, dtype=log_rates.dtype, device=log_rates.device)
        valid = ln[None, :] < entry_hazard[:, None]
        lag_ages = torch.where(
            valid, times[:, None] - ln[None, :] / log_entry_rate.exp(), times[:, None]
        )
        lag_weights = lw.log()[None, :].expand_as(ages).masked_fill(~valid, -torch.inf)
        lag_weights = lag_weights.log_softmax(-1) + log_entered[:, None]
        ages = torch.where(fast[:, None], lag_ages, ages)
        log_weights = torch.where(fast[:, None], lag_weights, log_weights)
    conditional = log_fragment_expectation(
        log_rates,
        starts[:, None].expand_as(ages),
        ends[:, None].expand_as(ages),
        ages,
        validate=False,
    )
    entered = torch.logsumexp(log_weights + conditional, dim=-1)
    intact = (starts == 0) & (ends == len(log_rates) + 1)
    unentered = torch.where(intact, -entry_hazard, -torch.inf)
    return torch.logaddexp(entered, unentered)


@dataclass(frozen=True)
class EntryPartitions:
    cuts: torch.Tensor
    entry_times: torch.Tensor
    entered: torch.Tensor
    ages: torch.Tensor
    log_prob: torch.Tensor


def sample_entry_partitions(log_rates, log_entry_rate, time_hours, count, *, generator):
    """Draw continuous entry clocks and discrete complete cut partitions.

    Return score-function log probabilities with sampled event times detached.
    An unentered trajectory has no cuts and retains the complete parent.
    """
    if count < 1 or time_hours <= 0 or log_entry_rate.numel() != 1:
        raise ValueError("Positive time/count and scalar log entry rate required")
    uniform = torch.rand(count, dtype=log_rates.dtype, device=log_rates.device, generator=generator)
    entry_times = -torch.log1p(-uniform) / log_entry_rate.detach().exp()
    entered = entry_times < time_hours
    ages = torch.where(entered, time_hours - entry_times, 0.0)
    # Safe placeholders avoid log(0) in unselected autograd branches.
    safe_ages = torch.where(entered, ages, 1.0)
    hazard = safe_ages[:, None] * log_rates.exp()[None, :]
    probability = -torch.expm1(-hazard)
    draws = torch.rand(
        (count, len(log_rates)),
        dtype=log_rates.dtype,
        device=log_rates.device,
        generator=generator,
    )
    cuts = (draws < probability.detach()) & entered[:, None]
    conditional = torch.where(cuts, probability.log(), -hazard).sum(-1)
    entry_score = torch.where(
        entered,
        log_entry_rate - log_entry_rate.exp() * entry_times,
        -log_entry_rate.exp() * time_hours,
    )
    return EntryPartitions(
        cuts, entry_times, entered, ages, entry_score + torch.where(entered, conditional, 0.0)
    )
