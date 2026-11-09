"""A bounded independent-cut hypothesis for unnormalized fragment production.

Rates are effective rates per hour, not calibrated proteasome mechanisms. A
fragment exists when its boundaries have cut and none of its internal bonds has.
Every contiguous product, including single residues, remains in the process.
"""

import math

import torch
from torch import nn

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def ordered_bond_features(sequence, *, context=None, dtype=torch.float64):
    """Intercept and 19 left/right contrasts against alanine; optional context."""
    if len(sequence) < 2 or set(sequence) - set(AMINO_ACIDS):
        raise ValueError("Use a canonical full sequence with at least two residues")
    if context is not None and not math.isfinite(context):
        raise ValueError("Context must be finite when supplied")
    result = torch.zeros(
        (len(sequence) - 1, 39 + int(context is not None)), dtype=dtype
    )
    result[:, 0] = 1
    for bond, (left, right) in enumerate(zip(sequence[:-1], sequence[1:])):
        for offset, aa in ((0, left), (19, right)):
            category = AMINO_ACIDS.index(aa)
            if category:
                result[bond, offset + category] = 1
    if context is not None:
        result[:, -1] = context
    return result


class IndependentCutRates(nn.Module):
    """Log-linear positive bond rates; train only supported feature coordinates."""

    def __init__(self, dimensions=39, initial_rate=0.01):
        super().__init__()
        if dimensions < 1 or not math.isfinite(initial_rate) or initial_rate <= 0:
            raise ValueError("Positive initial rate and feature count required")
        self.coefficients = nn.Parameter(torch.zeros(dimensions, dtype=torch.float64))
        with torch.no_grad():
            self.coefficients[0] = math.log(initial_rate)

    def forward(self, features):
        return features @ self.coefficients


def log_fragment_expectation(log_rates, starts, ends, times, *, validate=True):
    """Log expected copies per initial parent for selected [start,end) spans.

    Selected calculations are exact: they need no normalization over observed
    products and no enumeration of exponentially many complete cut states.
    """
    if validate:
        if log_rates.ndim != 1 or not log_rates.numel():
            raise ValueError("Expected one log rate per internal bond")
        if starts.dtype != torch.long or ends.dtype != torch.long:
            raise ValueError("Span coordinates must be int64")
        if starts.shape != ends.shape or starts.shape != times.shape:
            raise ValueError("Supply one positive time for each span")
        if not torch.isfinite(log_rates).all() or not torch.isfinite(times).all():
            raise ValueError("Rates and times must be finite")
        if (
            (times <= 0).any()
            or (starts < 0).any()
            or (ends <= starts).any()
            or (ends > len(log_rates) + 1).any()
        ):
            raise ValueError("Invalid positive-time fragment coordinates")
    rates = log_rates.exp()
    prefix = torch.cat((rates.new_zeros(1), rates.cumsum(0)))
    internal = prefix[ends - 1] - prefix[starts]
    # Sentinel rate avoids log(0) in branches for physical protein termini.
    extended = torch.cat((rates, rates.new_ones(1)))
    left = torch.where(starts > 0, starts - 1, len(rates))
    right = torch.where(ends <= len(rates), ends - 1, len(rates))
    left_log = torch.log(-torch.expm1(-extended[left] * times))
    right_log = torch.log(-torch.expm1(-extended[right] * times))
    return (
        torch.where(starts > 0, left_log, 0.0)
        + torch.where(ends <= len(rates), right_log, 0.0)
        - times * internal
    )


def sample_partitions(log_rates, time_hours, count, *, generator):
    """Return complete Bernoulli cut states and their differentiable log scores."""
    if count < 1 or time_hours <= 0:
        raise ValueError("Positive time and sample count required")
    hazard = log_rates.exp() * time_hours
    probability = -torch.expm1(-hazard)
    cuts = (
        torch.rand(
            (count, len(log_rates)),
            dtype=log_rates.dtype,
            device=log_rates.device,
            generator=generator,
        )
        < probability.detach()
    )
    logp = torch.where(cuts, torch.log(probability), -hazard).sum(-1)
    return cuts, logp


def partition_contains(cuts, starts, ends):
    """Presence of selected fragments in each complete sampled partition."""
    boundary = torch.cat(
        (
            torch.ones((len(cuts), 1), dtype=torch.bool, device=cuts.device),
            cuts,
            torch.ones((len(cuts), 1), dtype=torch.bool, device=cuts.device),
        ),
        dim=1,
    )
    prefix = torch.cat(
        (
            torch.zeros((len(cuts), 1), dtype=torch.long, device=cuts.device),
            cuts.long().cumsum(1),
        ),
        dim=1,
    )
    return (
        boundary[:, starts]
        & boundary[:, ends]
        & (prefix[:, ends - 1] == prefix[:, starts])
    )
