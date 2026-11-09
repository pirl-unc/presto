"""Complete-fragment branching with one explicit fragment-length rate effect.

Rates are effective hypotheses, not calibrated enzyme or cellular concentrations.
Every contiguous product, including single residues and intact parents, remains.
"""

import math
from dataclasses import dataclass

import torch
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class FragmentExpectationResult:
    counts: torch.Tensor  # time, start, inclusive last residue; lower triangle zero
    poisson_steps: int
    operator_applications: int
    expected_count_error_bound: float


@dataclass(frozen=True)
class FragmentPathSamples:
    cuts: torch.Tensor
    log_rate_score: torch.Tensor
    length_score: torch.Tensor
    log_density: torch.Tensor
    event_counts: torch.Tensor
    event_bonds: torch.Tensor | None = None
    event_times: torch.Tensor | None = None


def _check_parameters(log_rates, eta):
    if log_rates.ndim != 1 or not log_rates.numel():
        raise ValueError("One finite log rate per internal bond is required")
    if eta.ndim != 0 or eta.device != log_rates.device or eta.dtype != log_rates.dtype:
        raise ValueError("Length effect must be a scalar with the rates' dtype/device")
    if not bool(torch.isfinite(log_rates).all() and torch.isfinite(eta)):
        raise ValueError("Finite rate parameters and length effect are required")


def _geometry(log_rates, eta, anchor):
    if not math.isfinite(anchor) or anchor <= 0:
        raise ValueError("Length anchor must be positive and finite")
    size = log_rates.numel() + 1
    index = torch.arange(size, device=log_rates.device)
    lengths = (index[None, :] - index[:, None] + 1).to(log_rates.dtype)
    valid = lengths > 0
    safe_length = torch.where(valid, lengths, 1.0)
    multiplier = (safe_length / anchor).pow(eta) * valid
    rates = log_rates.exp()
    if not bool(torch.isfinite(rates).all() and (rates > 0).all()):
        raise ValueError("Exponential rates must remain positive and finite")
    if not bool(torch.isfinite(multiplier).all()):
        raise ValueError("Fragment-length multipliers must remain finite")
    prefix = torch.cat((rates.new_zeros(1), rates.cumsum(0)))
    internal = torch.where(valid, prefix[None, :] - prefix[:, None], 0.0)
    exit_rates = multiplier * internal
    left = torch.cat((rates.new_zeros(1), rates))[:, None]
    right = torch.cat((rates, rates.new_zeros(1)))[None, :]
    return safe_length, valid, multiplier, exit_rates, left, right


def fragment_count_derivative(counts, log_rates, eta, *, anchor=20.0):
    """Linear first-moment action in O(L^2) without a dense L^2-by-L^2 matrix."""
    _check_parameters(log_rates, eta)
    length, valid, multiplier, exit_rates, left, right = _geometry(log_rates, eta, anchor)
    if counts.shape != length.shape:
        raise ValueError("Expected a full parent-length square span matrix")
    weighted = multiplier * counts
    from_longer_end = weighted.flip(1).cumsum(1).flip(1) - weighted
    from_earlier_start = weighted.cumsum(0) - weighted
    return (right * from_longer_end + left * from_earlier_start - exit_rates * counts) * valid


def _poisson_order(mean, tail_budget):
    # P(Poisson(mean)>=k) <= exp(-mean+k*(1+log(mean/k))) for k>mean.
    last = max(0, math.ceil(mean))
    while True:
        k = last + 1
        log_bound = -mean + k * (1 + math.log(mean / k))
        if log_bound <= math.log(tail_budget):
            return last, math.exp(log_bound)
        last += 1


def _uniform_step(mass, log_rates, eta, dt, rate_bound, order, anchor):
    length, valid, multiplier, exit_rates, left, right = _geometry(log_rates, eta, anchor)
    diagonal = 1 - exit_rates / rate_bound
    mean = rate_bound * dt
    weight = math.exp(-mean)
    term = mass
    result = weight * mass
    for k in range(1, order + 1):
        weighted_counts = multiplier * (term / length)
        gain = right * (weighted_counts.flip(1).cumsum(1).flip(1) - weighted_counts)
        gain = gain + left * (weighted_counts.cumsum(0) - weighted_counts)
        term = diagonal * term + valid * length * gain / rate_bound
        weight *= mean / k
        result = result + weight * term
    return result


def fragment_expectations(
    log_rates,
    eta,
    times,
    *,
    anchor=20.0,
    absolute_error=1e-12,
    max_poisson_mean=8.0,
    checkpoint_steps=True,
):
    """Differentiable expected counts with a positive uniformization expansion.

    Counts are expected copies per initial intact parent. The error bound covers
    Poisson truncation in exact arithmetic, not floating-point roundoff. A
    dominating rate is detached; the exact exponential is independent of that
    numerical choice. No final-state renormalization or probability floor occurs.
    """
    _check_parameters(log_rates, eta)
    values = [float(t) for t in times]
    if not values or any(not math.isfinite(t) or t < 0 for t in values):
        raise ValueError("Nonnegative finite observation times are required")
    if any(b < a for a, b in zip(values, values[1:])):
        raise ValueError("Observation times must be nondecreasing")
    if not 0 < absolute_error < 1 or not 0 < max_poisson_mean <= 64:
        raise ValueError("Invalid numerical error budget or Poisson subdivision")
    length, _, _, exits, _, _ = _geometry(log_rates, eta, anchor)
    rate_bound = float(exits.detach().max())
    if not math.isfinite(rate_bound) or rate_bound <= 0:
        raise ValueError("Exit-rate bound must be positive and finite")
    size = len(log_rates) + 1
    intervals = [b - a for a, b in zip([0.0, *values[:-1]], values)]
    subdivisions = [
        max(1, math.ceil(rate_bound * dt / max_poisson_mean)) if dt else 0 for dt in intervals
    ]
    total_steps = sum(subdivisions)
    mass = log_rates.new_zeros((size, size))
    mass[0, -1] = 1
    output = []
    applications = 0
    bound = 0.0
    for dt, steps in zip(intervals, subdivisions):
        if steps:
            subtime = dt / steps
            order, tail = _poisson_order(
                rate_bound * subtime, absolute_error / (size * total_steps)
            )
            for _ in range(steps):
                if (
                    checkpoint_steps
                    and torch.is_grad_enabled()
                    and (log_rates.requires_grad or eta.requires_grad)
                ):
                    mass = checkpoint(
                        _uniform_step,
                        mass,
                        log_rates,
                        eta,
                        subtime,
                        rate_bound,
                        order,
                        anchor,
                        use_reentrant=False,
                    )
                else:
                    mass = _uniform_step(mass, log_rates, eta, subtime, rate_bound, order, anchor)
                applications += order
                bound += size * tail
        output.append(mass * size / length)
    return FragmentExpectationResult(torch.stack(output), total_steps, applications, bound)


def sample_fragment_paths(
    log_rates, eta, time_hours, count, *, generator, anchor=20.0, record_paths=False
):
    """Full continuous-time branching paths and exact likelihood scores.

    The score includes event hazards, waiting times and terminal survival. Random
    choices are detached; returned scores can be contracted with sequence-rate
    features for REINFORCE. No observed-fragment conditioning is performed.
    """
    _check_parameters(log_rates, eta)
    if not math.isfinite(time_hours) or time_hours < 0 or not isinstance(count, int) or count < 1:
        raise ValueError("Nonnegative finite time and positive integer count required")
    if not math.isfinite(anchor) or anchor <= 0:
        raise ValueError("Positive finite length anchor required")
    with torch.no_grad():
        rates = log_rates.detach().exp()
        if not bool(torch.isfinite(rates).all() and (rates > 0).all()):
            raise ValueError("Exponential rates must remain positive and finite")
        power = eta.detach()
        bonds = len(rates)
        size = bonds + 1
        cuts = torch.zeros((count, bonds), dtype=torch.bool, device=rates.device)
        rate_score = rates.new_zeros((count, bonds))
        length_score = rates.new_zeros(count)
        density = rates.new_zeros(count)
        elapsed = rates.new_zeros(count)
        event_bonds = (
            torch.full((count, bonds), -1, dtype=torch.long, device=rates.device)
            if record_paths
            else None
        )
        event_times = rates.new_zeros((count, bonds)) if record_paths else None
        positions = torch.arange(size + 1, device=rates.device)[None, :]
        active = (
            torch.arange(count, device=rates.device)
            if time_hours
            else torch.empty(0, dtype=torch.long, device=rates.device)
        )
        for event_index in range(bonds):
            if not active.numel():
                break
            boundary = torch.cat(
                (
                    torch.ones((len(active), 1), dtype=torch.bool, device=rates.device),
                    cuts[active],
                    torch.ones((len(active), 1), dtype=torch.bool, device=rates.device),
                ),
                1,
            )
            previous = torch.where(boundary, positions, 0).cummax(1).values[:, :bonds]
            following = (
                torch.where(boundary, positions, size).flip(1).cummin(1).values.flip(1)[:, 1:size]
            )
            log_length = ((following - previous).to(rates.dtype) / anchor).log()
            hazards = rates[None, :] * (power * log_length).exp() * (~cuts[active])
            if not bool(torch.isfinite(hazards).all()):
                raise ValueError("Path hazards must remain finite")
            total = hazards.sum(1)
            positive = total > 0
            active = active[positive]
            hazards, total, log_length = hazards[positive], total[positive], log_length[positive]
            if not active.numel():
                break
            waiting = (
                -(
                    1
                    - torch.rand(
                        len(active), dtype=rates.dtype, device=rates.device, generator=generator
                    )
                ).log()
                / total
            )
            remaining = time_hours - elapsed[active]
            duration = torch.minimum(waiting, remaining)
            integrated = hazards * duration[:, None]
            rate_score[active] -= integrated
            length_score[active] -= (integrated * log_length).sum(1)
            density[active] -= total * duration
            event = waiting < remaining
            event_rows = active[event]
            if event_rows.numel():
                chosen = torch.multinomial(hazards[event], 1, generator=generator).squeeze(1)
                index = torch.arange(len(chosen), device=rates.device)
                cuts[event_rows, chosen] = True
                rate_score[event_rows, chosen] += 1
                length_score[event_rows] += log_length[event][index, chosen]
                density[event_rows] += hazards[event][index, chosen].log()
                elapsed[event_rows] += waiting[event]
                if record_paths:
                    event_bonds[event_rows, event_index] = chosen
                    event_times[event_rows, event_index] = elapsed[event_rows]
            active = event_rows
    return FragmentPathSamples(
        cuts, rate_score, length_score, density, cuts.sum(1), event_bonds, event_times
    )
