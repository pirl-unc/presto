"""Positive target-fragment importance weights on complete branching paths."""

import math
from dataclasses import dataclass

import torch


@dataclass
class BoundarySamples:
    cuts: torch.Tensor
    log_weight: torch.Tensor
    original_score: torch.Tensor
    proposal_score: torch.Tensor
    original_log_density: torch.Tensor
    proposal_log_density: torch.Tensor
    boundary_times: torch.Tensor
    boundary_bonds: torch.Tensor
    event_counts: torch.Tensor
    event_bonds: torch.Tensor | None = None
    event_times: torch.Tensor | None = None
    event_forced: torch.Tensor | None = None


def sample(log_rates, eta, start, end, time, count, *, generator, record_paths=False):
    """Force required boundary times uniformly, simulate every other allowed cut.

    No biological support is dropped: this is a proposal for the unnormalized
    original path measure restricted to a target's presence. Importance weights
    restore its original probability. Scores are derivatives at fixed paths,
    never derivatives through random draws. All values remain in float64 logs.
    """
    if log_rates.ndim != 1 or not len(log_rates) or log_rates.dtype != torch.float64:
        raise ValueError("Finite float64 bond log-rates required")
    if not torch.isfinite(log_rates).all() or not math.isfinite(float(eta)):
        raise ValueError("Finite parameters required")
    size = len(log_rates) + 1
    if not (
        isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= size
    ):
        raise ValueError("Target must be a nonempty parent span")
    if not math.isfinite(time) or time < 0 or not isinstance(count, int) or count < 1:
        raise ValueError("Nonnegative time and positive sample count required")
    boundaries = ([start - 1] if start else []) + ([end - 1] if end < size else [])
    if time == 0 and boundaries:
        raise ValueError("Non-intact target has exactly zero probability at zero time")
    with torch.no_grad():
        rates = log_rates.detach().exp()
        if not (torch.isfinite(rates).all() and (rates > 0).all()):
            raise ValueError("Rates must be positive and finite")
        device = rates.device
        power = float(eta)
        bonds = len(rates)
        m = len(boundaries)
        cuts = torch.zeros((count, bonds), dtype=torch.bool, device=device)
        original_score = rates.new_zeros((count, bonds + 1))
        proposal_score = original_score.clone()
        logp = rates.new_zeros(count)
        logq = rates.new_full((count,), -m * math.log(time) if m else 0.0)
        logw = -logq.clone()
        raw_times = (
            torch.rand(
                (count, m), generator=generator, device=device, dtype=rates.dtype
            )
            * time
        )
        boundary_times, order = raw_times.sort(dim=1)
        boundary_bonds = (
            torch.tensor(boundaries, device=device, dtype=torch.long)
            .expand(count, m)
            .gather(1, order)
        )
        schedule_times = torch.cat(
            (boundary_times, rates.new_full((count, 1), time)), 1
        )
        schedule_bonds = torch.cat(
            (
                boundary_bonds,
                torch.full((count, 1), -1, device=device, dtype=torch.long),
            ),
            1,
        )
        pointer = torch.zeros(count, dtype=torch.long, device=device)
        elapsed = rates.new_zeros(count)
        events = torch.zeros(count, dtype=torch.long, device=device)
        trace_bonds = (
            torch.full((count, bonds), -1, device=device, dtype=torch.long)
            if record_paths
            else None
        )
        trace_times = rates.new_zeros((count, bonds)) if record_paths else None
        trace_forced = (
            torch.zeros((count, bonds), device=device, dtype=torch.bool)
            if record_paths
            else None
        )
        allowed = torch.ones(bonds, device=device, dtype=torch.bool)
        allowed[start : end - 1] = False
        if m:
            allowed[torch.tensor(boundaries, device=device)] = False
        positions = torch.arange(size + 1, device=device)[None, :]
        active = torch.arange(count, device=device)
        for _ in range(bonds + 1):
            if not active.numel():
                break
            state = cuts[active]
            edges = torch.cat(
                (
                    torch.ones((len(active), 1), device=device, dtype=torch.bool),
                    state,
                    torch.ones((len(active), 1), device=device, dtype=torch.bool),
                ),
                1,
            )
            left = torch.where(edges, positions, 0).cummax(1).values[:, :bonds]
            right = (
                torch.where(edges, positions, size)
                .flip(1)
                .cummin(1)
                .values.flip(1)[:, 1:size]
            )
            log_length = ((right - left).to(rates.dtype) / 20).log()
            hazards = rates[None, :] * torch.exp(power * log_length) * (~state)
            if not torch.isfinite(hazards).all():
                raise ValueError("Nonfinite path hazards")
            qhazards = hazards * allowed
            qtotal = qhazards.sum(1)
            draw = torch.rand(
                len(active), generator=generator, device=device, dtype=rates.dtype
            )
            waiting = torch.where(qtotal > 0, -torch.log1p(-draw) / qtotal, torch.inf)
            next_time = schedule_times[active, pointer[active]]
            to_schedule = next_time - elapsed[active]
            assert (to_schedule >= 0).all()
            natural = waiting < to_schedule
            dt = torch.minimum(waiting, to_schedule)
            original_score[active, :bonds] -= hazards * dt[:, None]
            original_score[active, -1] -= (hazards * log_length).sum(1) * dt
            proposal_score[active, :bonds] -= qhazards * dt[:, None]
            proposal_score[active, -1] -= (qhazards * log_length).sum(1) * dt
            logp[active] -= hazards.sum(1) * dt
            logq[active] -= qtotal * dt
            logw[active] -= (hazards * (~allowed)).sum(1) * dt
            elapsed[active] += dt
            selected = schedule_bonds[active, pointer[active]].clone()
            if natural.any():
                selected[natural] = torch.multinomial(
                    qhazards[natural], 1, generator=generator
                ).squeeze(1)
            is_event = selected >= 0
            local = torch.nonzero(is_event).squeeze(1)
            rows = active[local]
            chosen = selected[local]
            if len(rows):
                assert (~cuts[rows, chosen]).all()
                hazard = hazards[local, chosen]
                ll = log_length[local, chosen]
                logp[rows] += hazard.log()
                original_score[rows, chosen] += 1
                original_score[rows, -1] += ll
                forced = ~natural[local]
                if forced.any():
                    logw[rows[forced]] += hazard[forced].log()
                free = ~forced
                if free.any():
                    logq[rows[free]] += hazard[free].log()
                    proposal_score[rows[free], chosen[free]] += 1
                    proposal_score[rows[free], -1] += ll[free]
                if record_paths:
                    trace_bonds[rows, events[rows]] = chosen
                    trace_times[rows, events[rows]] = elapsed[rows]
                    trace_forced[rows, events[rows]] = forced
                events[rows] += 1
                cuts[rows, chosen] = True
            pointer[active[~natural]] += 1
            active = active[is_event]
        assert not active.numel()
        assert not cuts[:, start : end - 1].any()
        if m:
            assert cuts[:, boundaries].all()
        return BoundarySamples(
            cuts,
            logw,
            original_score,
            proposal_score,
            logp,
            logq,
            boundary_times,
            boundary_bonds,
            events,
            trace_bonds,
            trace_times,
            trace_forced,
        )
