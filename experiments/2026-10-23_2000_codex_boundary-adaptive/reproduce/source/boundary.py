"""State-adaptive target-boundary bridge on complete fragment partitions."""

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


def bernoulli_ratio(x):
    """x / (exp(x)-1), including its zero limit, without large-x overflow."""
    small = x.abs() < 1e-4
    polynomial = 1 - x / 2 + x.square() / 12 - x.pow(4) / 720
    denominator = -torch.expm1(-x)
    denominator = torch.where(small, torch.ones_like(x), denominator)
    regular = x * torch.exp(-x) / denominator
    return torch.where(small, polynomial, regular)


def boundary_segment(hazard, remaining, duration):
    """Log proposal survival and fixed-path log-hazard derivative for one segment."""
    after = remaining - duration
    if not ((hazard > 0).all() and (after > 0).all()):
        raise ValueError(
            "Pending boundaries require positive hazard and remaining time"
        )
    x0 = hazard * remaining
    x1 = hazard * after
    if not ((x0 > 0).all() and (x1 > 0).all() and torch.isfinite(x0).all()):
        raise ValueError("Boundary integrated hazards must be representable")
    normal0 = -torch.expm1(-x0)
    normal1 = -torch.expm1(-x1)
    log_survival = -hazard * duration + normal1.log() - normal0.log()
    score = -hazard * duration + bernoulli_ratio(x1) - bernoulli_ratio(x0)
    return log_survival, score


def sample(log_rates, eta, start, end, time, count, *, generator, record_paths=False):
    """Recompute truncated boundary clocks after every cut; preserve exact p/q."""
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
        boundary_index = torch.tensor(boundaries, device=device, dtype=torch.long)
        cuts = torch.zeros((count, bonds), dtype=torch.bool, device=device)
        original_score = rates.new_zeros((count, bonds + 1))
        proposal_score = original_score.clone()
        logp, logq, logw = (rates.new_zeros(count) for _ in range(3))
        boundary_times = rates.new_zeros((count, m))
        boundary_bonds = torch.full((count, m), -1, dtype=torch.long, device=device)
        boundary_count = torch.zeros(count, dtype=torch.long, device=device)
        elapsed = rates.new_zeros(count)
        events = torch.zeros(count, dtype=torch.long, device=device)
        trace_bonds = (
            torch.full((count, bonds), -1, dtype=torch.long, device=device)
            if record_paths
            else None
        )
        trace_times = rates.new_zeros((count, bonds)) if record_paths else None
        trace_forced = (
            torch.zeros((count, bonds), dtype=torch.bool, device=device)
            if record_paths
            else None
        )
        allowed = torch.ones(bonds, dtype=torch.bool, device=device)
        allowed[start : end - 1] = False
        allowed[boundary_index] = False
        positions = torch.arange(size + 1, device=device)[None, :]
        active = torch.arange(count, device=device)
        for _ in range(bonds + 1):
            if not active.numel():
                break
            state = cuts[active]
            edges = torch.cat(
                (
                    torch.ones((len(active), 1), dtype=torch.bool, device=device),
                    state,
                    torch.ones((len(active), 1), dtype=torch.bool, device=device),
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
            outside = hazards * allowed
            outside_total = outside.sum(1)
            remaining = time - elapsed[active]
            u = torch.rand(
                len(active), generator=generator, device=device, dtype=rates.dtype
            )
            natural_wait = torch.where(
                outside_total > 0, -torch.log1p(-u) / outside_total, torch.inf
            )
            boundary_wait = rates.new_full((len(active),), torch.inf)
            boundary_choice = torch.zeros(len(active), dtype=torch.long, device=device)
            if m:
                pending = ~state[:, boundary_index]
                safe_h = torch.where(pending, hazards[:, boundary_index], 1.0)
                hR = safe_h * remaining[:, None]
                if not ((hR[pending] > 0).all() and torch.isfinite(hR[pending]).all()):
                    raise ValueError(
                        "Boundary integrated hazards must be representable"
                    )
                norm = -torch.expm1(-hR)
                draws = torch.rand(
                    (len(active), m),
                    generator=generator,
                    device=device,
                    dtype=rates.dtype,
                )
                wait = -torch.log1p(-draws * norm) / safe_h
                if not (
                    wait[pending] < remaining[:, None].expand_as(wait)[pending]
                ).all():
                    raise ValueError("Pending boundary rounded to terminal time")
                wait = torch.where(pending, wait, torch.inf)
                boundary_wait, boundary_choice = wait.min(1)
            dt = torch.minimum(torch.minimum(natural_wait, boundary_wait), remaining)
            forced = (boundary_wait <= natural_wait) & (boundary_wait < remaining)
            natural = (natural_wait < boundary_wait) & (natural_wait < remaining)
            selected = torch.full((len(active),), -1, dtype=torch.long, device=device)
            if forced.any():
                selected[forced] = boundary_index[boundary_choice[forced]]
            if natural.any():
                selected[natural] = torch.multinomial(
                    outside[natural], 1, generator=generator
                ).squeeze(1)
            original_score[active, :bonds] -= hazards * dt[:, None]
            original_score[active, -1] -= (hazards * log_length).sum(1) * dt
            proposal_score[active, :bonds] -= outside * dt[:, None]
            proposal_score[active, -1] -= (outside * log_length).sum(1) * dt
            logp[active] -= hazards.sum(1) * dt
            logq[active] -= outside_total * dt
            logw[active] -= (hazards * (~allowed)).sum(1) * dt
            if m:
                # Only pending clocks contribute; inactive placeholders never enter logs.
                rr, cc = pending.nonzero(as_tuple=True)
                if len(rr):
                    h = hazards[rr, boundary_index[cc]]
                    survival, sc = boundary_segment(h, remaining[rr], dt[rr])
                    rows = active[rr]
                    chosen = boundary_index[cc]
                    logq.index_add_(0, rows, survival)
                    logw.index_add_(0, rows, -survival)
                    proposal_score[rows, chosen] += sc
                    proposal_score[:, -1].index_add_(
                        0, rows, sc * log_length[rr, chosen]
                    )
            elapsed[active] += dt
            is_event = forced | natural
            local = is_event.nonzero().squeeze(1)
            rows = active[local]
            chosen = selected[local]
            if len(rows):
                assert (~cuts[rows, chosen]).all()
                h = hazards[local, chosen]
                ll = log_length[local, chosen]
                logp[rows] += h.log()
                original_score[rows, chosen] += 1
                original_score[rows, -1] += ll
                force = forced[local]
                if force.any():
                    fr = rows[force]
                    fc = chosen[force]
                    h_remaining = h[force] * (time - elapsed[fr])
                    normal = -torch.expm1(-h_remaining)
                    assert (normal > 0).all()
                    logq[fr] += h[force].log() - normal.log()
                    logw[fr] += normal.log()
                    sc = 1 - bernoulli_ratio(h_remaining)
                    proposal_score[fr, fc] += sc
                    proposal_score[fr, -1] += sc * ll[force]
                    boundary_times[fr, boundary_count[fr]] = elapsed[fr]
                    boundary_bonds[fr, boundary_count[fr]] = fc
                    boundary_count[fr] += 1
                free = ~force
                if free.any():
                    logq[rows[free]] += h[free].log()
                    proposal_score[rows[free], chosen[free]] += 1
                    proposal_score[rows[free], -1] += ll[free]
                if record_paths:
                    trace_bonds[rows, events[rows]] = chosen
                    trace_times[rows, events[rows]] = elapsed[rows]
                    trace_forced[rows, events[rows]] = force
                events[rows] += 1
                cuts[rows, chosen] = True
            active = active[is_event]
        assert not active.numel()
        assert not cuts[:, start : end - 1].any()
        assert (boundary_count == m).all()
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
