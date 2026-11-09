"""Unchanged prior conditional mixture functions, extracted from the pinned source."""
import itertools
import math
import torch
from presto.models.independent_cut import log_fragment_expectation, partition_contains

def weights(record, kind):
    n = len(record["mass"])
    if kind == "uniform":
        return torch.full((n,), 1 / n, dtype=torch.float64)
    score = record["coefficient"].abs() * record["conditional"].norm(dim=1)
    if float(score.sum()) == 0:
        return torch.full((n,), 1 / n, dtype=torch.float64)
    return 0.9 * score / score.sum() + 0.1 / n

def draw(record, count, generator, w):
    choices = torch.multinomial(w, count, replacement=True, generator=generator)
    probability = -torch.expm1(-record["logs"].exp() * record["time"])
    cuts = (
        torch.rand((count, len(probability)), dtype=torch.float64, generator=generator)
        < probability
    )
    positions = torch.arange(len(probability))[None, :]
    starts, ends = record["starts"][choices, None], record["ends"][choices, None]
    cuts &= ~((positions >= starts) & (positions < ends - 1))
    cuts |= (positions == starts - 1) | (positions == ends - 1)
    return cuts, choices

def estimate(records, count, generator, kind):
    gradient = torch.zeros(records[0]["conditional"].shape[1], dtype=torch.float64)
    zero = events = partitions = bonds = terms = 0
    minimum = math.inf
    maximum = 0.0
    for r in records:
        w = weights(r, kind)
        cuts, chosen = draw(r, count, generator, w)
        present = partition_contains(cuts, r["starts"], r["ends"])
        assert present[torch.arange(count), chosen].all()
        indicators = present.double()
        density = indicators @ (w / r["mass"])
        assert torch.isfinite(density).all() and (density > 0).all()
        importance_counts = (indicators / density[:, None]).mean(0)
        gradient += (importance_counts * r["coefficient"] / r["mass"]) @ r[
            "conditional"
        ]
        zero += int((present.sum(0) == 0).sum())
        events += present.shape[1]
        partitions += count
        bonds += cuts.numel()
        terms += present.numel()
        minimum = min(minimum, float(density.min()))
        maximum = max(maximum, float(density.max()))
    return gradient, dict(
        zero_span_times=zero,
        span_times=events,
        partitions=partitions,
        bond_draws=bonds,
        presence_terms=terms,
        minimum_q_over_p=minimum,
        maximum_q_over_p=maximum,
    )

def toy_check(config):
    rates = torch.tensor([0.07, 0.13, 0.21, 0.1, 0.15], dtype=torch.float64)
    t = 2.1
    logs = rates.log()
    starts = torch.tensor([0, 1, 2])
    ends = torch.tensor([3, 4, 6])
    mass = log_fragment_expectation(
        logs, starts, ends, torch.full((3,), t, dtype=torch.float64)
    ).exp()
    coefficient = torch.tensor([0.3, -0.7, 0.4], dtype=torch.float64)
    hazard = rates * t
    cutscore = hazard * torch.exp(-hazard) / (-torch.expm1(-hazard))
    score = torch.zeros((3, 5), dtype=torch.float64)
    for j, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
        score[j, a : b - 1] = -hazard[a : b - 1]
        if a:
            score[j, a - 1] = cutscore[a - 1]
        if b <= 5:
            score[j, b - 1] = cutscore[b - 1]
    record = dict(
        logs=logs,
        time=t,
        starts=starts,
        ends=ends,
        mass=mass,
        coefficient=coefficient,
        conditional=score,
    )
    states = torch.tensor(
        list(itertools.product([False, True], repeat=5)), dtype=torch.bool
    )
    p = torch.where(states, -torch.expm1(-hazard), torch.exp(-hazard)).prod(1)
    indicators = partition_contains(states, starts, ends).double()
    exact = coefficient @ score
    results = []
    for kind in ("uniform", "gradient_mixture_90_10"):
        w = weights(record, kind)
        density = indicators @ (w / mass)
        q = p * density
        valid = q > 0
        assert abs(float(q.sum()) - 1) < 1e-12
        rb = (
            (q[valid] / density[valid])[:, None]
            * indicators[valid]
            * (coefficient / mass)[None, :]
        ).sum(0) @ score
        fullscore = torch.where(states, cutscore, -hazard)
        reward = indicators @ (coefficient / mass)
        full = (
            (q[valid] / density[valid] * reward[valid])[:, None] * fullscore[valid]
        ).sum(0)
        error = max(float((rb - exact).abs().max()), float((full - exact).abs().max()))
        assert error < 1e-12
        cuts, _ = draw(
            record,
            config["toy_draws"],
            torch.Generator().manual_seed(config["seed"]),
            w,
        )
        codes = (cuts.long() * torch.tensor([16, 8, 4, 2, 1])[None, :]).sum(1)
        observed = torch.bincount(codes, minlength=32).double()
        se = torch.sqrt(config["toy_draws"] * q * (1 - q))
        standard = float(
            ((observed[valid] - config["toy_draws"] * q[valid]).abs() / se[valid]).max()
        )
        assert (
            standard < config["toy_max_standard_errors"]
            and (observed[~valid] == 0).all()
        )
        results.append(
            dict(
                kind=kind,
                gradient_expectation_error=error,
                q=q.tolist(),
                draw_counts=observed.tolist(),
                maximum_binomial_standard_errors=standard,
            )
        )
    return results
