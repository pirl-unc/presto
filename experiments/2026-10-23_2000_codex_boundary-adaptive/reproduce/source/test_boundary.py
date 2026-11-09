import math

import pytest
import torch

from boundary import sample


def test_single_boundary_exact_density_and_scores():
    rate = 0.13
    horizon = 3.0
    result = sample(
        torch.tensor([math.log(rate)], dtype=torch.float64),
        0.0,
        0,
        1,
        horizon,
        32,
        generator=torch.Generator().manual_seed(3),
        record_paths=True,
    )
    t = result.boundary_times[:, 0]
    normalizer = -math.expm1(-rate * horizon)
    score = rate * horizon * math.exp(-rate * horizon) / normalizer
    torch.testing.assert_close(
        result.log_weight, torch.full_like(t, math.log(normalizer))
    )
    torch.testing.assert_close(result.original_score[:, 0], 1 - rate * t)
    torch.testing.assert_close(
        result.original_score[:, 1], (1 - rate * t) * math.log(2 / 20)
    )
    torch.testing.assert_close(result.proposal_score[:, 0], 1 - rate * t - score)
    torch.testing.assert_close(
        result.proposal_score[:, 1], (1 - rate * t - score) * math.log(2 / 20)
    )
    assert result.cuts.all() and result.event_forced[:, 0].all()


def test_intact_no_cut_survival_and_zero_time():
    rates = torch.tensor([0.02, 0.07, 0.11], dtype=torch.float64)
    eta = 0.4
    horizon = 7.0
    result = sample(
        rates.log(),
        eta,
        0,
        4,
        horizon,
        8,
        generator=torch.Generator().manual_seed(4),
        record_paths=True,
    )
    hazards = rates * (4 / 20) ** eta
    torch.testing.assert_close(
        result.log_weight,
        torch.full_like(result.log_weight, -float(hazards.sum()) * horizon),
    )
    torch.testing.assert_close(
        result.original_score[:, :3], (-hazards * horizon).expand(8, -1)
    )
    assert not result.cuts.any() and not result.event_counts.any()
    zero = sample(
        rates.log(), eta, 0, 4, 0.0, 8, generator=torch.Generator().manual_seed(5)
    )
    assert not zero.log_weight.any() and not zero.original_score.any()
    with pytest.raises(ValueError, match="zero probability"):
        sample(
            rates.log(), eta, 1, 3, 0.0, 8, generator=torch.Generator().manual_seed(5)
        )


def test_full_partition_and_density_ratio():
    result = sample(
        torch.tensor([0.02, 0.07, 0.11, 0.03, 0.15], dtype=torch.float64).log(),
        -0.4,
        1,
        4,
        8.0,
        100,
        generator=torch.Generator().manual_seed(6),
        record_paths=True,
    )
    assert result.cuts[:, 0].all() and result.cuts[:, 3].all()
    assert not result.cuts[:, 1:3].any()
    torch.testing.assert_close(
        result.log_weight, result.original_log_density - result.proposal_log_density
    )
    assert torch.isfinite(result.original_score).all()
    for row in result.cuts:
        edges = [0, *[int(k) + 1 for k in row.nonzero().flatten()], 6]
        assert sum(b - a for a, b in zip(edges[:-1], edges[1:])) == 6


def test_eta_zero_conditional_weights_are_constant_with_outside_cuts():
    rates = torch.tensor([0.02, 0.07, 0.11, 0.03, 0.15], dtype=torch.float64)
    for start, end in [(0, 3), (1, 3), (2, 3), (3, 6)]:
        result = sample(
            rates.log(),
            0.0,
            start,
            end,
            8.0,
            128,
            generator=torch.Generator().manual_seed(77),
        )
        truth = -8 * float(rates[start : end - 1].sum())
        for k in ([start - 1] if start else []) + ([end - 1] if end < 6 else []):
            truth += math.log(-math.expm1(-float(rates[k]) * 8))
        torch.testing.assert_close(
            result.log_weight,
            torch.full_like(result.log_weight, truth),
            atol=1e-12,
            rtol=1e-12,
        )


def test_complete_path_scores_against_independent_finite_differences():
    import numpy as np
    from path_reference import path_values

    rates = np.array([0.02, 0.07, 0.11, 0.03, 0.15])
    eta = 0.7
    result = sample(
        torch.tensor(np.log(rates), dtype=torch.float64),
        eta,
        1,
        4,
        8.0,
        16,
        generator=torch.Generator().manual_seed(23),
        record_paths=True,
    )
    for i in range(16):
        path = {
            k: getattr(result, k)[i].tolist()
            for k in [
                "cuts",
                "event_bonds",
                "event_times",
                "event_forced",
                "boundary_bonds",
                "boundary_times",
            ]
        }
        state = dict(rates=rates, eta=eta, time=8.0, target=[1, 4])
        lp, lq, sp, sq = path_values(state, path)
        np.testing.assert_allclose(
            [lp, lq],
            [result.original_log_density[i], result.proposal_log_density[i]],
            atol=1e-12,
        )
        np.testing.assert_allclose(sp, result.original_score[i], atol=1e-12)
        np.testing.assert_allclose(sq, result.proposal_score[i], atol=1e-12)
        h = 1e-5
        for j in range(6):
            d = np.zeros(6)
            d[j] = h
            up = path_values(
                dict(state, rates=rates * np.exp(d[:-1]), eta=eta + d[-1]), path
            )
            down = path_values(
                dict(state, rates=rates * np.exp(-d[:-1]), eta=eta - d[-1]), path
            )
            np.testing.assert_allclose(
                [(up[0] - down[0]) / (2 * h), (up[1] - down[1]) / (2 * h)],
                [sp[j], sq[j]],
                atol=2e-8,
                rtol=2e-6,
            )


def test_boundary_segment_scores_and_stable_limits():
    from boundary import bernoulli_ratio, boundary_segment

    h = torch.tensor([1e-8, 0.1, 100.0], dtype=torch.float64, requires_grad=True)
    r = torch.full_like(h, 3.0)
    dt = torch.full_like(h, 0.7)
    val, score = boundary_segment(h, r, dt)
    derivative = torch.autograd.grad(val.sum(), h)[0] * h
    torch.testing.assert_close(score, derivative, atol=1e-12, rtol=1e-6)
    x = torch.tensor([0.0, 1e-12, 1e3], dtype=torch.float64)
    torch.testing.assert_close(
        bernoulli_ratio(x), torch.tensor([1.0, 1.0 - 5e-13, 0.0], dtype=torch.float64)
    )
