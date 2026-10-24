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
    torch.testing.assert_close(result.log_weight, math.log(horizon * rate) - rate * t)
    torch.testing.assert_close(result.original_score[:, 0], 1 - rate * t)
    torch.testing.assert_close(
        result.original_score[:, 1], (1 - rate * t) * math.log(2 / 20)
    )
    torch.testing.assert_close(
        result.proposal_score, torch.zeros_like(result.proposal_score)
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
