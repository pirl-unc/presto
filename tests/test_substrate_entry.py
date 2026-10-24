import torch

from presto.models.independent_cut import partition_contains
from presto.models.substrate_entry import (
    log_entry_fragment_expectation,
    sample_entry_partitions,
)


def test_entry_matches_independent_complete_state_generator():
    rates = torch.tensor([0.2, 0.7], dtype=torch.float64)
    entry = torch.tensor(0.4, dtype=torch.float64)
    # States: unentered, active uncut, first cut, second cut, both cut.
    q = torch.tensor(
        [
            [-0.4, 0.4, 0, 0, 0],
            [0, -0.9, 0.2, 0.7, 0],
            [0, 0, -0.7, 0, 0.7],
            [0, 0, 0, -0.2, 0.2],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float64,
    )
    spans = [(a, b) for a in range(3) for b in range(a + 1, 4)]
    starts = torch.tensor([a for a, b in spans])
    ends = torch.tensor([b for a, b in spans])
    states = torch.tensor(
        [[False, False], [False, False], [True, False], [False, True], [True, True]]
    )
    presence = partition_contains(states, starts, ends).double()
    for t in (0.1, 1.0, 8.0):
        expected = torch.matrix_exp(q * t)[0] @ presence
        actual = log_entry_fragment_expectation(
            rates.log(),
            entry.log(),
            starts,
            ends,
            torch.full((6,), t, dtype=torch.float64),
            order=64,
        ).exp()
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
        torch.testing.assert_close(
            actual @ (ends - starts).double(), torch.tensor(3.0, dtype=torch.float64)
        )


def test_entry_gradient_with_equal_entry_and_internal_rate():
    theta = torch.tensor([-0.7, -0.7, -0.7], dtype=torch.float64, requires_grad=True)
    starts = torch.tensor([0, 1, 0])
    ends = torch.tensor([2, 3, 3])
    times = torch.tensor([0.2, 2.0, 4.0], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda p: log_entry_fragment_expectation(p[:2], p[2], starts, ends, times, order=64),
        (theta,),
    )


def test_fast_entry_and_quadrature_switch_match_complete_states():
    starts = torch.tensor([0, 0, 0, 1, 1, 2])
    ends = torch.tensor([1, 2, 3, 2, 3, 3])
    cuts = torch.tensor(
        [[False, False], [False, False], [True, False], [False, True], [True, True]]
    )
    presence = partition_contains(cuts, starts, ends).double()
    times = torch.ones(6, dtype=torch.float64)
    for rate in (63.999, 64.001, 1000.0, 1.0e6):
        theta = torch.tensor([0.2, 0.7, rate], dtype=torch.float64).log().requires_grad_()
        a, b, entry = theta.exp()
        zero = theta.new_zeros(())
        generator = torch.stack(
            [
                torch.stack((-entry, entry, zero, zero, zero)),
                torch.stack((zero, -a - b, a, b, zero)),
                torch.stack((zero, zero, -b, zero, b)),
                torch.stack((zero, zero, zero, -a, a)),
                torch.zeros(5, dtype=torch.float64),
            ]
        )
        expected = torch.matrix_exp(generator)[0] @ presence
        actual = log_entry_fragment_expectation(
            theta[:2], theta[2], starts, ends, times, order=64
        ).exp()
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-9)
        coefficients = torch.arange(1, 7, dtype=torch.float64)
        got_gradient = torch.autograd.grad(actual @ coefficients, theta, retain_graph=True)[0]
        ref_gradient = torch.autograd.grad(expected @ coefficients, theta)[0]
        torch.testing.assert_close(got_gradient, ref_gradient, atol=2e-9, rtol=2e-7)
        torch.testing.assert_close(
            actual @ (ends - starts).double(),
            torch.tensor(3.0, dtype=torch.float64),
            atol=1e-12,
            rtol=0,
        )


def test_unentered_paths_keep_parent_and_have_finite_scores():
    theta = torch.tensor([-1.0, -2.0, -30.0], dtype=torch.float64, requires_grad=True)
    draw = sample_entry_partitions(
        theta[:2], theta[2], 1.0, 64, generator=torch.Generator().manual_seed(17)
    )
    assert not draw.entered.any() and not draw.cuts.any()
    assert partition_contains(draw.cuts, torch.tensor([0]), torch.tensor([3])).all()
    gradient = torch.autograd.grad(draw.log_prob.mean(), theta)[0]
    torch.testing.assert_close(
        gradient, torch.stack((theta.new_zeros(()), theta.new_zeros(()), -theta[2].exp()))
    )
