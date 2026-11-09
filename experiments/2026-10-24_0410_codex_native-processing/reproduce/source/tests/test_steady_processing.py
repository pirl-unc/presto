import itertools

import pytest
import torch

from presto.models.independent_cut import partition_contains
from presto.models.steady_processing import (
    conditional_fragment_age,
    fragment_event_log_score,
    log_steady_fragment_amount,
)


def _small_chain(log_rates):
    rates = log_rates.exp()
    spans = [(a, b) for a in range(4) for b in range(a + 1, 5)]
    starts = torch.tensor([a for a, _ in spans])
    ends = torch.tensor([b for _, b in spans])
    states = torch.tensor(list(itertools.product((False, True), repeat=3)))
    generator = rates.new_zeros((8, 8))
    indices = {tuple(row.tolist()): i for i, row in enumerate(states)}
    for i, state in enumerate(states):
        for bond in range(3):
            if not state[bond]:
                after = state.clone()
                after[bond] = True
                generator[i, indices[tuple(after.tolist())]] = rates[bond]
                generator[i, i] = generator[i, i] - rates[bond]
    zero = rates.new_zeros(())
    internal = torch.stack([rates[a : b - 1].sum() for a, b in spans])
    left = torch.stack([rates[a - 1] if a else zero for a, _ in spans])
    right = torch.stack([rates[b - 1] if b < 4 else zero for _, b in spans])
    return internal, left, right, starts, ends, states, generator


@pytest.mark.parametrize("removal", [0.1, 1.0, 10.0])
def test_steady_matches_complete_partition_generator_and_residue_mass(removal):
    theta = torch.tensor([0.2, 0.7, 1.4], dtype=torch.float64).log().requires_grad_()
    internal, left, right, starts, ends, states, generator = _small_chain(theta)
    delta = theta.new_tensor(removal)
    initial = theta.new_zeros(8)
    initial[0] = 1.0
    occupancy = torch.linalg.solve((delta * torch.eye(8, dtype=theta.dtype) - generator).T, initial)
    expected = occupancy @ partition_contains(states, starts, ends).double()
    actual = log_steady_fragment_amount(internal, left, right, delta).exp()
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-11)
    torch.testing.assert_close(actual @ (ends - starts).double(), theta.new_tensor(4 / removal))
    coefficients = torch.linspace(-0.5, 1.5, len(actual), dtype=torch.float64)
    actual_gradient = torch.autograd.grad(actual.log() @ coefficients, theta, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.log() @ coefficients, theta)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-11, rtol=1e-10)


def test_conditional_ages_and_scores_match_independent_resolvent():
    theta = torch.tensor([0.2, 0.7, 1.4], dtype=torch.float64).log().requires_grad_()
    internal, left, right, starts, ends, states, generator = _small_chain(theta)
    delta = theta.new_tensor(0.6, requires_grad=True)
    log_amount = log_steady_fragment_amount(internal, left, right, delta)
    uniforms = torch.rand(
        (10, 32768, 4), dtype=torch.float64, generator=torch.Generator().manual_seed(19)
    )
    age = conditional_fragment_age(
        internal[:, None], left[:, None], right[:, None], delta, uniforms
    )
    assert not age.requires_grad
    initial = theta.new_zeros(8)
    initial[0] = 1
    presence = partition_contains(states, starts, ends).double()
    for shift in (0.3, 2.0):
        shifted = (
            torch.linalg.solve(
                ((delta + shift) * torch.eye(8, dtype=theta.dtype) - generator).T, initial
            )
            @ presence
        )
        expected_transform = shifted / log_amount.exp()
        transformed = (-shift * age).exp()
        standard_error = transformed.std(dim=1) / (transformed.shape[1] ** 0.5)
        assert torch.max(abs(transformed.mean(dim=1) - expected_transform) / standard_error) < 6
    score = fragment_event_log_score(internal[:, None], left[:, None], right[:, None], delta, age)
    coefficients = torch.linspace(-0.5, 1.5, 10, dtype=torch.float64)
    got = torch.autograd.grad(score.mean(dim=1) @ coefficients, theta, retain_graph=True)[0]
    expected = torch.autograd.grad(log_amount @ coefficients, theta)[0]
    torch.testing.assert_close(got, expected, atol=0.025, rtol=0.02)


def test_conditional_validation_and_finite_difference():
    theta = torch.tensor([-0.7, 0.2, 0.6, -0.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda p: log_steady_fragment_amount(*p.exp()), (theta,))
    zero = theta.new_zeros(())
    one = theta.new_ones(())
    with pytest.raises(ValueError, match="positive removal"):
        log_steady_fragment_amount(one, one, one, zero)
    with pytest.raises(ValueError, match="Uniforms"):
        conditional_fragment_age(one, zero, one, one, torch.ones(4))
    with pytest.raises(ValueError, match="positive ages"):
        fragment_event_log_score(one, zero, one, one, zero)
