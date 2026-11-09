"""Closed-form and independent generator checks for full-fragment branching."""

import pytest
import torch

from presto.models.fragment_state import (
    fragment_count_derivative,
    fragment_expectations,
    sample_fragment_paths,
)
from presto.models.independent_cut import log_fragment_expectation, partition_contains


def dense_generator(log_rates, eta):
    size = len(log_rates) + 1
    spans = [(a, b) for a in range(size) for b in range(a + 1, size + 1)]
    index = {span: i for i, span in enumerate(spans)}
    generator = log_rates.new_zeros((len(spans), len(spans)))
    for column, (a, b) in enumerate(spans):
        for k in range(a, b - 1):
            hazard = log_rates[k].exp() * ((b - a) / 20) ** eta
            generator[column, column] -= hazard
            generator[index[a, k + 1], column] += hazard
            generator[index[k + 1, b], column] += hazard
    return generator, spans, index


@pytest.mark.parametrize("eta_value", [-1.0, 0.0, 1.0])
def test_matrix_free_action_and_gradient_match_dense_fragment_generator(eta_value):
    logs = torch.tensor([0.02, 0.07, 0.11, 0.03, 0.15], dtype=torch.float64).log().requires_grad_()
    eta = torch.tensor(eta_value, dtype=torch.float64, requires_grad=True)
    generator, spans, index = dense_generator(logs, eta)
    initial = logs.new_zeros(len(spans))
    initial[index[0, 6]] = 1
    result = fragment_expectations(logs, eta, [0, 0.5, 2, 8])
    starts = torch.tensor([a for a, _ in spans])
    lasts = torch.tensor([b - 1 for _, b in spans])
    lengths = (lasts - starts + 1).double()
    actual = result.counts[:, starts, lasts]
    expected = torch.stack([torch.matrix_exp(generator * t) @ initial for t in [0, 0.5, 2, 8]])
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(actual @ lengths, logs.new_full((4,), 6), rtol=1e-12, atol=1e-11)
    assert (result.counts >= 0).all()
    assert result.expected_count_error_bound <= 1e-12
    weights = torch.sin(torch.arange(len(spans), dtype=torch.float64)) + 1.1
    computed = torch.autograd.grad((actual * weights).sum(), (logs, eta), retain_graph=True)
    reference = torch.autograd.grad((expected * weights).sum(), (logs, eta))
    for a, b in zip(computed, reference):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
    counts = logs.new_zeros((6, 6))
    vector = torch.arange(len(spans), dtype=torch.float64) / 30
    counts[starts, lasts] = vector
    action = fragment_count_derivative(counts, logs, eta)
    torch.testing.assert_close(action[starts, lasts], generator.detach() @ vector)


def test_zero_length_effect_recovers_every_independent_cut_product():
    logs = torch.tensor([0.02, 0.07, 0.11, 0.03, 0.15], dtype=torch.float64).log()
    starts, lasts = torch.triu_indices(6, 6)
    result = fragment_expectations(logs, logs.new_zeros(()), [0.5, 2, 8], checkpoint_steps=False)
    for i, time in enumerate([0.5, 2, 8]):
        expected = log_fragment_expectation(
            logs, starts, lasts + 1, logs.new_full(starts.shape, time)
        ).exp()
        torch.testing.assert_close(
            result.counts[i, starts, lasts], expected, atol=1e-12, rtol=1e-10
        )


def test_single_bond_path_score_includes_terminal_survival():
    logs = torch.tensor([0.2], dtype=torch.float64).log()
    eta = logs.new_tensor(0.4)
    time = 3.0
    samples = sample_fragment_paths(
        logs, eta, time, 4096, generator=torch.Generator().manual_seed(42)
    )
    hazard = (logs.exp() * (0.1**eta)).item()
    event = samples.cuts[:, 0]
    torch.testing.assert_close(
        samples.log_rate_score[~event, 0], logs.new_full((int((~event).sum()),), -hazard * time)
    )
    torch.testing.assert_close(
        samples.length_score, samples.log_rate_score[:, 0] * torch.log(logs.new_tensor(0.1))
    )
    torch.testing.assert_close(
        samples.log_rate_score[event, 0],
        1 + samples.log_density[event] - torch.log(logs.new_tensor(hazard)),
    )
    probability = 1 - torch.exp(logs.new_tensor(-hazard * time))
    assert abs(event.double().mean() - probability) < 6 * torch.sqrt(
        probability * (1 - probability) / 4096
    )
    starts, lasts = torch.triu_indices(2, 2)
    present = partition_contains(samples.cuts, starts, lasts + 1)
    assert (present.double() @ (lasts - starts + 1).double() == 2).all()


def test_time_zero_is_intact_and_invalid_domain_is_rejected():
    logs = torch.zeros(3, dtype=torch.float64)
    eta = logs.new_zeros(())
    result = fragment_expectations(logs, eta, [0])
    assert result.counts.sum() == result.counts[0, 0, -1] == 1
    sample = sample_fragment_paths(logs, eta, 0, 4, generator=torch.Generator().manual_seed(42))
    assert (
        not sample.cuts.any() and not sample.log_rate_score.any() and not sample.length_score.any()
    )
    with pytest.raises(ValueError, match="nondecreasing"):
        fragment_expectations(logs, eta, [2, 1])
    with pytest.raises(ValueError, match="Nonnegative"):
        fragment_expectations(logs, eta, [-1])
