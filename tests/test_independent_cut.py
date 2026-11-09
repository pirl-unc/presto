"""Physical conservation and exact/sampled gradient contracts."""

import itertools

import pytest
import torch

from presto.models.independent_cut import (
    log_fragment_expectation,
    ordered_bond_features,
    partition_contains,
)


def test_exact_spans_match_all_cut_states_and_conserve_residues():
    rates = torch.tensor([0.08, 0.18, 0.45, 0.9, 1.4], dtype=torch.float64)
    log_rates = rates.log().requires_grad_()
    cuts = torch.tensor(list(itertools.product((False, True), repeat=5)))
    starts, last = torch.triu_indices(6, 6)
    ends = last + 1
    presence = partition_contains(cuts, starts, ends).double()
    assert torch.all(presence @ (ends - starts).double() == 6)
    for time in (0.1, 0.7, 4.0, 20.0):
        h = log_rates.exp() * time
        state_logp = torch.where(cuts, torch.log(-torch.expm1(-h)), -h).sum(-1)
        enumerated = state_logp.exp() @ presence
        analytic = log_fragment_expectation(
            log_rates, starts, ends, torch.full((21,), time, dtype=torch.float64)
        ).exp()
        torch.testing.assert_close(analytic, enumerated, atol=1e-14, rtol=1e-13)
        torch.testing.assert_close(analytic @ (ends - starts).double(), rates.new_tensor(6.0))
        weighted = torch.linspace(0.1, 1.0, 21, dtype=torch.float64)
        a = torch.autograd.grad(analytic @ weighted, log_rates, retain_graph=True)[0]
        b = torch.autograd.grad(enumerated @ weighted, log_rates, retain_graph=True)[0]
        torch.testing.assert_close(a, b, atol=1e-14, rtol=1e-12)


def test_ordered_features_distinguish_left_and_right():
    a, b = ordered_bond_features("AC"), ordered_bond_features("CA")
    assert a.shape == b.shape == (1, 39)
    assert a[0, 20] == b[0, 1] == 1
    assert a[0, 1] == b[0, 20] == 0
    assert ordered_bond_features("AC", context=0.7)[0, -1] == 0.7


def test_invalid_times_are_not_interpreted_as_observed_zero_fragments():
    with pytest.raises(ValueError, match="positive-time"):
        log_fragment_expectation(
            torch.zeros(2), torch.tensor([0]), torch.tensor([1]), torch.tensor([0.0])
        )
