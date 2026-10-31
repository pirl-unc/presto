"""Complete-state proofs for exact-denominator fragment score gradients."""

import itertools

import torch

from presto.models.independent_cut import log_fragment_expectation, partition_contains


def test_full_score_and_integrated_presence_have_the_same_exact_expectation():
    logs = torch.tensor([-2.1, -1.3, -0.6, 0.2, 0.7], dtype=torch.float64, requires_grad=True)
    starts, last = torch.triu_indices(6, 6)
    ends = last + 1
    cuts = torch.tensor(list(itertools.product((False, True), repeat=5)))
    presence = partition_contains(cuts, starts, ends).double()
    coefficient = torch.linspace(-0.3, 0.7, 21, dtype=torch.float64)
    for time in (0.3, 1.0, 3.0):
        logmass = log_fragment_expectation(
            logs, starts, ends, torch.full((21,), time, dtype=torch.float64)
        )
        exact = torch.autograd.grad(coefficient @ logmass, logs, retain_graph=True)[0]
        h = logs.exp() * time
        cut_probability = -torch.expm1(-h)
        state_probability = torch.where(cuts, cut_probability, torch.exp(-h)).prod(1)
        cut_score = h * torch.exp(-h) / cut_probability
        full_score = torch.where(cuts, cut_score, -h)
        reward = presence @ (coefficient / logmass.exp())
        full_expectation = (state_probability[:, None] * reward[:, None] * full_score).sum(0)
        conditional = torch.zeros((21, 5), dtype=torch.float64)
        for row, (a, b) in enumerate(zip(starts.tolist(), ends.tolist())):
            conditional[row, a : b - 1] = -h[a : b - 1]
            if a:
                conditional[row, a - 1] = cut_score[a - 1]
            if b < 6:
                conditional[row, b - 1] = cut_score[b - 1]
        integrated = (presence * (coefficient / logmass.exp())) @ conditional
        integrated_expectation = (state_probability[:, None] * integrated).sum(0)
        torch.testing.assert_close(full_expectation, exact, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(integrated_expectation, exact, atol=1e-12, rtol=1e-12)
