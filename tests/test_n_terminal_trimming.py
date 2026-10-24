"""Independent recurrence, mass conservation and score identity for trimming."""

import numpy as np
import torch

from presto.models.n_terminal_trimming import (
    conditional_path_indices,
    grouped_logsumexp,
    log_trimming_paths,
    trimmed_log_amount,
)
from presto.models.steady_processing import log_steady_fragment_amount


def fixture():
    sequence = [0, 1, 0, 2, 1]
    rates = torch.tensor([0.3, 1.7, 0.8], dtype=torch.float64)
    queries, birth, counts, owner, starts, ends = [], [], [], [], [], []
    for end in range(1, len(sequence) + 1):
        for start in range(end):
            starts.append(len(birth))
            queries.append((start, end))
            for earlier in range(start + 1):
                x = torch.tensor(float(end - earlier - 1), dtype=torch.float64)
                left = torch.tensor(float(earlier > 0), dtype=torch.float64)
                right = torch.tensor(float(end < len(sequence)), dtype=torch.float64)
                birth.append(log_steady_fragment_amount(x, left, right, torch.ones_like(x)).item())
                counts.append(np.bincount(sequence[earlier:start], minlength=3))
                owner.append(len(queries) - 1)
            ends.append(len(birth))
    tensors = [
        torch.tensor(birth, dtype=torch.float64),
        torch.tensor(np.array(counts), dtype=torch.float64),
        torch.tensor(owner),
        torch.tensor(starts),
        torch.tensor(ends),
        torch.tensor([sequence[a] for a, b in queries]),
        torch.tensor([b - a == 1 for a, b in queries]),
    ]
    return sequence, rates, queries, tensors


def test_all_spans_recurrence_linear_system_and_residue_mass_balance():
    sequence, rates, queries, tensors = fixture()
    result = trimmed_log_amount(*tensors, rates.log(), torch.tensor(0.0)).exp().numpy()
    source = {}
    for a, b in queries:
        n = b - a - 1
        total = n + 1
        # Independent scalar birth formula, including true protein termini.
        if a == 0 and b == len(sequence):
            value = 1 / total
        elif a == 0 or b == len(sequence):
            value = 1 / (total * (total + 1))
        else:
            value = 2 / (total * (total + 1) * (total + 2))
        source[a, b] = value
    direct = {}
    for b in range(1, len(sequence) + 1):
        matrix = np.zeros((b, b))
        for a in range(b):
            trim = rates[sequence[a]].item() if b - a > 1 else 0
            matrix[a, a] = 1 + trim
            if a:
                matrix[a, a - 1] = -rates[sequence[a - 1]].item()
        solution = np.linalg.solve(matrix, [source[a, b] for a in range(b)])
        for a, value in enumerate(solution):
            direct[a, b] = value
    np.testing.assert_allclose(result, [direct[q] for q in queries], rtol=1e-13)
    input_mass = sum((b - a) * source[a, b] for a, b in queries)
    cleared_mass = sum((b - a) * direct[a, b] for a, b in queries)
    free_residue_flux = sum(
        rates[sequence[a]].item() * direct[a, b] for a, b in queries if b - a > 1
    )
    assert abs(input_mass - len(sequence)) < 1e-13
    assert abs(input_mass - cleared_mass - free_residue_flux) < 1e-13
    assert any(b - a == 1 and value > 0 for (a, b), value in zip(queries, result))


def test_exact_score_expectation_and_finite_difference():
    _, rates, queries, tensors = fixture()
    theta = rates.log().requires_grad_()
    scale = torch.tensor(-0.4, dtype=torch.float64, requires_grad=True)
    values = trimmed_log_amount(*tensors, theta, scale)
    exact = torch.autograd.grad(values.sum(), (theta, scale), retain_graph=True)
    birth, counts, owner, _, _, residue, monomer = tensors
    scores = log_trimming_paths(birth, counts, owner, residue, monomer, theta, scale)
    _, probabilities = grouped_logsumexp(scores, owner, len(queries))
    score_gradient = torch.autograd.grad((probabilities.detach() * scores).sum(), (theta, scale))
    for a, b in zip(exact, score_gradient):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    epsilon = 1e-6
    for index in range(len(theta)):
        direction = torch.zeros_like(theta)
        direction[index] = epsilon
        upper = trimmed_log_amount(*tensors, theta.detach() + direction, scale.detach()).sum()
        lower = trimmed_log_amount(*tensors, theta.detach() - direction, scale.detach()).sum()
        assert abs(((upper - lower) / (2 * epsilon)).item() - exact[0][index].item()) < 1e-8


def test_conditional_sampling_stays_in_query_and_keeps_exact_forward():
    _, rates, queries, tensors = fixture()
    theta = rates.log().requires_grad_()
    uniforms = torch.rand(
        (len(queries), 1000), generator=torch.Generator().manual_seed(42), dtype=torch.float64
    )
    exact = trimmed_log_amount(*tensors, theta, torch.tensor(0.0))
    sampled = trimmed_log_amount(*tensors, theta, torch.tensor(0.0), uniforms=uniforms)
    torch.testing.assert_close(exact, sampled, rtol=1e-14, atol=1e-14)
    sampled.sum().backward()
    assert torch.isfinite(theta.grad).all()
    # Zero-mass leading paths and exact zero quantiles cannot enter a previous query.
    probabilities = torch.tensor([0.0, 1.0, 0.0, 0.5, 0.5], dtype=torch.float64)
    indices = conditional_path_indices(
        probabilities,
        torch.tensor([0, 2]),
        torch.tensor([2, 5]),
        torch.tensor([[0.0, 0.99], [0.0, 0.99]], dtype=torch.float64),
    )
    assert indices.tolist() == [[1, 1], [3, 4]]
