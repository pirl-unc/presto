"""Verify all-pair and anchor objective/gradient identities before fitting."""

from itertools import combinations

import numpy as np


def verify(x, y):
    x, y = np.asarray(x), np.asarray(y)
    n, dimensions = x.shape
    assert n >= 2 and y.shape == (n,)
    z = np.column_stack((x, y))
    centered = z - z.mean(0)
    pairs = np.array([z[i] - z[j] for i, j in combinations(range(n), 2)])
    gram = pairs.T @ pairs / len(pairs)
    equivalent = 2 * n / (n - 1) * (centered.T @ centered / n)
    maximum = float(np.max(np.abs(gram - equivalent)))
    np.testing.assert_allclose(gram, equivalent, atol=2e-10, rtol=2e-10)
    beta = np.linspace(-0.1, 0.1, dimensions)
    coefficient = np.r_[beta, -1.0]
    pair_residual = pairs @ coefficient
    centered_residual = centered @ coefficient
    np.testing.assert_allclose(
        np.mean(pair_residual**2),
        2 * n / (n - 1) * np.mean(centered_residual**2),
        atol=2e-10,
        rtol=2e-10,
    )
    grad = 2 * pairs[:, :-1].T @ pair_residual / len(pairs)
    grad_equivalent = 4 / (n - 1) * centered[:, :-1].T @ centered_residual
    np.testing.assert_allclose(grad, grad_equivalent, atol=2e-10, rtol=2e-10)
    anchor_matrices = []
    for anchor in range(n):
        diffs = np.array([z[i] - z[anchor] for i in range(n) if i != anchor])
        actual = diffs.T @ diffs / (n - 1)
        offset = z.mean(0) - z[anchor]
        expected = n / (n - 1) * (centered.T @ centered / n + np.outer(offset, offset))
        maximum = max(maximum, float(np.max(np.abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(
            2 * diffs[:, :-1].T @ (diffs @ coefficient) / (n - 1),
            2 * expected[:-1] @ coefficient,
            atol=2e-10,
            rtol=2e-10,
        )
        anchor_matrices.append(actual)
    np.testing.assert_allclose(
        np.mean(anchor_matrices, axis=0), gram, atol=2e-10, rtol=2e-10
    )
    return dict(
        n_endpoints=n,
        n_pairs=len(pairs),
        n_reference_degrees=n - 1,
        maximum_gram_identity_error=maximum,
        maximum_gradient_identity_error=float(np.max(np.abs(grad - grad_equivalent))),
    )
