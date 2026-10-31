"""Numerical fixtures, not synthetic biological training data."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from model import (
    Observer,
    center_numpy,
    center_torch,
    numpy_observer,
    parameter_penalty,
)


def test_centering_and_nonlinear_gradient():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(11, 5))
    target = rng.normal(size=11)
    index = np.array([0] * 4 + [1] * 7)
    counts = np.array([4.0, 7.0])
    weights = 1 / (2 * counts[index])
    network = Observer(5)
    xt = torch.tensor(x)

    def loss():
        prediction = center_torch(network(xt), torch.tensor(index), torch.tensor(counts))
        return (
            torch.tensor(weights) * (prediction - torch.tensor(target)).square()
        ).sum() + 0.001 * parameter_penalty(network)

    loss().backward()
    state = {k: v.detach().numpy().copy() for k, v in network.state_dict().items()}
    expected = center_numpy(numpy_observer(x, state), index, counts)
    actual = center_torch(network(xt), torch.tensor(index), torch.tensor(counts)).detach().numpy()
    np.testing.assert_allclose(expected, actual, rtol=1e-12, atol=1e-12)
    for parameter in network.parameters():
        grad = parameter.grad.reshape(-1)[0].item()
        with torch.no_grad():
            coordinate = parameter.reshape(-1)
            value = coordinate[0].item()
            coordinate[0] = value + 1e-6
            plus = loss().item()
            coordinate[0] = value - 1e-6
            minus = loss().item()
            coordinate[0] = value
        np.testing.assert_allclose(grad, (plus - minus) / 2e-6, rtol=1e-6, atol=1e-8)


def test_ridge_normal_equations_and_group_shift_invariance():
    rng = np.random.default_rng(43)
    x = rng.normal(size=(12, 4))
    index = np.array([0] * 4 + [1] * 8)
    counts = np.array([4.0, 8.0])
    w = 1 / (2 * counts[index])
    y = center_numpy(rng.normal(size=12), index, counts)
    centered = center_numpy(x, index, counts)
    alpha = 0.1
    beta = np.linalg.solve(
        centered.T @ (w[:, None] * centered) + alpha / 4 * np.eye(4), centered.T @ (w * y)
    )
    residual = 2 * centered.T @ (w * (centered @ beta - y)) + 2 * alpha / 4 * beta
    np.testing.assert_allclose(residual, 0, atol=1e-12)
    shifted = x + np.array([4.0, -9.0])[index, None]
    np.testing.assert_allclose(
        center_numpy(shifted, index, counts) @ beta, centered @ beta, atol=1e-12
    )
