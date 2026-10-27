"""Numerical boundary checks on artificial data only."""

import numpy as np
import pytest
from scipy.optimize import minimize

from verify_controls import derivatives, newton


def fixture():
    generator = np.random.default_rng(17)
    matrix = np.column_stack([np.ones(70), generator.normal(size=(70, 4))])
    labels = np.tile([0.0, 1.0], 35)
    return matrix, labels


def test_gradient_and_hessian_match_finite_differences():
    matrix, labels = fixture()
    beta = np.asarray([0.3, -0.5, 0.2, 0.7, 0.8])
    _, gradient, hessian = derivatives(beta, matrix, labels)
    for i in range(len(beta)):
        offset = np.zeros_like(beta)
        offset[i] = 1e-5
        plus = derivatives(beta + offset, matrix, labels)
        minus = derivatives(beta - offset, matrix, labels)
        assert gradient[i] == pytest.approx((plus[0] - minus[0]) / 2e-5, abs=1e-8)
        np.testing.assert_allclose(hessian[:, i], (plus[1] - minus[1]) / 2e-5, atol=1e-8)


def test_newton_agrees_with_independent_optimizer():
    matrix, labels = fixture()
    beta, history = newton(matrix, labels)
    fitted = minimize(
        lambda b: derivatives(b, matrix, labels)[:2],
        np.zeros(matrix.shape[1]),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-10},
    )
    np.testing.assert_allclose(beta, fitted.x, atol=1e-7, rtol=0)
    assert history[-1]["max_gradient"] <= 1e-8
    assert all(b["objective"] <= a["objective"] for a, b in zip(history, history[1:]))


def test_intercept_unpenalized_and_failure_is_explicit():
    matrix, labels = fixture()
    intercept = np.array([1.0, 0, 0, 0, 0])
    assert derivatives(intercept, matrix, labels)[0] == pytest.approx(
        np.mean(np.logaddexp(0, matrix @ intercept) - labels * (matrix @ intercept))
    )
    with pytest.raises(AssertionError, match="did not converge"):
        newton(matrix, labels, max_steps=1)
