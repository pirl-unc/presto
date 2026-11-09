"""Small exact inequality-constrained weighted least-squares curve fits."""

import itertools

import numpy as np
from scipy.optimize import minimize


def constraints(time, concave=False):
    slopes = np.diff(np.eye(len(time)), axis=0) / np.diff(time)[:, None]
    return np.vstack([slopes, -np.diff(slopes, axis=0)]) if concave else slopes


def fit_shape(y, weight, time, concave=False):
    a = constraints(time, concave)
    gram = (a / weight) @ a.T
    best = None
    for n in range(min(len(y) - 1, len(a)) + 1):
        for active in itertools.combinations(range(len(a)), n):
            index = list(active)
            multipliers = np.zeros(len(a))
            if n:
                sub = gram[np.ix_(index, index)]
                if np.linalg.matrix_rank(sub) != n:
                    continue
                multipliers[index] = np.linalg.solve(sub, -(a @ y)[index])
            z = y + (a.T @ multipliers) / weight
            if np.min(a @ z) < -1e-9 or np.min(multipliers) < -1e-9:
                continue
            value = np.sum(weight * (z - y) ** 2)
            if best is None or value < best[0]:
                best = (value, z, active, multipliers)
    assert best is not None
    value, z, active, multipliers = best
    stationarity = weight * (z - y) - a.T @ multipliers
    complementarity = multipliers * (a @ z)
    residual = max(np.max(np.abs(stationarity)), np.max(np.abs(complementarity)),
                   max(0., -np.min(a @ z)), max(0., -np.min(multipliers)))
    assert residual <= 1e-9
    return z, {"active": list(active), "multipliers": multipliers.tolist(),
               "kkt_max_residual": float(residual), "weighted_error": float(value)}


def pava(y, weight):
    blocks = []
    for i, (value, w) in enumerate(zip(y, weight)):
        blocks.append([i, i + 1, w, w * value])
        while len(blocks) >= 2 and blocks[-2][3] / blocks[-2][2] > blocks[-1][3] / blocks[-1][2]:
            right = blocks.pop()
            left = blocks.pop()
            blocks.append([left[0], right[1], left[2] + right[2], left[3] + right[3]])
    result = np.zeros(len(y))
    for start, end, w, total in blocks:
        result[start:end] = total / w
    return result


def independent_slsqp(y, weight, time):
    a = constraints(time, True)
    result = minimize(lambda z: .5 * np.sum(weight * (z - y) ** 2), y,
                      jac=lambda z: weight * (z - y), method="SLSQP",
                      constraints={"type": "ineq", "fun": lambda z: a @ z, "jac": lambda z: a},
                      options={"ftol": 1e-14, "maxiter": 1000})
    assert result.success, result.message
    return result.x


def qualify():
    checks = []
    for y in (np.zeros(5), np.arange(5.), -np.arange(5.), np.array([0., 2., 4., 2., 0.])):
        time = np.array([0., 1., 2., 4., 7.])
        weight = np.array([1., 2., 3., 2., 1.])
        mono, _ = fit_shape(y, weight, time)
        concave, _ = fit_shape(y, weight, time, True)
        errors = [float(np.max(np.abs(mono - pava(y, weight)))),
                  float(np.max(np.abs(concave - independent_slsqp(y, weight, time))))]
        assert max(errors) <= 1e-7
        checks.append(errors)
    return checks
