"""Independent NumPy recurrence retained from the qualified solver verifier."""

import numpy as np


def numpy_solve(rhs, rates, eta, pole, dt):
    size = len(rates) + 1
    prefix = np.r_[0, np.cumsum(rates)]
    left, right = np.r_[0, rates], np.r_[rates, 0]
    row = np.zeros(size, dtype=complex)
    column = row.copy()
    result = np.zeros((size, size), dtype=complex)
    for length in range(size, 0, -1):
        a = np.arange(size - length + 1)
        b = a + length - 1
        g = (length / 20) ** eta
        numerator = rhs[a, b] + dt * (right[b] * row[a] + left[a] * column[b])
        x = numerator / (pole + dt * g * (prefix[b] - prefix[a]))
        result[a, b] = x
        row[a] += g * x
        column[b] += g * x
    return result


def numpy_counts(logs, eta, order, substeps, coefficients):
    coef = coefficients[str(order)]
    size = len(logs) + 1
    y = np.zeros((size, size))
    y[0, -1] = 1
    for _ in range(substeps):
        for a, p in zip(coef["alpha"], coef["theta"]):
            y -= (
                2
                * (
                    complex(*a)
                    * numpy_solve(y, np.exp(logs), eta, complex(*p), 24 / substeps)
                ).real
            )
        y *= coef["alpha0"]
    return y
