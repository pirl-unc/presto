"""Independent NumPy triangular recurrence, generalized to the registered times."""
import numpy as np


def solve(rhs, rates, eta, pole, dt):
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
        x = (rhs[a, b] + dt * (right[b]*row[a] + left[a]*column[b])) / (pole + dt*g*(prefix[b]-prefix[a]))
        result[a, b] = x
        row[a] += g*x
        column[b] += g*x
    return result


def counts(logs, eta, time_hours, substeps, coefficients):
    coef = coefficients['48']
    size = len(logs) + 1
    y = np.zeros((size, size))
    y[0, -1] = 1
    rates = np.exp(logs)
    for _ in range(substeps):
        for a, p in zip(coef['alpha'], coef['theta']):
            y -= 2*(complex(*a)*solve(y, rates, eta, complex(*p), time_hours/substeps)).real
        y *= coef['alpha0']
    return y


def features(sequence, context):
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    x = np.zeros((len(sequence)-1, 40))
    x[:, 0] = 1
    x[:, -1] = context
    for i, (a,b) in enumerate(zip(sequence[:-1], sequence[1:])):
        if a != 'A':
            x[i, alphabet.index(a)] = 1
        if b != 'A':
            x[i, 19+alphabet.index(b)] = 1
    return x


def analytical_logs(logs, spans, time_hours):
    rates = np.exp(logs)
    out = []
    for a,b in spans:
        value = -time_hours*rates[a:b-1].sum()
        if a:
            value += np.log(-np.expm1(-time_hours*rates[a-1]))
        if b <= len(rates):
            value += np.log(-np.expm1(-time_hours*rates[b-1]))
        out.append(value)
    return np.array(out)
