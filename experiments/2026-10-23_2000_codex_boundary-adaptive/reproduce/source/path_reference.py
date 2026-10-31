"""Independent NumPy/scalar complete-path density and score reconstruction."""

import math
import numpy as np


def bfun(x):
    if abs(x) < 1e-4:
        return 1 - x / 2 + x * x / 12 - x**4 / 720
    return x * math.exp(-x) / (-math.expm1(-x))


def path_values(state, path):
    rates = np.asarray(state["rates"], dtype=float)
    eta = state["eta"]
    horizon = state["time"]
    size = len(rates) + 1
    a, b = state["target"]
    boundaries = ([a - 1] if a else []) + ([b - 1] if b < size else [])
    forbidden = set(range(a, b - 1)) | set(boundaries)
    cuts = set()
    previous = 0.0
    logp = 0.0
    logq = 0.0
    sp = np.zeros(size)
    sq = np.zeros(size)
    events = [
        (k, t, f)
        for k, t, f in zip(
            path["event_bonds"], path["event_times"], path["event_forced"]
        )
        if k >= 0
    ]
    for bond, at, forced in events + [(-1, horizon, False)]:
        dt = at - previous
        assert dt >= 0
        edges = [0, *[k + 1 for k in sorted(cuts)], size]
        hazard = np.zeros(size - 1)
        length_log = np.zeros(size - 1)
        for left, right in zip(edges[:-1], edges[1:]):
            ll = math.log((right - left) / 20)
            hazard[left : right - 1] = rates[left : right - 1] * math.exp(eta * ll)
            length_log[left : right - 1] = ll
        outside = hazard.copy()
        outside[list(forbidden)] = 0
        logp -= dt * hazard.sum()
        logq -= dt * outside.sum()
        sp[:-1] -= dt * hazard
        sp[-1] -= dt * (hazard @ length_log)
        sq[:-1] -= dt * outside
        sq[-1] -= dt * (outside @ length_log)
        for k in boundaries:
            if k in cuts:
                continue
            h = hazard[k]
            r0 = horizon - previous
            r1 = horizon - at
            assert r1 > 0
            logq += (
                -h * dt
                + math.log(-math.expm1(-h * r1))
                - math.log(-math.expm1(-h * r0))
            )
            g = -h * dt + bfun(h * r1) - bfun(h * r0)
            sq[k] += g
            sq[-1] += g * length_log[k]
        if bond >= 0:
            assert bond not in cuts and bond not in range(a, b - 1)
            assert forced == (bond in boundaries)
            h = hazard[bond]
            ll = length_log[bond]
            logp += math.log(h)
            sp[bond] += 1
            sp[-1] += ll
            logq += math.log(h)
            g = 1.0
            if forced:
                x = h * (horizon - at)
                logq -= math.log(-math.expm1(-x))
                g -= bfun(x)
            sq[bond] += g
            sq[-1] += g * ll
            cuts.add(bond)
        previous = at
    edges = [0, *[k + 1 for k in sorted(cuts)], size]
    assert (a, b) in list(zip(edges[:-1], edges[1:]))
    assert [k in cuts for k in range(size - 1)] == path["cuts"]
    actual = [(k, t) for k, t, f in events if f]
    assert actual == list(zip(path["boundary_bonds"], path["boundary_times"]))
    return float(logp), float(logq), sp, sq
