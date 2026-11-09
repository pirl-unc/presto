"""Experiment-only direct-rate closure of the original temporal-ratio model."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse import csr_array

from source import FAMILY, read_groups

LN10 = math.log(10)


def log_exprel_negative(x):
    """log((1-exp(-x))/x) and derivative, including the continuous x=0 limit."""
    x = np.asarray(x, dtype=float)
    assert np.isfinite(x).all() and (x >= 0).all()
    small = x < 0.001
    z = x[small]
    value, derivative = np.empty_like(x), np.empty_like(x)
    value[small] = -z / 2 + z**2 / 24 - z**4 / 2880 + z**6 / 181440
    derivative[small] = -0.5 + z / 12 - z**3 / 720 + z**5 / 30240
    z = x[~small]
    value[~small] = np.log(-np.expm1(-z)) - np.log(z)
    derivative[~small] = np.exp(-z) / (-np.expm1(-z)) - 1 / z
    return value, derivative


def boundary_ratio(rates, t, ref):
    v, d = log_exprel_negative(rates * t)
    vr, dr = log_exprel_negative(rates * ref)
    return np.log(t / ref) + v - vr, t * d - ref * dr


class Block:
    def __init__(self, group, allocation, initial):
        self.g = group
        self.members = allocation["members"]
        self.initial = np.asarray(initial, dtype=float)
        self.n = len(self.initial)
        assert self.n == len(self.members)
        self.bond_index = np.full(len(group["sequence"]) - 1, -1, dtype=int)
        for column, bonds in enumerate(self.members):
            self.bond_index[bonds] = column
        self.boundaries = []
        rr, cc, vv = [], [], []
        for row, (a, b) in enumerate(zip(group["a"], group["b"])):
            internal = self.bond_index[a : b - 1]
            assert (internal >= 0).all()
            indices, counts = np.unique(internal, return_counts=True)
            rr.extend([row] * len(indices))
            cc.extend(indices)
            vv.extend(counts)
        shape = (len(group["y"]), self.n)
        self.internal = csr_array((np.array(vv, dtype=float), (rr, cc)), shape=shape)
        for index, valid in ((group["a"] - 1, group["a"] > 0),
                             (group["b"] - 1, group["b"] < len(group["sequence"]))):
            rows = np.flatnonzero(valid)
            columns = self.bond_index[index[valid]]
            assert (columns >= 0).all()
            self.boundaries.append((rows, columns))
        self.sqrtw = np.sqrt(group["w"])
        self.internal_jac = self.internal.multiply(
            (-(group["t"] - group["ref"]) / LN10)[:, None]
        ).tocsr()

    def prediction_and_jacobian(self, rates):
        g = self.g
        assert np.isfinite(rates).all() and (rates >= 0).all()
        p = -(g["t"] - g["ref"]) * (self.internal @ rates)
        jac = self.internal_jac.copy()
        for rows, columns in self.boundaries:
            value, derivative = boundary_ratio(rates[columns], g["t"][rows], g["ref"][rows])
            p[rows] += value
            jac += csr_array((derivative / LN10, (rows, columns)), shape=jac.shape)
        return p / LN10, jac

    def residual_and_jacobian(self, rates):
        p, jac = self.prediction_and_jacobian(rates)
        return self.sqrtw * (p - self.g["y"]), jac.multiply(self.sqrtw[:, None]).tocsr()

    def reference_log_mass(self, rates):
        result = -self.g["ref"] * (self.internal @ rates)
        for rows, columns in self.boundaries:
            with np.errstate(divide="ignore"):
                result[rows] += np.log(-np.expm1(-rates[columns] * self.g["ref"][rows]))
        return result


def load_blocks():
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    old = Path(cfg["previous_family"])
    for name, expected in cfg["previous_files"].items():
        assert hashlib.sha256((old / name).read_bytes()).hexdigest() == expected
    groups = read_groups(cfg)
    assert len(groups) == 7 and sum(len(g["y"]) for g in groups) == 67576
    assert {g["split"] for g in groups} == {"train"}
    allocations = json.loads((old / "results/rate-allocation.json").read_text())
    saved = json.loads((old / "results/capacity-fits.json").read_text())
    theta = np.asarray(saved["free_condition_rates"]["coefficients"])
    blocks, offset = [], 0
    for g, allocation in zip(groups, allocations):
        assert g["condition"] == allocation["condition"]
        n = len(allocation["members"])
        block = Block(g, allocation, np.exp(theta[offset:offset + n]))
        block.offset = offset
        blocks.append(block)
        offset += n
    assert offset == len(theta) == 2121
    return cfg, blocks, saved


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
