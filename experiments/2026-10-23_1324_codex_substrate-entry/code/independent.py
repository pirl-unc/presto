"""Independent source reader and NumPy temporal forwards; prior test excluded."""

import gzip
import hashlib
import json
import math
import resource
import time
from functools import lru_cache
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from scipy.special import logsumexp, roots_laguerre, roots_legendre

FAMILY = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def read_groups(cfg):
    path = Path(cfg["recipe_directory"])
    assert (
        hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
        == cfg["recipe_manifest_sha256"]
    )
    source = json.loads((path / "examples.json").read_text())
    grouped = defaultdict(list)
    for row in source:
        if row["split"] == "test":
            continue
        grouped[row["condition_id"]].append(row)
    groups = []
    for condition, curves in sorted(grouped.items()):
        first = curves[0]
        sequence = first["parent_sequence"]
        ctx = first["context"]
        columns = [np.ones(len(sequence) - 1)]
        for side in (sequence[:-1], sequence[1:]):
            for aa in AA[1:]:
                columns.append(np.array([float(residue == aa) for residue in side]))
        context = math.log(
            ctx["substrate_concentration_um"]
            / ctx["enzyme_substrate_denominator"]
            / 0.2
        )
        columns.append(np.full(len(sequence) - 1, context))
        x = np.stack(columns, axis=1)
        rows = []
        for curve in curves:
            values = {v["time_hours"]: v["target_value"] for v in curve["observations"]}
            times = curve["common_positive_times"]
            ref = times[0]
            for t in times[1:]:
                rows.append((curve, t, ref, values[t], values[ref]))
        groups.append(
            dict(
                condition=condition,
                parent=first["construct_id"],
                split=first["split"],
                sequence=sequence,
                x=x,
                rows=rows,
                ncurves=len(curves),
                a=np.array([r[0]["start_0based"] for r in rows]),
                b=np.array([r[0]["end_0based"] for r in rows]),
                t=np.array([r[1] for r in rows]),
                ref=np.array([r[2] for r in rows]),
                y=np.array([math.log10(r[3]) - math.log10(r[4]) for r in rows]),
            )
        )
    for split in ("train", "val"):
        subset = [g for g in groups if g["split"] == split]
        parents = Counter(g["parent"] for g in subset)
        for g in subset:
            g["w"] = np.array(
                [
                    1
                    / (
                        len(parents)
                        * parents[g["parent"]]
                        * g["ncurves"]
                        * (len(r[0]["common_positive_times"]) - 1)
                    )
                    for r in g["rows"]
                ]
            )
    return groups


def feature(g, variant):
    return (
        g["x"][:, :1]
        if variant == "constant"
        else g["x"][:, :39]
        if variant == "sequence"
        else g["x"]
    )


def forward(theta, g, x, gradient=False):
    rates = np.exp(x @ theta)
    a, b, t, ref = g["a"], g["b"], g["t"], g["ref"]
    n = len(rates)
    # Independent vector gather for physical termini and internal survival.
    prefix = np.r_[0.0, np.cumsum(rates)]
    internal = prefix[b - 1] - prefix[a]
    prediction = -(t - ref) * internal
    boundary_parts = []
    for index, mask in ((a - 1, a > 0), (b - 1, b <= n)):
        safe = np.clip(index, 0, n - 1)
        ht = rates[safe] * t
        hr = rates[safe] * ref
        prediction += np.where(
            mask, np.log(-np.expm1(-ht)) - np.log(-np.expm1(-hr)), 0.0
        )
        derivative = np.where(
            mask,
            ht * np.exp(-ht) / (-np.expm1(-ht)) - hr * np.exp(-hr) / (-np.expm1(-hr)),
            0.0,
        )
        boundary_parts.append((safe, derivative))
    prediction /= math.log(10)
    if not gradient:
        return prediction
    coefficient = 2 * g["w"] * (prediction - g["y"]) / math.log(10)
    bond = np.zeros(n)
    for index, derivative in boundary_parts:
        bond += np.bincount(index, weights=coefficient * derivative, minlength=n)
    difference = np.bincount(
        a, weights=-coefficient * (t - ref), minlength=n + 1
    ) - np.bincount(b - 1, weights=-coefficient * (t - ref), minlength=n + 1)
    bond += np.cumsum(difference)[:-1] * rates
    return prediction, x.T @ bond


@lru_cache(maxsize=4)
def rules(order):
    node, weight = roots_legendre(order)
    lag_node, lag_weight = roots_laguerre(order)
    return (node + 1) / 2, weight / weight.sum(), lag_node, lag_weight


def entry_points(theta, x, starts, ends, times, order=128):
    """Independent quadrature and continuous-process score derivatives."""
    rates = np.exp(x @ theta[:-1])
    lam = np.exp(theta[-1])
    n = len(rates)
    prefix = np.r_[0.0, np.cumsum(rates)]
    node, weight, lag_node, lag_weight = rules(order)
    chunks = []
    for offset in range(0, len(starts), 4096):
        a = starts[offset : offset + 4096]
        b = ends[offset : offset + 4096]
        t = times[offset : offset + 4096]
        internal = prefix[b - 1] - prefix[a]
        age = t[:, None] * node
        hazard = lam * t
        weights = np.log(weight)[None, :] - hazard[:, None] * (1 - node)
        fast = (hazard >= 64) & (lam > 2 * rates.sum())
        if np.any(fast):
            valid = lag_node[None, :] < hazard[:, None]
            lag_age = np.where(valid, t[:, None] - lag_node / lam, t[:, None])
            with np.errstate(divide="ignore"):
                lag_w = np.where(valid, np.log(lag_weight)[None, :], -np.inf)
            age = np.where(fast[:, None], lag_age, age)
            weights = np.where(fast[:, None], lag_w, weights)
        weights = (
            weights
            - logsumexp(weights, axis=1)[:, None]
            + np.log(-np.expm1(-hazard))[:, None]
        )
        logm = -age * internal[:, None]
        boundary = []
        for index, mask in ((a - 1, a > 0), (b - 1, b <= n)):
            h = rates[np.clip(index, 0, n - 1)][:, None] * age
            logm += np.where(mask[:, None], np.log(-np.expm1(-h)), 0.0)
            boundary.append(
                np.where(mask[:, None], h * np.exp(-h) / (-np.expm1(-h)), 0.0)
            )
        entered = logsumexp(weights + logm, axis=1)
        total = np.logaddexp(
            entered, np.where((a == 0) & (b == n + 1), -hazard, -np.inf)
        )
        posterior = np.exp(weights + logm - total[:, None])
        # Censored unentered paths contribute only for the complete parent.
        entry_score = np.sum(posterior * (1 - lam * (t[:, None] - age)), axis=1)
        entry_score += np.where(
            (a == 0) & (b == n + 1), -hazard * np.exp(-hazard - total), 0.0
        )
        chunks.append(
            np.column_stack(
                (
                    total,
                    np.sum(posterior * age, axis=1),
                    np.sum(posterior * boundary[0], axis=1),
                    np.sum(posterior * boundary[1], axis=1),
                    entry_score,
                )
            )
        )
    return np.concatenate(chunks)


def entry_forward(theta, g, x, gradient=False, order=128):
    keys = sorted(set(zip(g["a"], g["b"], g["t"])) | set(zip(g["a"], g["b"], g["ref"])))
    lookup = {key: i for i, key in enumerate(keys)}
    starts = np.array([a for a, b, t in keys])
    ends = np.array([b for a, b, t in keys])
    times = np.array([t for a, b, t in keys])
    target = np.array([lookup[k] for k in zip(g["a"], g["b"], g["t"])])
    reference = np.array([lookup[k] for k in zip(g["a"], g["b"], g["ref"])])
    out = entry_points(theta, x, starts, ends, times, order)
    prediction = (out[target, 0] - out[reference, 0]) / math.log(10)
    if not gradient:
        return prediction
    coefficient = 2 * g["w"] * (prediction - g["y"]) / math.log(10)
    combined = np.bincount(
        target, weights=coefficient, minlength=len(keys)
    ) - np.bincount(reference, weights=coefficient, minlength=len(keys))
    n = len(x)
    rates = np.exp(x @ theta[:-1])
    diff = np.bincount(
        starts, weights=-combined * out[:, 1], minlength=n + 1
    ) - np.bincount(ends - 1, weights=-combined * out[:, 1], minlength=n + 1)
    bond = np.cumsum(diff)[:-1] * rates
    for index, mask, column in ((starts - 1, starts > 0, 2), (ends - 1, ends <= n, 3)):
        bond += np.bincount(
            np.clip(index, 0, n - 1),
            weights=combined * out[:, column] * mask,
            minlength=n,
        )
    return prediction, np.r_[x.T @ bond, combined @ out[:, 4]]
