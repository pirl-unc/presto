"""Independent NumPy forward/analytic-gradient/Adam replay and outcome checks."""

import gzip
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

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
    for split in ("train", "val", "test"):
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


def main():
    started = time.monotonic()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    groups = read_groups(cfg)
    fits = json.loads((FAMILY / "results/fits.json").read_text())
    scores = json.loads((FAMILY / "results/metrics.json").read_text())
    replay = []
    for variant in cfg["variants"]:
        record = fits[variant]
        basis = np.array(record["feature_audit"]["basis"])
        original = np.zeros(basis.shape[0])
        original[0] = math.log(cfg["initial_rate_per_hour"])
        theta = basis.T @ original
        m = np.zeros_like(theta)
        v = m.copy()
        maximum = 0.0
        moment_checks = []
        train = [
            (g, feature(g, variant) @ basis) for g in groups if g["split"] == "train"
        ]
        for step in range(1, cfg["epochs"] + 1):
            grad = np.zeros_like(theta)
            for g, x in train:
                grad += forward(theta, g, x, True)[1]
            raw = basis @ theta
            reg = cfg["l2_nonintercept"] * raw
            reg[0] = 0
            grad += basis.T @ reg
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            theta -= (
                cfg["learning_rate"]
                * (m / (1 - 0.9**step))
                / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
            )
            if str(step) in record["checkpoints"]:
                error = float(
                    np.max(np.abs(theta - np.array(record["checkpoints"][str(step)])))
                )
                maximum = max(maximum, error)
                moment_checks.append(dict(step=step, max_coefficient_error=error))
        assert maximum < 1e-9, (variant, maximum)
        assert np.max(np.abs(theta - np.array(record["basis_coefficients"]))) < 1e-9
        replay.append(
            dict(
                variant=variant, updates=cfg["epochs"], checkpoint_errors=moment_checks
            )
        )
        print("Independent Adam replay", variant, maximum, flush=True)
    expected = {}
    for variant in list(fits) + ["no_change"]:
        for g in groups:
            if variant == "no_change":
                p = np.zeros(len(g["y"]))
            elif variant == "shared_time":
                z = np.log(g["t"] / g["ref"])
                beta = np.array(fits[variant]["coefficients"])
                p = beta[0] * z + beta[1] * z * z
            else:
                p = forward(
                    np.array(fits[variant]["coefficients"]), g, feature(g, variant)
                )
            for row, pred, w in zip(g["rows"], p, g["w"]):
                curve, t, ref, signal, reference = row
                expected[variant, curve["curve_id"], t] = (
                    float(pred),
                    float(w),
                    curve,
                    t,
                    ref,
                    signal,
                    reference,
                )
    # Shared-time solution recomputed using normal equations, not lstsq.
    xtwx = np.zeros((2, 2))
    xtwy = np.zeros(2)
    for g in groups:
        if g["split"] == "train":
            z = np.log(g["t"] / g["ref"])
            x = np.column_stack((z, z * z))
            xtwx += x.T @ (g["w"][:, None] * x)
            xtwy += x.T @ (g["w"] * g["y"])
    assert (
        np.max(
            np.abs(np.linalg.solve(xtwx, xtwy) - fits["shared_time"]["coefficients"])
        )
        < 1e-10
    )
    seen = set()
    rows = defaultdict(list)
    maximum = 0.0
    with gzip.open(FAMILY / "results/predictions.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            key = r["variant"], r["curve_id"], r["time"]
            assert key not in seen
            seen.add(key)
            pred, w, curve, t, ref, signal, reference = expected[key]
            y = math.log10(signal) - math.log10(reference)
            for name, value in [
                ("weight", w),
                ("time", t),
                ("reference_time", ref),
                ("signal", signal),
                ("reference_signal", reference),
            ]:
                assert r[name] == value
            for name in (
                "construct_id",
                "condition_id",
                "split",
                "peptide_sequence",
                "biological_replicate_raw",
            ):
                assert r[name] == curve[name]
            assert (
                r["start"] == curve["start_0based"] and r["end"] == curve["end_0based"]
            )
            maximum = max(maximum, abs(pred - r["prediction"]))
            assert abs(y - r["target"]) < 1e-12
            assert (
                abs(math.log10(r["predicted_signal"]) - r["predicted_log10_signal"])
                < 1e-11
            )
            assert (
                abs(r["predicted_log10_signal"] - math.log10(reference) - pred) < 1e-11
            )
            rows[r["variant"], r["split"]].append((pred, y, w, curve["construct_id"]))
    assert seen == set(expected) and maximum < 1e-10
    metric_error = 0.0
    for (variant, split), values in rows.items():
        p, y, w = map(np.array, zip(*[r[:3] for r in values]))
        metric = scores[variant][split]
        assert abs(float(w.sum()) - 1) < 1e-12
        local = dict(
            mse=float(w @ (p - y) ** 2),
            rmse=math.sqrt(float(w @ (p - y) ** 2)),
            pooled_rmse=float(np.sqrt(np.mean((p - y) ** 2))),
            prediction_sd=float(np.std(p)),
            target_sd=float(np.std(y)),
        )
        if variant != "no_change":
            local.update(
                pearson=float(pearsonr(p, y).statistic),
                spearman=float(spearmanr(p, y).statistic),
            )
        else:
            assert metric["pearson"] is None and metric["spearman"] is None
        for name, value in local.items():
            error = abs(value - metric[name])
            metric_error = max(metric_error, error)
            # Analytically tied one-rate/shared-time predictions may differ in
            # floating-point ranking; assess separately, never waive RMSE.
            if name != "spearman":
                assert error < 1e-10, (variant, split, name, error)
        for parent, by in metric["by_parent"].items():
            mask = np.array([r[3] == parent for r in values])
            rmse = math.sqrt(
                float(np.sum(w[mask] * (p[mask] - y[mask]) ** 2) / w[mask].sum())
            )
            assert abs(rmse - by["rmse"]) < 1e-11
    result = dict(
        passed=True,
        optimizer_replay=replay,
        predictions_verified=len(seen),
        maximum_prediction_error=maximum,
        maximum_metric_error_including_rank_ties=metric_error,
        no_producer_imports=True,
        wall_seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / "results/independent-verification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
