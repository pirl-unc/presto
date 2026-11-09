"""Independent original-row reconstruction and convex optimality certificates."""

import gzip
import hashlib
import json
import math
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]


def main():
    tick = time.perf_counter()
    cfg = json.loads((FAMILY / "reproduce/launch.json").read_text())
    path = Path(cfg["recipe_directory"])
    assert (
        hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest()
        == cfg["recipe_manifest_sha256"]
    )
    manifest = json.loads((path / "manifest.json").read_text())
    assert (
        hashlib.sha256((path / "examples.json").read_bytes()).hexdigest()
        == manifest["files"]["examples.json"]["sha256"]
    )
    examples = [
        r
        for r in json.loads((path / "examples.json").read_text())
        if r["split"] == "train"
    ]
    conditions = {r["condition_id"]: r["construct_id"] for r in examples}
    parent_conditions = Counter(conditions.values())
    curve_counts = Counter(r["condition_id"] for r in examples)
    expected = {}
    span_groups = defaultdict(list)
    for curve in examples:
        times = curve["common_positive_times"]
        reference = times[0]
        values = {v["time_hours"]: v["target_value"] for v in curve["observations"]}
        weight = 1 / (
            len(parent_conditions)
            * parent_conditions[curve["construct_id"]]
            * curve_counts[curve["condition_id"]]
            * (len(times) - 1)
        )
        for t in times[1:]:
            row = dict(
                curve=curve,
                time=t,
                reference=reference,
                signal=values[t],
                reference_signal=values[reference],
                target=math.log10(values[t] / values[reference]),
                weight=weight,
            )
            key = curve["curve_id"], t
            assert key not in expected
            expected[key] = row
            span_groups[
                curve["condition_id"], curve["start_0based"], curve["end_0based"]
            ].append(row)
    maximum = 0.0
    optimality = 0.0
    count = 0
    predictions = {}
    perturbed_rejected = False
    with gzip.open(FAMILY / "results/shape-curves.jsonl.gz", "rt") as stream:
        seen = set()
        for line in stream:
            row = json.loads(line)
            key = row["condition"], row["start"], row["end"]
            assert key not in seen
            seen.add(key)
            source = span_groups[key]
            times = np.array(
                sorted({r["time"] for r in source} | {r["reference"] for r in source})
            )
            assert times.tolist() == row["times"]
            assert all(r["reference"] == times[0] for r in source)
            w = np.array(
                [sum(r["weight"] for r in source if r["time"] == t) for t in times[1:]]
            )
            mean = np.array(
                [
                    sum(r["weight"] * r["target"] for r in source if r["time"] == t)
                    / w[i]
                    for i, t in enumerate(times[1:])
                ]
            )
            assert (
                np.max(np.abs(w - row["weights"])) < 1e-14
                and np.max(np.abs(mean - row["means"])) < 1e-12
            )
            n = len(times) - 1

            def operator(v):
                z = np.r_[0.0, v]
                return np.r_[np.diff(z), np.diff(np.diff(z) / np.diff(times))]

            a = np.column_stack([operator(v) for v in np.eye(n)])
            boundary = int(row["start"] > 0) + int(
                row["end"] < len(source[0]["curve"]["parent_sequence"])
            )
            assert boundary == row["boundaries"]
            for variant, extra in (("immediate_shape", 0), ("entry_shape", 1)):
                z = np.array(row[variant])
                dual = np.array(row[variant + "_dual"])
                b = np.r_[
                    (boundary + extra) * np.log10(times[1:] / times[:-1]),
                    np.zeros(n - 1),
                ]
                slack = a @ z - b
                stationarity = w / w.sum() * (z - mean) + a.T @ dual
                error = max(
                    float(np.max(np.abs(stationarity))),
                    float(max(0, slack.max())),
                    float(max(0, -dual.min())),
                    float(np.max(np.abs(dual * slack))),
                )
                assert error < 1e-9
                optimality = max(optimality, error)
                count += 1
                altered = z.copy()
                altered[0] += 0.1
                assert (
                    np.max(np.abs(w / w.sum() * (altered - mean) + a.T @ dual)) > 1e-6
                )
                perturbed_rejected = True
            for r in source:
                j = times.tolist().index(r["time"]) - 1
                predictions[r["curve"]["curve_id"], r["time"]] = {
                    variant: row[variant][j]
                    for variant in ("immediate_shape", "entry_shape")
                }
                predictions[r["curve"]["curve_id"], r["time"]]["replicate_mean"] = mean[
                    j
                ]
        assert seen == set(span_groups)
    totals = defaultdict(float)
    seen = set()
    with gzip.open(FAMILY / "results/shape-predictions.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["curve_id"], row["time"]
            assert key not in seen
            seen.add(key)
            r = expected[key]
            assert (
                row["construct_id"] == r["curve"]["construct_id"]
                and row["condition_id"] == r["curve"]["condition_id"]
            )
            for name in ("signal", "reference_signal", "weight"):
                assert row[name] == r[name]
            assert (
                row["reference_time"] == r["reference"]
                and abs(row["target"] - r["target"]) < 1e-12
            )
            for v, p in predictions[key].items():
                error = abs(row[v] - p)
                maximum = max(maximum, error)
                assert error < 1e-12
                totals[v] += r["weight"] * (p - r["target"]) ** 2
    assert seen == set(expected)
    metrics = json.loads((FAMILY / "results/shape-metrics.json").read_text())
    for variant, mse in totals.items():
        assert abs(mse - metrics[variant]["mse"]) < 1e-12
    result = dict(
        curve_certificates_verified=count,
        predictions_verified=len(seen),
        max_prediction_error=maximum,
        max_kkt_error=optimality,
        per_curve_perturbed_optimality_certificates_rejected=perturbed_rejected,
        mse=dict(totals),
        no_heldout_targets_constructed=True,
        wall_seconds=time.perf_counter() - tick,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    (FAMILY / "results/independent-shape-verification.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
