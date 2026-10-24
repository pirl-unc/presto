"""Independent final-model arithmetic and source/metric closure; never fit."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import math
import resource
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMILY = Path(__file__).resolve().parents[1]
AA = "ACDEFGHIKLMNPQRSTVWY"


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt") as stream:
        return json.load(stream)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same(actual, expected):
    if isinstance(expected, float):
        assert math.isclose(actual, expected, abs_tol=1e-10, rel_tol=1e-10), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def coordinates(raw, arrays, prefix=""):
    keep = arrays[prefix + "keep"]
    z = (raw[keep] - arrays[prefix + "mean"][keep]) / arrays[prefix + "scale"]
    return np.r_[1.0, z @ arrays[prefix + "projection"]]


def amount(internal, left, right):
    total = 1.0 + internal
    if left and right:
        return (
            left
            * right
            * (2 * total + left + right)
            / (total * (total + left) * (total + right) * (total + left + right))
        )
    if left or right:
        boundary = left or right
        return boundary / (total * (total + boundary))
    return 1.0 / total


def physical_predictions(arm, groups, parents, origins, transforms, coverage, metric_module):
    theta = np.load(FAMILY / "results/arms" / arm / "fit.npz")["theta"]
    rank = 38 if arm.startswith("sequence_") else 0
    assert rank == (coverage["bond_features"]["n_supported_coordinates"] if rank else 0)
    nc = coverage["condition_features"]["n_parameters"]
    assert len(theta) == 1 + rank + 2 * nc
    raw_coefficients = np.zeros(40)
    if rank:
        raw_coefficients[transforms["bond_keep"]] = (
            transforms["bond_projection"] @ theta[1 : 1 + rank]
        ) / transforms["bond_scale"]
    intercept = theta[0] - transforms["bond_center"] @ raw_coefficients
    rates = {}
    for parent, sequence in parents.items():
        indices = np.array([AA.index(a) for a in sequence])
        rates[parent] = np.exp(
            intercept + raw_coefficients[indices[:-1]] + raw_coefficients[20 + indices[1:]]
        )
    names = coverage["condition_features"]["all_feature_names"]
    condition = {}
    cache, result = {}, {}
    for group in groups:
        cid = group["condition_id"]
        if cid not in condition:
            raw = np.array([metric_module.feature(group, name) for name in names])
            cx = coordinates(raw, transforms, "condition_")
            condition[cid] = (
                math.exp(float(cx @ theta[1 + rank : 1 + rank + nc])),
                float(cx @ theta[1 + rank + nc :]),
            )
        peptide = group["peptide_sequence"]
        key = cid, peptide
        if key not in cache:
            scale, offset = condition[cid]
            controls, treated = [], []
            for parent, start, end in sorted(origins[peptide]):
                r = rates[parent]
                internal = math.fsum(r[start : end - 1])
                left = float(r[start - 1]) if start else 0.0
                right = float(r[end - 1]) if end < len(parents[parent]) else 0.0
                controls.append(amount(internal, left, right))
                treated.append(amount(internal * scale, left * scale, right * scale))
            nparents = len({p for p, _, _ in origins[peptide]})
            control = math.fsum(controls) / nparents
            treatment = math.fsum(treated) / nparents
            cache[key] = (
                offset + math.log10(treatment / control),
                math.log(control),
                math.log(treatment),
            )
        result[group["response_group_id"]] = cache[key]
    return result


def main():
    start = time.perf_counter()
    output = FAMILY / "results/independent-verification.json"
    assert not output.exists()
    config = read(FAMILY / "reproduce/launch.json")
    execution = read(FAMILY / "reproduce/execution.json")
    for path, digest in execution["source_sha256"].items():
        assert sha(path) == digest, path
    final = read(FAMILY / "results/final-evaluation-inputs.json")
    assert sha(FAMILY / "reproduce/execution.json") == final["source_receipt_sha256"]
    for path, digest in final["checkpoints"].items():
        assert sha(FAMILY / path) == digest, path
    sealed = read(FAMILY / "reproduce/pre-evaluation-receipt.json")
    for path, digest in sealed["sha256"].items():
        assert sha(FAMILY / path) == digest, path
    directory = Path(config["recipe_directory"])
    assert sha(directory / "manifest.json") == config["recipe_manifest_sha256"]
    manifest = read(directory / "manifest.json")
    for name in ("examples", "parents", "occurrences"):
        assert (
            sha(directory / (name + ".json.gz")) == manifest["files"][name + ".json.gz"]["sha256"]
        )
    groups = [g for g in read(directory / "examples.json.gz") if g["full_parent_model_supported"]]
    by_observation = {o["observation_id"]: (g, o) for g in groups for o in g["observations"]}
    assert len(by_observation) == sum(len(g["observations"]) for g in groups)
    normalizers = {
        s: math.fsum(g["group_weight"] for g in groups if g["split"] == s)
        for s in ("train", "val", "test")
    }
    spec = importlib.util.spec_from_file_location(
        "native_independent_metrics", Path(config["baseline_family"]) / "analysis/verify.py"
    )
    metrics_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics_module)
    coverage = read(FAMILY / "results/data-coverage.json")
    transforms = np.load(FAMILY / "results/feature-transforms.npz")
    parents = {p["sequence_sha256"]: p["sequence"] for p in read(directory / "parents.json.gz")}
    eligible = {g["peptide_sequence"] for g in groups}
    origins = defaultdict(set)
    for o in read(directory / "occurrences.json.gz"):
        if o["peptide_sequence"] in eligible:
            parent, a, b = o["reference_sequence_sha256"], o["start_1based"] - 1, o["end_1based"]
            assert parents[parent][a:b] == o["peptide_sequence"]
            origins[o["peptide_sequence"]].add((parent, a, b))
    used_parents = {p for values in origins.values() for p, _, _ in values}
    parents = {p: parents[p] for p in used_parents}
    summary = read(FAMILY / "results/evaluation/summary.json")
    strata = read(FAMILY / "results/evaluation/stratified-metrics.json")
    baselines = {r["variant"]: r for r in summary["baseline_fits"]}
    all_checks, count, metric_count, max_error = [], 0, 0, 0.0
    for path in sorted((FAMILY / "results/evaluation").glob("*-predictions.jsonl.gz")):
        variant = path.name.removesuffix("-predictions.jsonl.gz")
        with gzip.open(path, "rt") as stream:
            rows = [json.loads(line) for line in stream]
        assert len(rows) == len(by_observation)
        assert {r["observation_id"] for r in rows} == by_observation.keys()
        if variant in baselines:
            fit = baselines[variant]
            expected, cache = {}, {}
            arrays = (
                None
                if "constant" in fit
                else np.load(FAMILY / "results/baselines" / (variant + ".npz"))
            )
            for group in groups:
                if arrays is None:
                    value = fit["constant"]
                else:
                    raw = tuple(
                        metrics_module.feature(group, name)
                        for name in fit["features"]["all_feature_names"]
                    )
                    if raw not in cache:
                        cx = coordinates(np.array(raw), arrays)
                        cache[raw] = float(math.fsum(cx * arrays["theta"]))
                    value = cache[raw]
                expected[group["response_group_id"]] = (value,)
        else:
            expected = physical_predictions(
                variant, groups, parents, origins, transforms, coverage, metrics_module
            )
        local_error = 0.0
        for row in rows:
            group, original = by_observation[row["observation_id"]]
            for key, value in original.items():
                if key == "weight":
                    assert abs(row["weight"] - value / normalizers[group["split"]]) < 1e-12
                    assert row["canonical_weight"] == value
                else:
                    assert row[key] == value, (variant, key)
            for key in (
                "response_group_id",
                "condition_id",
                "condition",
                "component_id",
                "split",
                "peptide_sequence",
                "modifications",
                "full_parent_model_supported",
            ):
                assert row[key] == group[key], key
            assert row["censor_code"] == {None: 0, "lt": -1, "gt": 1}[original["inequality"]]
            assert row["n_candidate_parent_sequences"] == len(
                {p for p, _, _ in origins[group["peptide_sequence"]]}
            )
            assert row["n_candidate_occurrences"] == len(origins[group["peptide_sequence"]])
            values = expected[group["response_group_id"]]
            fields = ["prediction_log10_ratio"]
            if len(values) == 3:
                fields += ["log_control_amount_per_unit_input", "log_treated_amount_per_unit_input"]
            for field, value in zip(fields, values):
                err = abs(row[field] - value)
                assert err < 1e-9, (variant, field, err)
                local_error = max(local_error, err)
        for split in ("train", "val", "test"):
            selected = [r for r in rows if r["split"] == split]
            actual = metrics_module.metrics(selected)
            stored = next(
                m for m in summary["metrics"] if m["variant"] == variant and m["split"] == split
            )
            for key, value in actual.items():
                same(stored[key], value)
            metric_count += 1
            for field in (
                "condition_id",
                "component_id",
                "modified_form",
                "multiple_parent_candidates",
            ):
                grouped = defaultdict(list)
                for row in selected:
                    value = (
                        bool(row["modifications"])
                        if field == "modified_form"
                        else row["n_candidate_parent_sequences"] > 1
                        if field == "multiple_parent_candidates"
                        else row[field]
                    )
                    grouped[str(value)].append(row)
                known = {
                    s["stratum_id"]: s
                    for s in strata
                    if s["variant"] == variant and s["split"] == split and s["stratum"] == field
                }
                assert grouped.keys() == known.keys()
                for label, members in grouped.items():
                    for key, value in metrics_module.metrics(members).items():
                        same(known[label][key], value)
                    metric_count += 1
        count += len(rows)
        max_error = max(max_error, local_error)
        all_checks.append(
            {
                "variant": variant,
                "predictions": len(rows),
                "max_prediction_or_log_amount_error": local_error,
            }
        )
    elapsed = time.perf_counter() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert elapsed < 180 and rss < 4 * 1024**3
    result = {
        "status": "passed",
        "n_source_linked_predictions": count,
        "n_global_and_stratified_metric_records": metric_count,
        "max_prediction_or_log_amount_error": max_error,
        "seconds": elapsed,
        "peak_rss_bytes": rss,
        "checks": all_checks,
        "test_evaluated": True,
        "scope": "Final predictions and source/metric arithmetic; no optimizer replay or independent biological validation",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
