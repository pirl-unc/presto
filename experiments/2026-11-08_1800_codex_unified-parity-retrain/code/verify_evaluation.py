#!/usr/bin/env python3
"""Independently replay saved-row point estimates from a parity evaluation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


EXPECTED_ENDPOINTS = {
    "ec50": 69,
    "ic50": 1597,
    "kd_via_ec50": 528,
    "kd_via_ic50": 342,
}
EXPECTED_AFFINITY_ROWS = 3 * sum(EXPECTED_ENDPOINTS.values()) + EXPECTED_ENDPOINTS["ic50"]
EXPECTED_BINARY_ROWS = 48530 * 8
TOLERANCE = 1e-12


def canonical_json(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def grouped(rows: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    return result


def regression_metric(rows: list[dict], metric: str) -> float:
    target = np.asarray([row["target_log10_nM"] for row in rows], dtype=float)
    prediction = np.asarray([row["prediction_log10_nM"] for row in rows], dtype=float)
    if metric == "spearman":
        return float(spearmanr(target, prediction).statistic)
    if metric == "pearson":
        return float(pearsonr(target, prediction).statistic)
    if metric == "rmse":
        return float(np.sqrt(np.mean(np.square(prediction - target))))
    raise ValueError(f"unsupported regression metric {metric}")


def binary_metric(rows: list[dict], metric: str) -> float:
    target = np.asarray([row["target"] for row in rows], dtype=float)
    score = np.asarray([row["score"] for row in rows], dtype=float)
    if metric == "auroc":
        return float(roc_auc_score(target, score))
    if metric == "auprc":
        return float(average_precision_score(target, score))
    raise ValueError(f"unsupported binary metric {metric}")


def equal_study(rows: list[dict], metric: str, kind: str) -> float:
    values = []
    for study_rows in grouped(rows, "study").values():
        values.append(
            regression_metric(study_rows, metric)
            if kind == "regression"
            else binary_metric(study_rows, metric)
        )
    return float(np.mean(values))


def require_close(observed: float, expected: float, label: str) -> float:
    delta = abs(observed - expected)
    if not math.isfinite(observed) or delta > TOLERANCE:
        raise ValueError(
            f"independent replay differs for {label}: "
            f"{observed!r} != {expected!r} (delta={delta!r})"
        )
    return delta


def unique_row_count(rows: list[dict], endpoint: str) -> int:
    return len({row["observation_id"] for row in rows if row["endpoint"] == endpoint})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)

    affinity_path = args.evaluation_root / "predictions/matched-affinity.jsonl.gz"
    binary_path = args.evaluation_root / "predictions/matched-presentation-processing.jsonl.gz"
    metrics_path = args.evaluation_root / "tables/metrics.json"
    bootstrap_path = args.evaluation_root / "tables/study-bootstrap.json"
    paths = (affinity_path, binary_path, metrics_path, bootstrap_path)
    for path in paths:
        if not path.is_file():
            raise ValueError(f"missing evaluation artifact {path}")

    affinity = read_jsonl_gz(affinity_path)
    binary = read_jsonl_gz(binary_path)
    metrics = read_json(metrics_path)
    bootstraps = read_json(bootstrap_path)
    if len(affinity) != EXPECTED_AFFINITY_ROWS:
        raise ValueError(f"unexpected affinity long-table rows: {len(affinity)}")
    if len(binary) != EXPECTED_BINARY_ROWS:
        raise ValueError(f"unexpected binary long-table rows: {len(binary)}")
    census = {endpoint: unique_row_count(affinity, endpoint) for endpoint in EXPECTED_ENDPOINTS}
    if census != EXPECTED_ENDPOINTS:
        raise ValueError(f"affinity endpoint census changed: {census}")

    comparisons = []
    equal_values = {}
    exact_ic50 = [
        row for row in affinity if row["endpoint"] == "ic50" and row["qualifier"] in ("", "=", "==")
    ]
    for (mhc_class, method), rows in sorted(grouped(exact_ic50, "mhc_class", "method").items()):
        for metric in ("spearman", "pearson", "rmse"):
            observed = regression_metric(rows, metric)
            matches = [
                row
                for row in metrics
                if row.get("task") == "affinity"
                and row.get("mhc_class") == mhc_class
                and row.get("assay_method") == "all_ic50"
                and row.get("population") == "exact"
                and row.get("method") == method
                and row.get("metric") == metric
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one pooled affinity metric for {mhc_class}/{method}/{metric}"
                )
            delta = require_close(
                observed, float(matches[0]["value"]), f"{mhc_class}/{method}/{metric}/pooled"
            )
            comparisons.append(
                {
                    "task": "affinity",
                    "mhc_class": mhc_class,
                    "method": method,
                    "metric": metric,
                    "estimate_type": "pooled",
                    "value": observed,
                    "absolute_delta": delta,
                }
            )
            equal_values[("affinity", mhc_class, method, metric)] = equal_study(
                rows, metric, "regression"
            )

    for (task, method), rows in sorted(grouped(binary, "task", "method").items()):
        for metric in ("auroc", "auprc"):
            observed = binary_metric(rows, metric)
            matches = [
                row
                for row in metrics
                if row.get("task") == task
                and row.get("population") == "matched_all"
                and row.get("method") == method
                and row.get("metric") == metric
            ]
            if len(matches) != 1:
                raise ValueError(f"expected one pooled binary metric for {task}/{method}/{metric}")
            delta = require_close(
                observed, float(matches[0]["value"]), f"{task}/{method}/{metric}/pooled"
            )
            comparisons.append(
                {
                    "task": task,
                    "method": method,
                    "metric": metric,
                    "estimate_type": "pooled",
                    "value": observed,
                    "absolute_delta": delta,
                }
            )
            equal_values[(task, None, method, metric)] = equal_study(rows, metric, "binary")

    for row in bootstraps:
        task = row["task"]
        mhc_class = row.get("mhc_class")
        metric = row["metric"]
        if row["comparison"] == "method_value":
            key = (task, mhc_class, row["method"], metric)
            observed = equal_values[key]
            label = f"{task}/{mhc_class}/{row['method']}/{metric}/equal-study"
        else:
            first = equal_values[(task, mhc_class, row["first"], metric)]
            second = equal_values[(task, mhc_class, row["second"], metric)]
            observed = first - second
            label = f"{task}/{mhc_class}/{row['first']}-{row['second']}/{metric}/equal-study"
        delta = require_close(observed, float(row["estimate"]), label)
        comparisons.append(
            {
                "task": task,
                "mhc_class": mhc_class,
                "method": row.get("method") or f"{row['first']} minus {row['second']}",
                "metric": metric,
                "estimate_type": row["comparison"],
                "value": observed,
                "absolute_delta": delta,
            }
        )

    receipt = {
        "schema": "presto.unified_parity_evaluation_verification.v1",
        "status": "passed",
        "implementation": "SciPy correlations plus scikit-learn AUROC/AUPRC",
        "tolerance": TOLERANCE,
        "affinity_long_rows": len(affinity),
        "binary_long_rows": len(binary),
        "endpoint_census": census,
        "comparisons": len(comparisons),
        "maximum_absolute_delta": max(row["absolute_delta"] for row in comparisons),
        "inputs": {
            str(path.relative_to(args.evaluation_root)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in paths
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(receipt), encoding="utf-8")
    print(canonical_json(receipt), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
