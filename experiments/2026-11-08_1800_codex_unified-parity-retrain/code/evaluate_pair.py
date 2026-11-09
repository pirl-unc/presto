#!/usr/bin/env python3
"""Evaluate completed unified parity fits on the frozen matched development rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import time

import numpy as np

from presto.evaluation.paired_benchmark import (
    affinity_threshold_metrics,
    censor_metrics,
    operational_processing_logit,
    paired_study_bootstrap,
    paired_study_metrics,
    require_aligned_predictions,
    sigmoid,
    study_bootstrap,
    study_id,
)
from presto.evaluation.performance_figures import (
    render_affinity_calibration,
    render_affinity_scatter,
    render_discrimination_curves,
    render_study_deltas,
)
from presto.training.holdout_eval import binary_metrics, regression_metrics

EXPECTED_CENSUS = {
    "ec50": 69,
    "elution_observation": 48530,
    "ic50": 1597,
    "kd_via_ec50": 528,
    "kd_via_ic50": 342,
}
EXPECTED_ROWS = sum(EXPECTED_CENSUS.values())
MODEL_METHODS = ("Presto reference", "Presto parity", "Presto Stage B")
INPUT_KEYS = (
    "reference_predictions",
    "parity_predictions",
    "stage_b_predictions",
    "affinity_comparator",
    "presentation_comparator",
)


def canonical_json(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_json(path: Path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl_gz(path: Path) -> list[dict]:
    result = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: invalid JSON") from error
    return result


def write_json(path: Path, value) -> None:
    path.write_text(canonical_json(value), encoding="utf-8")


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    with open(path, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write(
                    (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()
                )


def verify_inputs(contract_path: Path, input_root: Path) -> tuple[dict, dict[str, Path]]:
    contract = read_json(contract_path)
    if contract.get("schema") != "presto.unified_parity_evaluation.v1":
        raise ValueError("unsupported evaluation contract")
    if contract.get("bootstrap") != {"replicates": 2000, "seed": 42}:
        raise ValueError("bootstrap contract changed")
    expected = {
        "development_rows": EXPECTED_ROWS,
        "endpoint_census": EXPECTED_CENSUS,
    }
    if contract.get("expected") != expected:
        raise ValueError("development cohort contract changed")
    evaluator = contract.get("evaluator", {})
    source_path = Path(__file__).resolve()
    if evaluator.get("sha256") != sha256(source_path):
        raise ValueError("evaluation source differs from the frozen contract")
    specs = contract.get("inputs", {})
    if set(specs) != set(INPUT_KEYS):
        raise ValueError("evaluation contract input set changed")
    paths = {}
    for key in INPUT_KEYS:
        spec = specs[key]
        path = input_root / spec["filename"]
        if not path.is_file():
            raise ValueError(f"missing evaluation input {path}")
        if path.stat().st_size != spec["bytes"] or sha256(path) != spec["sha256"]:
            raise ValueError(f"evaluation input changed: {key}")
        paths[key] = path
    return contract, paths


def assay_key(record: dict) -> str:
    method = record["assay_method"]
    return "/".join(
        (
            record["endpoint"],
            method["preparation"],
            method["mechanism"],
            method["detector"],
            method["review_status"],
        )
    )


def comparator_affinity_method(mhc_class: str) -> str:
    if mhc_class == "I":
        return "NetMHCpan 4.2c"
    if mhc_class == "II":
        return "NetMHCIIpan 4.3e"
    raise ValueError(f"unsupported MHC class {mhc_class}")


def mhcflurry_method(task: str) -> str:
    output = {
        "presentation": "presentation",
        "operational_processing": "processing",
    }[task]
    return f"MHCflurry 2.3.16 {output}"


def affinity_rows(indexed: dict, comparator: list[dict]) -> list[dict]:
    comparator_index = {}
    for row in comparator:
        observation_id = row["observation_id"]
        if observation_id in comparator_index:
            raise ValueError(f"duplicate affinity comparator row {observation_id}")
        comparator_index[observation_id] = row
    ic50_ids = {
        observation_id
        for observation_id, row in indexed["Presto Stage B"].items()
        if row["endpoint"] == "ic50"
    }
    if set(comparator_index) != ic50_ids:
        raise ValueError("affinity comparator does not exactly cover IC50 development rows")

    result = []
    for observation_id in sorted(ic50_ids):
        truth = indexed["Presto Stage B"][observation_id]
        baseline = comparator_index[observation_id]
        mhc_classes = truth["mhc_classes"]
        if len(mhc_classes) != 1:
            raise ValueError(f"affinity row {observation_id} has ambiguous class")
        mhc_class = mhc_classes[0]
        method = truth["assay_method"]
        expected = {
            "peptide": truth["peptide"],
            "mhc_class": mhc_class,
            "endpoint": truth["endpoint"],
            "qualifier": truth["qualifier"],
            "target_log10": truth["target"],
            "raw_nM": truth["raw_value"],
            "source": truth["source"],
            "study": study_id(truth),
            "preparation": method["preparation"],
            "mechanism": method["mechanism"],
            "detector": method["detector"],
            "assay_review_status": method["review_status"],
        }
        for key, value in expected.items():
            observed = baseline.get(key)
            if isinstance(value, float):
                equal = math.isclose(
                    float(observed), value, rel_tol=1e-12, abs_tol=1e-12
                )
            else:
                equal = observed == value
            if not equal:
                raise ValueError(
                    f"affinity comparator truth differs at {observation_id}/{key}: "
                    f"{observed!r} != {value!r}"
                )
        common = {
            "observation_id": observation_id,
            "mhc_class": mhc_class,
            "endpoint": "ic50",
            "assay_method": assay_key(truth),
            "qualifier": truth["qualifier"],
            "target_log10_nM": float(truth["target"]),
            "target": float(truth["target"]),
            "study": study_id(truth),
            "peptide": truth["peptide"],
        }
        for model_method in MODEL_METHODS:
            prediction = float(indexed[model_method][observation_id]["prediction"])
            result.append(
                {
                    **common,
                    "method": model_method,
                    "prediction_log10_nM": prediction,
                    "score": prediction,
                }
            )
        prediction = float(baseline["comparator_log10"])
        result.append(
            {
                **common,
                "method": comparator_affinity_method(mhc_class),
                "prediction_log10_nM": prediction,
                "score": prediction,
            }
        )
    return result


def presentation_rows(indexed: dict, comparator: list[dict]) -> list[dict]:
    comparator_index = {}
    for row in comparator:
        observation_id = row["observation_id"]
        if observation_id in comparator_index:
            raise ValueError(f"duplicate presentation comparator row {observation_id}")
        comparator_index[observation_id] = row
    elution_ids = {
        observation_id
        for observation_id, row in indexed["Presto Stage B"].items()
        if row["endpoint"] == "elution_observation"
    }
    if set(comparator_index) != elution_ids:
        raise ValueError("presentation comparator does not exactly cover development rows")

    result = []
    for observation_id in sorted(elution_ids):
        truth = indexed["Presto Stage B"][observation_id]
        baseline = comparator_index[observation_id]
        provenance = truth["observation"]["provenance"]
        expected = {
            "pair_id": provenance["pair_id"],
            "peptide": truth["peptide"],
            "parent_peptide": provenance["parent_peptide"],
            "sample": provenance["sample_id"],
            "study": truth["group_id"],
            "target": truth["target"],
        }
        for key, value in expected.items():
            observed = baseline.get(key)
            equal = (
                math.isclose(float(observed), float(value), rel_tol=0.0, abs_tol=1e-12)
                if isinstance(value, float)
                else observed == value
            )
            if not equal:
                raise ValueError(
                    f"presentation comparator truth differs at {observation_id}/{key}"
                )
        common = {
            "observation_id": observation_id,
            "study": truth["group_id"],
            "sample": provenance["sample_id"],
            "pair_id": provenance["pair_id"],
            "peptide": truth["peptide"],
            "target": float(truth["target"]),
        }
        for method in MODEL_METHODS:
            row = indexed[method][observation_id]
            result.append(
                {
                    **common,
                    "task": "presentation",
                    "method": method,
                    "score": sigmoid(float(row["prediction"])),
                }
            )
            result.append(
                {
                    **common,
                    "task": "operational_processing",
                    "method": method,
                    "score": sigmoid(operational_processing_logit(row)),
                }
            )
        stage_presentation = next(
            row["score"]
            for row in result[-6:]
            if row["task"] == "presentation" and row["method"] == "Presto Stage B"
        )
        stage_processing = next(
            row["score"]
            for row in result[-6:]
            if row["task"] == "operational_processing"
            and row["method"] == "Presto Stage B"
        )
        if not math.isclose(
            stage_presentation,
            float(baseline["presto_presentation"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"historical presentation derivation differs at {observation_id}")
        if not math.isclose(
            stage_processing,
            float(baseline["presto_processing_ms"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"historical processing derivation differs at {observation_id}")
        result.extend(
            (
                {
                    **common,
                    "task": "presentation",
                    "method": mhcflurry_method("presentation"),
                    "score": float(baseline["presentation_score"]),
                },
                {
                    **common,
                    "task": "operational_processing",
                    "method": mhcflurry_method("operational_processing"),
                    "score": float(baseline["processing_score"]),
                },
            )
        )
    return result


def affinity_summary(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    summaries = []
    bootstraps = []
    methods_by_class = {
        "I": (*MODEL_METHODS, "NetMHCpan 4.2c"),
        "II": (*MODEL_METHODS, "NetMHCIIpan 4.3e"),
    }
    for mhc_class, methods in methods_by_class.items():
        class_rows = [row for row in rows if row["mhc_class"] == mhc_class]
        for method in methods:
            selected = [row for row in class_rows if row["method"] == method]
            exact = [row for row in selected if row["qualifier"] in ("=", "==", "")]
            regression = regression_metrics(
                np.asarray([row["target_log10_nM"] for row in exact]),
                np.asarray([row["prediction_log10_nM"] for row in exact]),
            )
            for metric in ("spearman", "pearson", "rmse", "mae"):
                summaries.append(
                    {
                        "task": "affinity",
                        "mhc_class": mhc_class,
                        "assay_method": "all_ic50",
                        "population": "exact",
                        "estimate_type": "pooled",
                        "method": method,
                        "metric": metric,
                        "value": regression[metric],
                        "n": len(exact),
                    }
                )
            threshold_metrics = affinity_threshold_metrics(selected)
            for metric, value in threshold_metrics.items():
                summaries.append(
                    {
                        "task": "affinity_binding_500nM",
                        "mhc_class": mhc_class,
                        "assay_method": "all_ic50",
                        "population": "qualifier_definite",
                        "estimate_type": "pooled",
                        "method": method,
                        "metric": metric,
                        "value": value,
                        "n": int(threshold_metrics["n"]),
                    }
                )
            censored = censor_metrics(selected)
            for metric, value in censored.items():
                summaries.append(
                    {
                        "task": "affinity_censor",
                        "mhc_class": mhc_class,
                        "assay_method": "all_ic50",
                        "population": "censored",
                        "estimate_type": "pooled",
                        "method": method,
                        "metric": metric,
                        "value": value,
                        "n": int(censored["n"]),
                    }
                )
            for metric in ("spearman", "pearson", "rmse"):
                bootstraps.append(
                    {
                        "task": "affinity",
                        "mhc_class": mhc_class,
                        "comparison": "method_value",
                        **study_bootstrap(
                            [
                                {
                                    **row,
                                    "target": row["target_log10_nM"],
                                    "score": row["prediction_log10_nM"],
                                }
                                for row in exact
                            ],
                            method,
                            metric=metric,
                            kind="regression",
                        ),
                    }
                )
        exact_class = [row for row in class_rows if row["qualifier"] in ("=", "==", "")]
        for first, second in (
            ("Presto parity", "Presto reference"),
            ("Presto parity", "Presto Stage B"),
            ("Presto parity", comparator_affinity_method(mhc_class)),
        ):
            for metric in ("spearman", "pearson", "rmse"):
                bootstraps.append(
                    {
                        "task": "affinity",
                        "mhc_class": mhc_class,
                        "comparison": "paired_difference",
                        **paired_study_bootstrap(
                            exact_class,
                            first,
                            second,
                            metric=metric,
                            kind="regression",
                        ),
                    }
                )

        by_assay = defaultdict(list)
        for row in class_rows:
            by_assay[(row["method"], row["assay_method"])].append(row)
        for (method, method_key), selected in sorted(by_assay.items()):
            exact = [row for row in selected if row["qualifier"] in ("=", "==", "")]
            if len(exact) < 2:
                continue
            regression = regression_metrics(
                np.asarray([row["target_log10_nM"] for row in exact]),
                np.asarray([row["prediction_log10_nM"] for row in exact]),
            )
            for metric in ("spearman", "pearson", "rmse"):
                if metric in regression:
                    summaries.append(
                        {
                            "task": "affinity",
                            "mhc_class": mhc_class,
                            "assay_method": method_key,
                            "population": "exact",
                            "estimate_type": "pooled",
                            "method": method,
                            "metric": metric,
                            "value": regression[metric],
                            "n": len(exact),
                        }
                    )
    return summaries, bootstraps


def binary_summary(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    summaries, bootstraps, per_study = [], [], []
    baseline = {
        task: mhcflurry_method(task)
        for task in ("presentation", "operational_processing")
    }
    for task in sorted(baseline):
        task_rows = [row for row in rows if row["task"] == task]
        methods = (*MODEL_METHODS, baseline[task])
        for method in methods:
            selected = [row for row in task_rows if row["method"] == method]
            metrics = binary_metrics(
                np.asarray([row["target"] for row in selected]),
                np.asarray([row["score"] for row in selected]),
            )
            for metric, value in sorted(metrics.items()):
                summaries.append(
                    {
                        "task": task,
                        "population": "matched_all",
                        "estimate_type": "pooled",
                        "method": method,
                        "metric": metric,
                        "value": value,
                        "n": len(selected),
                    }
                )
            for metric in ("auroc", "auprc"):
                bootstraps.append(
                    {
                        "task": task,
                        "comparison": "method_value",
                        **study_bootstrap(
                            task_rows, method, metric=metric, kind="binary"
                        ),
                    }
                )
        for first, second in (
            ("Presto parity", "Presto reference"),
            ("Presto parity", "Presto Stage B"),
            ("Presto parity", baseline[task]),
        ):
            for metric in ("auroc", "auprc"):
                bootstraps.append(
                    {
                        "task": task,
                        "comparison": "paired_difference",
                        **paired_study_bootstrap(
                            task_rows,
                            first,
                            second,
                            metric=metric,
                            kind="binary",
                        ),
                    }
                )
            for row in paired_study_metrics(
                task_rows, first, second, metric="auroc", kind="binary"
            ):
                per_study.append({"task": task, **row})
    return summaries, bootstraps, per_study


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda row: tuple(str(row.get(key, "")) for key in columns),
            )
        )


def _bootstrap_forest_row(
    bootstraps: list[dict], *, task: str, first: str, second: str, metric: str
) -> dict:
    matches = [
        row
        for row in bootstraps
        if row["task"] == task
        and row["comparison"] == "paired_difference"
        and row["first"] == first
        and row["second"] == second
        and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one pooled bootstrap for {task}/{first}/{second}/{metric}")
    row = matches[0]
    return {
        "study": "pooled_bootstrap",
        "metric": metric,
        "delta": row["estimate"],
        "ci_low": row["ci_low"],
        "ci_high": row["ci_high"],
    }


def render_figures(
    affinity: list[dict],
    binary: list[dict],
    bootstraps: list[dict],
    per_study: list[dict],
    output: Path,
) -> None:
    exact = [row for row in affinity if row["qualifier"] in ("=", "==", "")]
    render_affinity_scatter(exact, output)
    render_affinity_calibration(exact, output)
    for mhc_class in ("I", "II"):
        curves = []
        cutoff = math.log10(500.0)
        for row in affinity:
            if row["mhc_class"] != mhc_class:
                continue
            qualifier = row["qualifier"]
            target = row["target_log10_nM"]
            usable = (
                qualifier in ("=", "==", "")
                or qualifier in ("<=", "<")
                and target <= cutoff
                or qualifier in (">=", ">")
                and target > cutoff
            )
            if usable:
                curves.append(
                    {
                        **row,
                        "target": float(target <= cutoff),
                        "score": -row["prediction_log10_nM"],
                    }
                )
        render_discrimination_curves(
            curves,
            output,
            stem=f"affinity_binding_curves_mhc_{mhc_class.lower()}",
            title=f"MHC-{mhc_class} IC50 binding at 500 nM",
        )
        comparator = comparator_affinity_method(mhc_class)
        exact_class = [
            row
            for row in exact
            if row["mhc_class"] == mhc_class
            and row["method"] in ("Presto parity", comparator)
        ]
        deltas = [
            {**row, "ci_low": row["delta"], "ci_high": row["delta"]}
            for row in paired_study_metrics(
                exact_class,
                "Presto parity",
                comparator,
                metric="spearman",
                kind="regression",
            )
        ]
        deltas.append(
            _bootstrap_forest_row(
                bootstraps,
                task="affinity",
                first="Presto parity",
                second=comparator,
                metric="spearman",
            )
        )
        render_study_deltas(
            deltas,
            output,
            stem=f"affinity_study_deltas_mhc_{mhc_class.lower()}",
            title=f"MHC-{mhc_class} exact IC50: Presto parity − {comparator}",
        )
    for task, title in (
        ("presentation", "MHC-I presentation"),
        ("operational_processing", "MHC-I operational processing"),
    ):
        selected = [row for row in binary if row["task"] == task]
        render_discrimination_curves(
            selected, output, stem=f"{task}_curves", title=title
        )
        deltas = [
            {
                **row,
                "ci_low": row["delta"],
                "ci_high": row["delta"],
            }
            for row in per_study
            if row["task"] == task and row["first"] == "Presto parity"
            and row["second"].startswith("MHCflurry")
        ]
        deltas.append(
            _bootstrap_forest_row(
                bootstraps,
                task=task,
                first="Presto parity",
                second=mhcflurry_method(task),
                metric="auroc",
            )
        )
        render_study_deltas(
            deltas,
            output,
            stem=f"{task}_study_deltas",
            title=f"{title}: Presto parity − MHCflurry AUROC by study",
        )


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux reports KiB.
    return int(value if platform.system() == "Darwin" else value * 1024)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    began = time.monotonic()
    if args.output.exists():
        raise FileExistsError(f"evaluation output already exists: {args.output}")
    contract, paths = verify_inputs(args.contract, args.input_root)
    args.output.mkdir(parents=True)
    for name in ("figures", "predictions", "tables"):
        (args.output / name).mkdir()

    predictions = {
        "Presto reference": read_jsonl_gz(paths["reference_predictions"]),
        "Presto parity": read_jsonl_gz(paths["parity_predictions"]),
        "Presto Stage B": read_jsonl_gz(paths["stage_b_predictions"]),
    }
    indexed = require_aligned_predictions(predictions, expected_rows=EXPECTED_ROWS)
    census = Counter(row["endpoint"] for row in predictions["Presto Stage B"])
    if dict(sorted(census.items())) != EXPECTED_CENSUS:
        raise ValueError(f"development endpoint census changed: {dict(census)}")

    affinity = affinity_rows(indexed, read_json(paths["affinity_comparator"]))
    binary = presentation_rows(
        indexed, read_jsonl_gz(paths["presentation_comparator"])
    )
    affinity_metrics, affinity_bootstraps = affinity_summary(affinity)
    binary_metrics_rows, binary_bootstraps, per_study = binary_summary(binary)
    all_metrics = affinity_metrics + binary_metrics_rows
    all_bootstraps = affinity_bootstraps + binary_bootstraps

    write_jsonl_gz(args.output / "predictions" / "matched-affinity.jsonl.gz", affinity)
    write_jsonl_gz(
        args.output / "predictions" / "matched-presentation-processing.jsonl.gz",
        binary,
    )
    write_json(args.output / "tables" / "metrics.json", all_metrics)
    write_json(args.output / "tables" / "study-bootstrap.json", all_bootstraps)
    write_json(args.output / "tables" / "per-study.json", per_study)
    write_csv(args.output / "tables" / "metrics.csv", all_metrics)
    write_csv(args.output / "tables" / "study-bootstrap.csv", all_bootstraps)
    write_csv(args.output / "tables" / "per-study.csv", per_study)
    render_figures(
        affinity, binary, all_bootstraps, per_study, args.output / "figures"
    )

    files = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(args.output))] = artifact(path, args.output)
    receipt = {
        "schema": "presto.unified_parity_evaluation_receipt.v1",
        "status": "completed",
        "contract_sha256": sha256(args.contract),
        "inputs": contract["inputs"],
        "development_rows": EXPECTED_ROWS,
        "endpoint_census": EXPECTED_CENSUS,
        "affinity_long_rows": len(affinity),
        "presentation_processing_long_rows": len(binary),
        "bootstrap": contract["bootstrap"],
        "test_accessed": False,
        "locked_confirmation_accessed": False,
        "elapsed_seconds": time.monotonic() - began,
        "peak_rss_bytes": peak_rss_bytes(),
        "files": files,
    }
    write_json(args.output / "receipt.json", receipt)
    print(canonical_json(receipt), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
