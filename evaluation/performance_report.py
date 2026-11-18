"""Generate immutable, timestamped Presto performance-report bundles."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import tempfile

from .performance_figures import render_comparator_overview, render_data_composition

SCHEMA_VERSION = 1
REPORT_ENGINE_VERSION = "2"
ROLES = {"train", "calibration", "development", "confirmation", "test"}
SHA256 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
METRIC_COLUMNS = (
    "comparison_id",
    "cohort_id",
    "task",
    "output_id",
    "split",
    "estimate_type",
    "method",
    "method_output_id",
    "metric",
    "value",
    "interval_kind",
    "ci_low",
    "ci_high",
    "higher_is_better",
    "eligible_rows",
    "studies",
    "primary",
)


def sha256_file(path: os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _write_json(path: Path, value) -> None:
    path.write_text(_canonical_json(value), encoding="utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_identifier(value, field: str) -> None:
    _require(isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None, field)


def _validate_artifact(artifact: dict, field: str, manifest_dir: Path) -> None:
    _require(isinstance(artifact, dict), f"{field} must be an object")
    for key in ("uri", "sha256", "bytes", "license"):
        _require(key in artifact, f"{field}.{key} is required")
    _require(isinstance(artifact["uri"], str) and artifact["uri"], f"{field}.uri")
    _require(SHA256.fullmatch(str(artifact["sha256"])) is not None, f"{field}.sha256")
    _require(
        isinstance(artifact["bytes"], int) and artifact["bytes"] >= 0,
        f"{field}.bytes",
    )
    _require(
        isinstance(artifact["license"], str) and artifact["license"],
        f"{field}.license",
    )
    include = artifact.get("include_in_report", False)
    _require(isinstance(include, bool), f"{field}.include_in_report")
    local_path = artifact.get("local_path")
    if include:
        _require(isinstance(local_path, str) and local_path, f"{field}.local_path")
    if local_path:
        path = Path(local_path)
        if not path.is_absolute():
            path = manifest_dir / path
        _require(path.is_file(), f"{field}.local_path does not exist")
        _require(path.stat().st_size == artifact["bytes"], f"{field}.bytes mismatch")
        _require(sha256_file(path) == artifact["sha256"], f"{field}.sha256 mismatch")


def _timestamp(value: str) -> datetime:
    _require(isinstance(value, str) and value.endswith("Z"), "timestamp_utc must end in Z")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ValueError("timestamp_utc must be ISO-8601 UTC") from exc
    _require(parsed.tzinfo == timezone.utc, "timestamp_utc must be UTC")
    _require(parsed.microsecond == 0, "timestamp_utc must have whole-second precision")
    return parsed


def validate_manifest(manifest: dict, manifest_path: Path) -> None:
    """Validate provenance, split semantics, metric pairing, and local hashes."""
    _require(isinstance(manifest, dict), "manifest must be an object")
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    _timestamp(manifest.get("timestamp_utc"))

    git = manifest.get("git")
    _require(isinstance(git, dict), "git is required")
    _require(COMMIT.fullmatch(str(git.get("commit", ""))) is not None, "git.commit")
    _require(isinstance(git.get("dirty"), bool), "git.dirty")

    state = manifest.get("state")
    _require(isinstance(state, dict), "state is required")
    for key in ("headline", "status", "qualified_claims", "blocked_claims", "limitations"):
        _require(key in state, f"state.{key} is required")
    _require(state["status"] in {"development", "candidate", "released"}, "state.status")
    for key in ("qualified_claims", "blocked_claims", "limitations"):
        _require(
            isinstance(state[key], list) and all(isinstance(v, str) and v for v in state[key]),
            f"state.{key}",
        )

    model = manifest.get("model")
    _require(isinstance(model, dict), "model is required")
    _validate_identifier(model.get("id"), "model.id")
    _require(isinstance(model.get("training_recipe"), dict), "model.training_recipe")
    _validate_artifact(model.get("checkpoint"), "model.checkpoint", manifest_path.parent)

    data = manifest.get("data")
    _require(isinstance(data, dict), "data is required")
    datasets = data.get("datasets")
    _require(isinstance(datasets, list) and datasets, "data.datasets")
    seen = set()
    roles = set()
    for index, dataset in enumerate(datasets):
        field = f"data.datasets[{index}]"
        _require(isinstance(dataset, dict), field)
        _validate_identifier(dataset.get("id"), f"{field}.id")
        _require(dataset["id"] not in seen, f"duplicate dataset id {dataset['id']}")
        seen.add(dataset["id"])
        role = dataset.get("role")
        _require(role in ROLES, f"{field}.role")
        roles.add(role)
        _require(isinstance(dataset.get("rows"), int) and dataset["rows"] >= 0, f"{field}.rows")
        for key in ("source", "curation_contract", "split_contract"):
            _require(isinstance(dataset.get(key), str) and dataset[key], f"{field}.{key}")
        _validate_artifact(dataset.get("artifact"), f"{field}.artifact", manifest_path.parent)
    _require("train" in roles, "data.datasets must include TRAIN")
    _require(roles & {"development", "confirmation", "test"}, "evaluation dataset is required")
    test_status = data.get("test_status")
    _require(isinstance(test_status, dict), "data.test_status is required")
    _require(isinstance(test_status.get("available"), bool), "data.test_status.available")
    if "test" in roles:
        _require(test_status["available"], "TEST rows require test_status.available=true")
    else:
        _require(not test_status["available"], "test_status conflicts with dataset roles")
        _require(
            isinstance(test_status.get("reason"), str) and test_status["reason"],
            "development-only reports require data.test_status.reason",
        )
    audits = data.get("overlap_audits")
    _require(isinstance(audits, list), "data.overlap_audits")

    comparators = manifest.get("comparators")
    _require(isinstance(comparators, list), "comparators is required")
    for index, comparator in enumerate(comparators):
        field = f"comparators[{index}]"
        _validate_identifier(comparator.get("id"), f"{field}.id")
        _require(isinstance(comparator.get("version"), str), f"{field}.version")
        _validate_artifact(comparator.get("receipt"), f"{field}.receipt", manifest_path.parent)

    prediction_artifacts = manifest.get("prediction_artifacts")
    _require(isinstance(prediction_artifacts, list), "prediction_artifacts is required")
    prediction_ids = set()
    for index, artifact in enumerate(prediction_artifacts):
        field = f"prediction_artifacts[{index}]"
        _validate_identifier(artifact.get("id"), f"{field}.id")
        _require(artifact["id"] not in prediction_ids, f"duplicate prediction id {artifact['id']}")
        prediction_ids.add(artifact["id"])
        _require(artifact.get("split") in ROLES - {"train"}, f"{field}.split")
        _validate_artifact(artifact, field, manifest_path.parent)

    supplemental = manifest.get("supplemental_artifacts", [])
    _require(isinstance(supplemental, list), "supplemental_artifacts must be a list")
    supplemental_ids = set()
    for index, artifact in enumerate(supplemental):
        field = f"supplemental_artifacts[{index}]"
        _validate_identifier(artifact.get("id"), f"{field}.id")
        _require(
            artifact["id"] not in supplemental_ids,
            f"duplicate supplemental artifact id {artifact['id']}",
        )
        supplemental_ids.add(artifact["id"])
        _require(artifact.get("kind") in {"figure", "table"}, f"{field}.kind")
        _validate_artifact(artifact, field, manifest_path.parent)

    metrics = manifest.get("metrics")
    _require(isinstance(metrics, list) and metrics, "metrics is required")
    paired = {}
    for index, metric in enumerate(metrics):
        field = f"metrics[{index}]"
        _require(isinstance(metric, dict), field)
        for key in (
            "comparison_id",
            "cohort_id",
            "task",
            "output_id",
            "split",
            "method",
            "metric",
        ):
            _validate_identifier(metric.get(key), f"{field}.{key}")
        _require(
            metric.get("estimate_type") in {"method_value", "paired_difference"},
            f"{field}.estimate_type",
        )
        _validate_identifier(
            metric.get("method_output_id", metric["output_id"]),
            f"{field}.method_output_id",
        )
        _require(metric["split"] in ROLES - {"train", "calibration"}, f"{field}.split")
        _require(
            isinstance(metric.get("value"), (int, float)) and math.isfinite(metric["value"]),
            f"{field}.value",
        )
        interval_kind = metric.get("interval_kind")
        _require(
            interval_kind in {"bootstrap_95", "analytic_95", "not_available"},
            f"{field}.interval_kind",
        )
        if interval_kind == "not_available":
            _require(
                metric.get("ci_low") is None and metric.get("ci_high") is None,
                f"{field} must use null bounds when no interval is available",
            )
        else:
            for key in ("ci_low", "ci_high"):
                _require(
                    isinstance(metric.get(key), (int, float)) and math.isfinite(metric[key]),
                    f"{field}.{key}",
                )
            _require(
                metric["ci_low"] <= metric["value"] <= metric["ci_high"],
                f"{field}.ci",
            )
        _require(isinstance(metric.get("higher_is_better"), bool), f"{field}.higher_is_better")
        _require(isinstance(metric.get("primary"), bool), f"{field}.primary")
        _require(
            isinstance(metric.get("eligible_rows"), int) and metric["eligible_rows"] > 0,
            f"{field}.eligible_rows",
        )
        studies = metric.get("studies")
        _require(studies is None or isinstance(studies, int) and studies > 0, f"{field}.studies")
        key = (
            metric["comparison_id"],
            metric["cohort_id"],
            metric["task"],
            metric["output_id"],
            metric["split"],
            metric["metric"],
            metric["estimate_type"],
        )
        contract = (
            metric["eligible_rows"],
            studies,
            metric["higher_is_better"],
            metric["interval_kind"],
        )
        _require(
            key not in paired or paired[key] == contract,
            f"{field} changes matched-row contract",
        )
        paired[key] = contract
    for key in paired:
        if key[-1] == "paired_difference":
            continue
        methods = {
            metric["method"]
            for metric in metrics
            if key
            == (
                metric["comparison_id"],
                metric["cohort_id"],
                metric["task"],
                metric["output_id"],
                metric["split"],
                metric["metric"],
                metric["estimate_type"],
            )
        }
        _require(len(methods) >= 2, f"comparison {key[0]} must contain at least two methods")

    gates = manifest.get("promotion_gates")
    _require(isinstance(gates, list) and gates, "promotion_gates is required")
    for index, gate in enumerate(gates):
        _validate_identifier(gate.get("id"), f"promotion_gates[{index}].id")
        _require(
            gate.get("status") in {"pass", "fail", "not_evaluated"},
            f"promotion_gates[{index}].status",
        )
        _require(
            isinstance(gate.get("reason"), str) and gate["reason"],
            f"promotion_gates[{index}].reason",
        )


def _environment() -> dict:
    packages = {}
    for name in ("matplotlib", "numpy", "presto"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "source-checkout"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def _copy_predictions(manifest: dict, manifest_dir: Path, output: Path) -> list[dict]:
    result = []
    for artifact in sorted(manifest["prediction_artifacts"], key=lambda row: row["id"]):
        record = dict(artifact)
        if artifact.get("include_in_report"):
            source = Path(artifact["local_path"])
            if not source.is_absolute():
                source = manifest_dir / source
            suffix = "".join(source.suffixes)
            destination = output / f"{artifact['id']}{suffix}"
            shutil.copyfile(source, destination)
            _require(
                destination.stat().st_size == artifact["bytes"],
                "copied prediction size mismatch",
            )
            _require(
                sha256_file(destination) == artifact["sha256"],
                "copied prediction hash mismatch",
            )
            record["report_path"] = str(destination.relative_to(output.parent))
        result.append(record)
    _write_json(output / "index.json", {"schema_version": 1, "artifacts": result})
    return result


def _copy_supplemental(manifest: dict, manifest_dir: Path, output: Path) -> list[dict]:
    result = []
    for item in sorted(manifest.get("supplemental_artifacts", []), key=lambda row: row["id"]):
        record = dict(item)
        if item.get("include_in_report"):
            source = Path(item["local_path"])
            if not source.is_absolute():
                source = manifest_dir / source
            suffix = "".join(source.suffixes)
            directory = "figures" if item["kind"] == "figure" else "tables"
            destination = output / directory / f"{item['id']}{suffix}"
            shutil.copyfile(source, destination)
            _require(destination.stat().st_size == item["bytes"], "copied artifact size mismatch")
            _require(sha256_file(destination) == item["sha256"], "copied artifact hash mismatch")
            record["report_path"] = str(destination.relative_to(output))
        result.append(record)
    return result


def _write_metrics(metrics: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(
            metrics,
            key=lambda value: (
                value["comparison_id"],
                value["task"],
                value["output_id"],
                value["metric"],
                value["method"],
            ),
        ):
            writer.writerow({key: row.get(key, "") for key in METRIC_COLUMNS})


def _readme(manifest: dict, report_id: str) -> str:
    state = manifest["state"]
    lines = [
        f"# Presto performance report — {report_id}",
        "",
        state["headline"],
        "",
        f"Status: **{state['status']}**  ",
        f"Generated at: `{manifest['timestamp_utc']}`  ",
        f"Git: `{manifest['git']['commit']}` (dirty: `{str(manifest['git']['dirty']).lower()}`)  ",
        f"Model: `{manifest['model']['id']}`",
        "",
        "## Qualified claims",
        "",
    ]
    lines.extend(f"- {value}" for value in state["qualified_claims"])
    lines.extend(["", "## Blocked claims", ""])
    lines.extend(f"- {value}" for value in state["blocked_claims"])
    lines.extend(["", "## Primary matched-cohort metrics", ""])
    lines.append(
        "| Task / comparison output | Split | Metric | Method / source output | Value | "
        "95% interval | Rows |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|")
    for row in sorted(
        (metric for metric in manifest["metrics"] if metric["primary"]),
        key=lambda metric: (
            metric["task"],
            metric["output_id"],
            metric["metric"],
            metric["method"],
        ),
    ):
        interval = (
            "not available"
            if row["interval_kind"] == "not_available"
            else f"[{row['ci_low']:.6g}, {row['ci_high']:.6g}]"
        )
        method = f"{row['method']} / {row.get('method_output_id', row['output_id'])}"
        lines.append(
            "| {task} / {output_id} | {split} | {metric} | {method_label} | {value:.6g} | "
            "{interval} | {eligible_rows:,} |".format(interval=interval, method_label=method, **row)
        )
    lines.extend(["", "## Promotion gates", ""])
    for gate in manifest["promotion_gates"]:
        lines.append(f"- **{gate['status']} — {gate['id']}**: {gate['reason']}")
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {value}" for value in state["limitations"])
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            "See `report.json`, `data_manifest.json`, `predictions/index.json`, "
            "`tables/metrics.csv`, `reproduce/`, and `checksums.sha256`. Artifact "
            "references are valid only when their recorded byte counts and SHA-256 hashes verify.",
            "",
        ]
    )
    return "\n".join(lines)


def _checksums(root: Path) -> str:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            rows.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    return "\n".join(rows) + "\n"


def generate_report(manifest_path: os.PathLike, output_root: os.PathLike) -> Path:
    """Validate and atomically generate one immutable report directory."""
    manifest_path = Path(manifest_path).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, manifest_path)
    timestamp = _timestamp(manifest["timestamp_utc"])
    report_id = f"{timestamp:%Y-%m-%d_%H%M%S}_{manifest['git']['commit'][:8]}"
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / report_id
    if destination.exists():
        raise FileExistsError(f"report already exists: {destination}")

    temporary = Path(tempfile.mkdtemp(prefix=f".{report_id}.", dir=output_root))
    try:
        for name in ("figures", "tables", "predictions", "reproduce"):
            (temporary / name).mkdir()
        canonical_manifest = _canonical_json(manifest)
        (temporary / "input_manifest.json").write_text(canonical_manifest, encoding="utf-8")
        data_manifest = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": manifest["timestamp_utc"],
            **manifest["data"],
        }
        _write_json(temporary / "data_manifest.json", data_manifest)
        prediction_index = _copy_predictions(
            manifest, manifest_path.parent, temporary / "predictions"
        )
        supplemental = _copy_supplemental(manifest, manifest_path.parent, temporary)
        _write_metrics(manifest["metrics"], temporary / "tables" / "metrics.csv")
        render_comparator_overview(manifest["metrics"], temporary / "figures")
        render_data_composition(manifest["data"]["datasets"], temporary / "figures")

        report = {
            "schema_version": SCHEMA_VERSION,
            "report_engine_version": REPORT_ENGINE_VERSION,
            "report_id": report_id,
            "timestamp_utc": manifest["timestamp_utc"],
            "git": manifest["git"],
            "state": manifest["state"],
            "model": manifest["model"],
            "comparators": manifest["comparators"],
            "metrics": manifest["metrics"],
            "prediction_artifacts": prediction_index,
            "supplemental_artifacts": supplemental,
            "promotion_gates": manifest["promotion_gates"],
            "runtime": manifest.get("runtime", {}),
            "input_manifest_sha256": sha256_file(manifest_path),
            "canonical_manifest_sha256": hashlib.sha256(canonical_manifest.encode()).hexdigest(),
        }
        _write_json(temporary / "report.json", report)
        (temporary / "README.md").write_text(_readme(manifest, report_id), encoding="utf-8")
        _write_json(temporary / "reproduce" / "environment.json", _environment())
        (temporary / "reproduce" / "command.txt").write_text(
            "presto-report --manifest <REPORT_MANIFEST.json> --output-root <REPORTS_DIR>\n",
            encoding="utf-8",
        )
        (temporary / "checksums.sha256").write_text(_checksums(temporary), encoding="utf-8")
        temporary.rename(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    output = generate_report(args.manifest, args.output_root)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
