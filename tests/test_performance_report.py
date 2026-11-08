from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from presto.evaluation.performance_report import generate_report, validate_manifest


def _artifact(path: Path, *, include=False):
    payload = path.read_bytes()
    return {
        "uri": f"artifact://fixture/{path.name}",
        "local_path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "license": "CC0-1.0",
        "include_in_report": include,
    }


def _manifest(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint.pt"
    train = tmp_path / "train.tsv"
    development = tmp_path / "development.tsv"
    predictions = tmp_path / "predictions.tsv"
    comparator = tmp_path / "comparator.json"
    checkpoint.write_bytes(b"frozen-checkpoint\n")
    train.write_text("peptide\tlabel\nAAAA\t1\n")
    development.write_text("peptide\tlabel\nBBBB\t0\nCCCC\t1\n")
    predictions.write_text("id\tpresto\tbaseline\n1\t0.9\t0.7\n2\t0.1\t0.4\n")
    comparator.write_text('{"name":"fixture-baseline","version":"1"}\n')
    metrics = []
    for method, value, low, high in (
        ("Presto", 0.72, 0.68, 0.76),
        ("NetMHCIIpan", 0.78, 0.74, 0.81),
    ):
        metrics.append(
            {
                "comparison_id": "mhcii_dev",
                "cohort_id": "new_mhcii_development",
                "task": "mhc_ii_affinity",
                "output_id": "unspecified_assay_ic50",
                "split": "development",
                "estimate_type": "method_value",
                "method": method,
                "method_output_id": "unspecified_assay_ic50",
                "metric": "spearman",
                "value": value,
                "interval_kind": "bootstrap_95",
                "ci_low": low,
                "ci_high": high,
                "higher_is_better": True,
                "eligible_rows": 200,
                "studies": 5,
                "primary": True,
            }
        )
    return {
        "schema_version": 1,
        "timestamp_utc": "2026-11-08T22:15:00Z",
        "git": {"commit": "5a44a0dd50b249d277f13be13517ce8b439badb4", "dirty": False},
        "state": {
            "headline": "Fixture report for the development-only unified checkpoint.",
            "status": "development",
            "qualified_claims": ["The rows are matched within this fixture comparison."],
            "blocked_claims": ["No independent TEST superiority claim is available."],
            "limitations": ["This is synthetic fixture data."],
        },
        "model": {
            "id": "presto_fixture",
            "checkpoint": _artifact(checkpoint),
            "training_recipe": {"id": "fixture_recipe", "epochs": 1},
        },
        "data": {
            "datasets": [
                {
                    "id": "fixture_train",
                    "role": "train",
                    "rows": 1,
                    "source": "synthetic fixture",
                    "curation_contract": "one exact row",
                    "split_contract": "fixture TRAIN only",
                    "artifact": _artifact(train),
                },
                {
                    "id": "fixture_development",
                    "role": "development",
                    "rows": 2,
                    "source": "synthetic fixture",
                    "curation_contract": "two exact rows",
                    "split_contract": "disjoint fixture development",
                    "artifact": _artifact(development),
                },
            ],
            "test_status": {
                "available": False,
                "reason": "The fixture deliberately exercises a development-only report.",
            },
            "overlap_audits": [{"name": "exact_peptide", "train_evaluation_overlaps": 0}],
        },
        "comparators": [
            {
                "id": "netmhciipan",
                "version": "fixture-1",
                "receipt": _artifact(comparator),
            }
        ],
        "prediction_artifacts": [
            {
                "id": "mhcii_development_predictions",
                "split": "development",
                **_artifact(predictions, include=True),
            }
        ],
        "metrics": metrics,
        "promotion_gates": [
            {
                "id": "mhcii_affinity",
                "status": "fail",
                "reason": "Fixture Presto Spearman is below the comparator.",
            },
            {
                "id": "independent_test",
                "status": "not_evaluated",
                "reason": "No TEST split is present.",
            },
        ],
        "runtime": {"requested_gpu": "H100!", "observed_gpu": "fixture"},
    }


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def test_generate_report_is_complete_and_deterministic(tmp_path):
    manifest = _manifest(tmp_path)
    manifest_path = _write_manifest(tmp_path, manifest)
    first = generate_report(manifest_path, tmp_path / "reports-a")
    second = generate_report(manifest_path, tmp_path / "reports-b")

    assert first.name == "2026-11-08_221500_5a44a0dd"
    expected = {
        "README.md",
        "checksums.sha256",
        "data_manifest.json",
        "figures/comparator_overview.png",
        "figures/comparator_overview.svg",
        "figures/data_composition.png",
        "figures/data_composition.svg",
        "input_manifest.json",
        "predictions/index.json",
        "predictions/mhcii_development_predictions.tsv",
        "report.json",
        "reproduce/command.txt",
        "reproduce/environment.json",
        "tables/metrics.csv",
    }
    assert {str(path.relative_to(first)) for path in first.rglob("*") if path.is_file()} == expected
    for relative in expected:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
    readme = (first / "README.md").read_text()
    assert "No independent TEST superiority claim" in readme
    assert "NetMHCIIpan" in readme


def test_report_refuses_overwrite(tmp_path):
    manifest_path = _write_manifest(tmp_path, _manifest(tmp_path))
    generate_report(manifest_path, tmp_path / "reports")
    with pytest.raises(FileExistsError, match="report already exists"):
        generate_report(manifest_path, tmp_path / "reports")


def test_report_requires_explicit_reason_without_test(tmp_path):
    manifest = _manifest(tmp_path)
    del manifest["data"]["test_status"]["reason"]
    path = _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="development-only reports"):
        validate_manifest(manifest, path)


def test_report_verifies_local_artifact_hashes(tmp_path):
    manifest = _manifest(tmp_path)
    manifest["model"]["checkpoint"]["sha256"] = "0" * 64
    path = _write_manifest(tmp_path, manifest)
    with pytest.raises(ValueError, match="checkpoint.sha256 mismatch"):
        validate_manifest(manifest, path)
