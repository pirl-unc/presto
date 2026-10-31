"""Reporting/continuation failure-boundary checks, without training or test access."""

import importlib.util
from pathlib import Path

import pytest

import report

spec = importlib.util.spec_from_file_location(
    "continuation", Path(__file__).with_name("continue.py")
)
continuation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(continuation)


def test_strata_conserve_rows_and_joint_pair():
    rows = []
    for i, (target, prediction, study) in enumerate(
        [(1.0, 1.0, "study-A"), (4.0, 1.0, "study-A"), (1.0, 4.0, None)]
    ):
        rows.append(
            dict(
                peptide=f"fixture-{i}",
                qualifier="=",
                target=target,
                prediction=prediction,
                loss=(target - prediction) ** 2,
                mhc_classes=["II"],
                observation=dict(
                    mhc=[dict(provenance=dict(catalog_alleles=["alpha", "beta"]))],
                    provenance=dict(raw_record=dict(reference_iri=study)),
                ),
            )
        )
    result = report.stratify(rows)
    assert len(result["allele_pair"]) == 1
    assert next(iter(result["allele_pair"].values()))["n"] == 3
    assert result["study"]["unknown"]["n"] == 1
    assert result["overall"]["exact_binding_confusion"] == dict(tp=1, fp=1, fn=1, tn=0)
    assert result["study"]["unknown"]["exact"]["spearman"] is None
    assert result["study"]["study-A"]["exact_regression_supported"] is False
    assert result["class"]["II"]["n"] == 3


def test_failure_receipt_stops_without_waiting(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"
    path.write_text('{"status":"failed","failure":"resource guard"}')
    monkeypatch.setattr(
        continuation.time, "sleep", lambda _: pytest.fail("must not wait after failure")
    )
    with pytest.raises(RuntimeError, match="resource guard"):
        continuation.wait_for_receipt(path, 0)


def test_existing_log_prevents_automatic_retry(tmp_path, monkeypatch):
    path = tmp_path / "attempt.log"
    path.write_text("original failure\n")
    monkeypatch.setattr(continuation, "check_sources", lambda: {})
    monkeypatch.setattr(
        continuation.subprocess, "run", lambda *a, **k: pytest.fail("must not rerun")
    )
    with pytest.raises(FileExistsError):
        continuation.execute("launch.py", "seed43", path)
    assert path.read_text() == "original failure\n"
