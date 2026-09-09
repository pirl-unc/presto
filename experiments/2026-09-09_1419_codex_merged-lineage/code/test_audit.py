"""Verify audit coverage before caps and restoration of runner-local hooks."""

import csv
import importlib.util
from pathlib import Path

import pytest

from presto.scripts import train_iedb as runner

spec = importlib.util.spec_from_file_location(
    "lineage_audit", Path(__file__).with_name("launch.py")
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def source(path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["peptide", "mhc_allele", "record_type", "value_type", "value", "pmid"],
            delimiter="\t",
        )
        writer.writeheader()
        for value in range(1, 5):
            writer.writerow(
                dict(
                    peptide="ACDEFGHIK",
                    mhc_allele="HLA-A*02:01",
                    record_type="binding",
                    value_type="IC50",
                    value=value,
                    pmid="1" if value < 4 else "",
                )
            )
    return path


def test_all_records_observed_before_cap_and_csv_module_untouched(tmp_path, monkeypatch):
    constructor = runner.BindingRecord
    original_reader = csv.DictReader
    original_append = runner._append_with_cap_sampling

    def controlled_record(**kwargs):
        record = constructor(**kwargs)
        record.pmid = {1: "", 2: "changed", 3: "1", 4: "invented"}[record.value]
        return record

    monkeypatch.setattr(runner, "BindingRecord", controlled_record)
    result = audit.inspect_loader(source(tmp_path / "source.tsv"))
    assert result["input_rows"] == 4
    assert result["stats"]["records_loaded"]["binding"] == 1
    counts = next(
        row
        for row in result["field_counts"]
        if (row["modality"], row["field"]) == ("binding", "pmid")
    )
    assert counts == dict(
        modality="binding",
        field="pmid",
        records=4,
        source_present=3,
        record_present=3,
        matched=1,
        dropped=1,
        invented=1,
        changed=1,
    )
    assert len(result["field_counts"]) == 42
    assert runner.csv is csv
    assert csv.DictReader is original_reader
    assert runner._append_with_cap_sampling is original_append


def test_hooks_restored_after_loader_failure(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(runner, "_append_with_cap_sampling", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        audit.inspect_loader(source(tmp_path / "source.tsv"))
    assert runner.csv is csv
    assert runner._append_with_cap_sampling is fail
