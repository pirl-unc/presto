"""Prove the inventory sees records beyond its retention cap and restores hooks."""

import csv
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "routing_audit", Path(__file__).with_name("launch.py")
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_observes_every_payload_before_retained_cap(tmp_path):
    path = tmp_path / "source.tsv"
    fields = ["peptide", "mhc_allele", "source", "record_type", "value", "value_type", "qualifier"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for value in [42, 84, 126]:
            writer.writerow(
                dict(
                    peptide="ACDEFGHIK",
                    mhc_allele="HLA-A*02:01",
                    source="iedb",
                    record_type="binding",
                    value=value,
                    value_type="IC50",
                    qualifier=0,
                )
            )
    result = audit.inspect_loader(path)
    assert result["stats"]["records_loaded"]["binding"] == 1
    assert result["stats"]["counts_before_cap"]["binding"] == 3
    assert result["observed_pre_cap_records"]["binding"] == 3
    group = result["groups"][0]
    assert group["rows"] == 3
    assert group["emitted"]["binding"]["records"] == 3
    assert len(group["emitted"]["binding"]["ordered_sha256"]) == 64
    # A payload beyond the retained first row must change the observed fingerprint.
    path.write_text(path.read_text().replace("126", "127"))
    changed = audit.inspect_loader(path)["groups"][0]
    assert changed["emitted"] != group["emitted"]


def test_restores_production_hooks_after_loader_failure(tmp_path):
    from presto.scripts import train_iedb as runner

    classify, append = runner.classify_assay_type, runner._append_with_cap_sampling
    with pytest.raises(FileNotFoundError):
        audit.inspect_loader(tmp_path / "missing.tsv")
    assert runner.classify_assay_type is classify
    assert runner._append_with_cap_sampling is append
