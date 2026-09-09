"""Selector audit counts actual constructions and restores runner-local hooks."""

import csv
import importlib.util
from pathlib import Path

import pytest

from presto.scripts import train_iedb as runner

spec = importlib.util.spec_from_file_location(
    "selector_audit", Path(__file__).with_name("launch.py")
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def source(path):
    rows = [
        dict(
            peptide="ACDEFGHIK",
            mhc_allele=allele,
            record_type="binding",
            value=100 + repeat,
            value_type="IC50",
            qualifier=-1,
            assay_type="dissociation constant KD (~IC50)",
            assay_method="purified MHC/direct/fluorescence",
            effector_culture_condition="Direct ex vivo",
            apc_culture_condition="cultured",
        )
        for repeat in range(3)
        for allele in audit.ALLELES
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


@pytest.mark.parametrize("selector,constructed,retained", [("panel", 6, 1), ("bootstrap", 2, 2)])
def test_precap_constructions_and_real_bootstrap_selection(
    tmp_path, selector, constructed, retained
):
    original_constructor, original_classifier = runner.BindingRecord, runner.classify_assay_type
    result = audit.inspect_selector(source(tmp_path / "source.tsv"), selector)
    assert result["observed_records"] == constructed
    assert result["retained_records"] == retained
    for row in result["field_counts"]:
        assert row["source_present"] == row["records"] == constructed
        assert row["matched"] + row["dropped"] == constructed
        assert row["invented"] == row["changed"] == 0
    assert result["expected_columns"] == [
        dict(
            assay_type="KD_PROXY_IC50",
            assay_method="PURIFIED_DIRECT_FLUORESCENCE",
            prep="PURIFIED",
            geometry="DIRECT",
            readout="FLUORESCENCE",
            records=constructed,
        )
    ]
    assert runner.BindingRecord is original_constructor
    assert runner.classify_assay_type is original_classifier


def test_failure_restores_hooks(tmp_path, monkeypatch):
    original_classifier = runner.classify_assay_type

    def fail(**kwargs):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(runner, "BindingRecord", fail)
    with pytest.raises(RuntimeError, match="fixture failure"):
        audit.inspect_selector(source(tmp_path / "source.tsv"), "panel")
    assert runner.BindingRecord is fail
    assert runner.classify_assay_type is original_classifier
