"""Tests for machine-readable split-support and data-contract gates."""

import json

import pytest

from presto.data.collate import PrestoSample
from presto.training.data_support import (
    _StreamingMultisetHash,
    audit_split_support,
    validate_split_support,
    write_data_funnel_artifacts,
    write_split_support_artifacts,
)


def _sample(sample_id: str, label: float, *, flank: str = "AAAA") -> PrestoSample:
    return PrestoSample(
        peptide="SIINFEKL",
        flank_n=flank,
        flank_c="CCCC",
        elution_label=label,
        sample_source="iedb",
        sample_id=sample_id,
        primary_allele="HLA-A*02:01",
        source_mhc_alleles=("HLA-A*02:01",),
        resolved_mhc_alleles=("HLA-A*02:01",),
        evidence_row_id=f"ms:{sample_id}",
        assay_iri=f"assay:{sample_id}",
        mapping_protein_id="ENSP1",
        mapping_position=10,
        mapping_proteome="Homo sapiens",
        source_mapping_n_candidates=1,
    )


def _balanced_splits(flank: str = "AAAA"):
    return {
        split: [
            _sample(f"{split}-positive", 1.0, flank=flank),
            _sample(f"{split}-negative", 0.0, flank=flank),
        ]
        for split in ("train", "val", "test")
    }


def test_support_audit_counts_targets_classes_and_lineage():
    audit = audit_split_support(_balanced_splits())

    for split in ("train", "val", "test"):
        counts = audit["splits"][split]["targets"]["elution"]
        assert counts["count"] == 2
        assert counts["positive"] == 1
        assert counts["negative"] == 1
    assert audit["lineage"]["issue_count"] == 0
    assert audit["lineage"]["duplicate_sample_id_count"] == 0
    assert audit["fake_null_sequences"]["count"] == 0

    validate_split_support(
        audit,
        require_all_active=True,
        binary_balance_targets=["elution"],
        require_traceable_lineage=True,
        forbid_fake_null_sequences=True,
    )


def test_support_gate_reports_every_missing_and_one_class_condition():
    splits = _balanced_splits()
    splits["val"] = [_sample("val-only-positive", 1.0)]
    audit = audit_split_support(splits)

    with pytest.raises(RuntimeError) as error:
        validate_split_support(
            audit,
            required_targets=["elution", "koff"],
            binary_balance_targets=["elution"],
        )
    message = str(error.value)
    assert "train:koff has 0 examples" in message
    assert "val:elution is one-class" in message
    assert "test:koff has 0 examples" in message


def test_support_gate_can_require_balance_for_every_active_binary_target():
    splits = _balanced_splits()
    splits["val"] = [_sample("val-only-positive", 1.0)]

    with pytest.raises(RuntimeError, match="val:elution is one-class"):
        validate_split_support(
            audit_split_support(splits),
            require_all_active_binary_balance=True,
        )


def test_traceable_lineage_gate_rejects_mapped_sample_without_observation_id():
    splits = _balanced_splits()
    broken = splits["val"][0]
    broken.evidence_row_id = ""

    audit = audit_split_support(splits)

    assert audit["lineage"]["issue_count"] == 1
    assert "missing_evidence_row_id" in audit["lineage"]["issue_examples"][0]
    with pytest.raises(RuntimeError, match="source lineage is incomplete"):
        validate_split_support(audit, require_traceable_lineage=True)


def test_traceable_lineage_gate_rejects_unmapped_source_observation_without_id():
    splits = _balanced_splits()
    broken = splits["val"][0]
    broken.evidence_row_id = ""
    broken.source_mapping_category = "unmapped"
    broken.source_mapping_n_candidates = 0
    broken.mapping_protein_id = ""
    broken.mapping_position = None
    broken.mapping_proteome = ""

    audit = audit_split_support(splits)

    assert audit["lineage"]["issue_count"] == 1
    assert "missing_evidence_row_id" in audit["lineage"]["issue_examples"][0]
    with pytest.raises(RuntimeError, match="source lineage is incomplete"):
        validate_split_support(audit, require_traceable_lineage=True)


def test_traceable_lineage_does_not_require_source_ids_for_generated_or_legacy_rows():
    generated = _sample("generated", 0.0)
    generated.sample_source = "synthetic_negative_binding"
    generated.evidence_row_id = ""
    generated.assay_iri = ""
    generated.source_mapping_category = ""
    generated.source_mapping_n_candidates = 0
    generated.mapping_protein_id = ""
    generated.mapping_position = None
    generated.mapping_proteome = ""
    legacy = _sample("legacy", 1.0)
    legacy.sample_source = "legacy_tsv"
    legacy.evidence_row_id = ""
    legacy.assay_iri = ""
    legacy.source_mapping_category = ""
    legacy.source_mapping_n_candidates = 0
    legacy.mapping_protein_id = ""
    legacy.mapping_position = None
    legacy.mapping_proteome = ""

    audit = audit_split_support({"train": [generated, legacy]})

    assert audit["lineage"]["issue_count"] == 0
    validate_split_support(audit, require_traceable_lineage=True)


def test_streaming_multiset_hash_is_order_independent_and_preserves_multiplicity():
    first = _StreamingMultisetHash()
    second = _StreamingMultisetHash()
    duplicated = _StreamingMultisetHash()
    for value in (b"alpha", b"beta", b"gamma"):
        first.update(value)
    for value in (b"gamma", b"alpha", b"beta"):
        second.update(value)
        duplicated.update(value)
    duplicated.update(b"alpha")

    assert first.hexdigest() == second.hexdigest()
    assert duplicated.hexdigest() != first.hexdigest()
    assert not hasattr(first, "__dict__")


def test_policy_input_changes_full_hash_but_not_supervision_hash():
    legacy_splits = _balanced_splits(flank="NAN")
    masked_splits = _balanced_splits(flank="")
    for samples in [*legacy_splits.values(), *masked_splits.values()]:
        for sample in samples:
            sample.mapping_position = None
            sample.mapping_protein_id = ""
            sample.mapping_proteome = ""
            sample.source_mapping_n_candidates = 0
    legacy = audit_split_support(legacy_splits)
    masked = audit_split_support(masked_splits)

    assert legacy["dataset_contract_sha256"] != masked["dataset_contract_sha256"]
    assert (
        legacy["dataset_supervision_contract_sha256"]
        == masked["dataset_supervision_contract_sha256"]
    )
    with pytest.raises(RuntimeError, match="optional sequence"):
        validate_split_support(legacy, forbid_fake_null_sequences=True)


def test_support_artifacts_round_trip(tmp_path):
    audit = audit_split_support(_balanced_splits())
    paths = write_split_support_artifacts(tmp_path, audit)

    assert json.loads(paths["json"].read_text()) == audit
    csv_text = paths["csv"].read_text()
    assert "train,2,elution,2" in csv_text


def test_data_funnel_artifacts_preserve_stages_and_drop_reasons(tmp_path):
    paths = write_data_funnel_artifacts(
        tmp_path,
        {
            "schema_version": 1,
            "stages": {"before_cap": {"elution": 100}, "after_cap": {"elution": 20}},
            "drop_reasons": {"cap": {"elution": 80}},
            "additions": {"synthetic": {"elution": 10}},
            "diagnostics": {"mhc_resolution": {"resolved_mhcseqs": 92}},
        },
    )

    payload = json.loads(paths["json"].read_text())
    assert payload["stages"]["before_cap"]["elution"] == 100
    assert len(payload["sha256"]) == 64
    assert "drop_reasons,cap,elution,80" in paths["csv"].read_text()
    assert "diagnostics,mhc_resolution,resolved_mhcseqs,92" in paths["csv"].read_text()
