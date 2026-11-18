"""Generated targets retain real parents without inventing measured assays."""

import json
from dataclasses import replace

import pytest

from presto.data.collate import PrestoCollator
from presto.data.loaders import BindingRecord, ElutionRecord, PrestoDataset, ProcessingRecord
from presto.data.source_lineage import generated_record, record_lineage_id
from presto.legacy.scripts.train_iedb import (
    augment_binding_records_with_synthetic_negatives,
    augment_elution_records_with_synthetic_negatives,
    augment_processing_records_with_synthetic_negatives,
    cascade_binding_negatives_to_downstream,
)


def lineage(identity):
    return dict(
        evidence_row_id=identity, source_bundle_id="sha256:fixture", source_partition="train"
    )


def binding_inputs():
    return [
        BindingRecord("ACDEFGHIK", "HLA-A*02:01", 20, mhc_class="I", **lineage("binding:1")),
        BindingRecord("LMNPQRSTV", "HLA-B*07:02", 30, mhc_class="I", **lineage("binding:2")),
    ]


def binding_controls(rows, seed=42):
    return augment_binding_records_with_synthetic_negatives(
        rows,
        {},
        negative_ratio=2,
        weak_value_min_nM=10000,
        weak_value_max_nM=50000,
        seed=seed,
        modes=["allele_mismatch"],
    )


def test_binding_mismatch_retains_both_real_parents_and_stable_identity():
    originals = binding_inputs()
    rows, stats = binding_controls(originals)
    controls = rows[len(originals) :]
    assert stats["added"] == len(controls) == 4
    assert len({row.generated_id for row in controls}) == 4
    assert [row.generated_id for row in controls] == [
        row.generated_id for row in binding_controls(originals)[0][2:]
    ]
    for row in controls:
        assert set(json.loads(row.root_observation_ids)) == {"binding:1", "binding:2"}
        assert set(json.loads(row.parent_ids)) == {
            record_lineage_id(source) for source in originals
        }
        assert row.generation_lineage_status == "resolved"
        assert row.source_partition == "train"
        assert not row.evidence_row_id and not row.assay_iri and not row.source_value


def test_promiscuous_measured_binding_pairs_are_not_relabelled_as_controls():
    rows = binding_inputs()
    rows[1] = replace(rows[1], peptide=rows[0].peptide)
    result, stats = binding_controls(rows)
    assert result == rows
    assert stats["n_rejected_measured_pairs"] == 4


@pytest.mark.parametrize(
    "field,value",
    [("source_partition", "val"), ("source_bundle_id", "sha256:other"), ("evidence_row_id", "")],
)
def test_frozen_controls_cannot_hide_missing_or_cross_partition_parents(field, value):
    rows = binding_inputs()
    rows[1] = replace(rows[1], **{field: value})
    with pytest.raises(ValueError, match="parents|parent identity"):
        binding_controls(rows)


def test_one_allele_elution_cannot_copy_a_positive_as_negative():
    source = ElutionRecord("ACDEFGHIK", ["HLA-A*02:01"], True, mhc_class="I", **lineage("ms:1"))
    rows, stats = augment_elution_records_with_synthetic_negatives([source], 3, 42)
    assert len(rows) == 3
    assert stats["n_rejected_known_positive"] == 1
    for row in rows[1:]:
        assert row.peptide != source.peptide
        assert json.loads(row.root_observation_ids) == ["ms:1"]


def test_elution_allele_donor_and_hard_pair_evidence_are_retained():
    positive = ElutionRecord("ACDEFGHIK", ["HLA-A*02:01"], True, mhc_class="I", **lineage("ms:1"))
    negative = ElutionRecord("ACDEFGHIK", ["HLA-B*07:02"], False, mhc_class="I", **lineage("ms:2"))
    rows, _ = augment_elution_records_with_synthetic_negatives([positive, negative], 3, 42)
    hard = [row for row in rows if row.source == "synthetic_negative_hard_pair"]
    assert len(hard) == 1
    assert json.loads(hard[0].root_observation_ids) == ["ms:1", "ms:2"]
    assert len(json.loads(hard[0].parent_ids)) == 2
    conflicting = replace(negative, detected=True, evidence_row_id="ms:3")
    rows, stats = augment_elution_records_with_synthetic_negatives(
        [positive, negative, conflicting], 0, 42
    )
    assert len(rows) == 3
    assert stats["n_rejected_known_positive"] > 0


def test_processing_flank_donors_are_real_parents():
    originals = [
        ProcessingRecord("ACDEFGHIK", "ACDEFG", "HIKLMN", 1.0, **lineage("proc:1")),
        ProcessingRecord("LMNPQRSTV", "QRSTVW", "YACDEF", 1.0, **lineage("proc:2")),
    ]
    rows, _ = augment_processing_records_with_synthetic_negatives(originals, 6, 42)
    shuffled = [row for row in rows[2:] if row.generation_kind.endswith("flank_shuffle")]
    assert any(len(json.loads(row.root_observation_ids)) == 2 for row in shuffled)
    assert all(json.loads(row.root_observation_ids) for row in rows[2:])


def test_cascade_keeps_immediate_generated_parent_and_transitive_roots():
    binding, _ = binding_controls(binding_inputs())
    positive = ElutionRecord("ACDEFGHIK", ["HLA-A*02:01"], True, mhc_class="I", **lineage("ms:1"))
    elution, tcell, stats = cascade_binding_negatives_to_downstream(
        binding, [positive], [], 2, 2, 42
    )
    assert stats["elution_added"] > 0
    assert tcell == []
    immediate = {row.generated_id for row in binding if row.generated_id}
    for row in elution[1:]:
        assert set(json.loads(row.parent_ids)) <= immediate
        assert json.loads(row.root_observation_ids) == ["binding:1", "binding:2"]
        assert row.generated_id not in immediate
        assert not row.evidence_row_id


def test_generated_identity_and_roots_survive_dataset_and_host_batch():
    rows, _ = binding_controls(binding_inputs())
    row = rows[2]
    dataset = PrestoDataset(
        binding_records=[row],
        mhc_sequences={row.mhc_allele: "A" * 181},
        strict_mhc_resolution=False,
    )
    sample = dataset[0]
    assert sample.sample_id == f"bind:{row.generated_id}"
    assert sample.root_observation_ids == row.root_observation_ids
    batch = PrestoCollator()([sample])
    assert batch.source_lineage["generated_id"] == [row.generated_id]
    assert batch.source_lineage["root_observation_ids"] == [row.root_observation_ids]
    assert batch.to("cpu").source_lineage["source_partition"] == ["train"]


def test_legacy_unknown_parents_are_explicit_and_measured_identity_is_not_copied():
    parent = BindingRecord("ACDEFGHIK", "HLA-A*02:01", 20)
    target = replace(parent, value=10000)
    result = generated_record(target, parents=[parent], kind="test", seed=1, draw_index=0)
    assert result.generation_lineage_status == "unresolved"
    assert result.root_observation_ids == "[]"
    with pytest.raises(ValueError, match="measured assay identity"):
        generated_record(
            replace(target, evidence_row_id="measured"),
            parents=[parent],
            kind="test",
            seed=1,
            draw_index=0,
        )
