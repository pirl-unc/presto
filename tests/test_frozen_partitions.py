"""Real frozen artifacts, actual model membership and forbidden parent edges."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest

from presto.data.frozen_hitlist import load_frozen_hitlist_bundle, _sha256
from presto.data.frozen_partitions import (
    _components,
    prepare_frozen_partitions,
    augment_within_partitions,
    finalize_frozen_membership,
)
from presto.data.hitlist_source import load_records_from_hitlist
from presto.data.loaders import PrestoDataset, BindingRecord
from presto.data.source_lineage import generated_record
from presto.scripts.train_iedb import (
    augment_binding_records_with_synthetic_negatives,
    cascade_binding_negatives_to_downstream,
)
from test_frozen_hitlist import frozen_bundle as _frozen_bundle

frozen_bundle = _frozen_bundle


def source_groups(directory):
    loaded = load_records_from_hitlist(bundle_dir=directory, include_flanks=False)
    return dict(
        zip(
            (
                "binding_records",
                "kinetics_records",
                "stability_records",
                "processing_records",
                "elution_records",
                "tcell_records",
                "tcr_evidence_records",
            ),
            loaded[:-1],
            strict=True,
        )
    )


def prepare(directory, **kwargs):
    bundle = load_frozen_hitlist_bundle(directory)
    groups = source_groups(directory)
    partitions = prepare_frozen_partitions(
        bundle, groups, seed=17, val_frac=0.2, test_frac=0.2, **kwargs
    )
    return partitions, groups


def model_dataset(groups):
    return PrestoDataset(
        **groups,
        mhc_sequences={"HLA-A*02:01": "ACDEFGHIKLMNPQRSTVWY" * 10},
        strict_mhc_resolution=False,
    )


def test_insufficient_independent_components_fail(frozen_bundle):
    with pytest.raises(ValueError, match="Too few independent"):
        prepare(frozen_bundle)


@pytest.mark.parametrize("frozen_bundle", ["three_peptides"], indirect=True)
def test_reproducible_assignments_and_actual_membership(frozen_bundle):
    partitions, groups = prepare(frozen_bundle)
    assert partitions.receipt["coverage"]["n_selected_observations"] == 9
    assert partitions.receipt["coverage"]["n_components"] == 3
    assert partitions.assignments.groupby("peptide").source_partition.nunique().eq(1).all()
    other, _ = prepare(frozen_bundle)
    assert other.receipt == partitions.receipt
    assert pd.read_csv(partitions.paths["assignments.tsv"], sep="\t").shape[0] == 9
    actual = model_dataset(groups)
    indices, receipt = finalize_frozen_membership(partitions, actual, groups)
    assert {key: len(value) for key, value in indices.items()} == dict(train=3, val=3, test=3)
    from pathlib import Path

    root = Path(receipt["directory"])
    rows = pd.read_parquet(root / "membership.parquet")
    assert rows.sample_id.tolist() == [sample.sample_id for sample in actual.samples]
    assert rows.evidence_row_id.nunique() == 9
    for name in ("root_lineage_audit.json", "model_peptide_audit.json"):
        assert json.loads((root / name).read_text())["verdict"] == "pass"
    for name, digest in receipt["artifacts"].items():
        assert _sha256(root / name) == digest["sha256"]


@pytest.mark.parametrize("frozen_bundle", ["three_peptides"], indirect=True)
def test_external_assignment_pin_and_source_context(frozen_bundle, tmp_path):
    original, groups = prepare(frozen_bundle)
    external = tmp_path / "assignments.parquet"
    original.assignments.to_parquet(external, index=False)
    restored, _ = prepare(
        frozen_bundle, assignment_file=external, assignment_sha256=_sha256(external)
    )
    assert restored.assignments.equals(original.assignments)
    with pytest.raises(ValueError, match="both a file"):
        prepare(frozen_bundle, assignment_file=external)
    with pytest.raises(ValueError, match="[Cc]hecksum|SHA|digest"):
        prepare(frozen_bundle, assignment_file=external, assignment_sha256="0" * 64)
    bad = original.assignments.copy()
    bad.loc[0, "source_partition"] = next(
        x for x in ("train", "val", "test") if x != bad.loc[0, "source_partition"]
    )
    bad.to_parquet(external, index=False)
    with pytest.raises(ValueError, match="split reused"):
        prepare(frozen_bundle, assignment_file=external, assignment_sha256=_sha256(external))
    groups["binding_records"][0].peptide = "VVVVVVVVV"
    with pytest.raises(ValueError, match="frozen observation identity"):
        prepare_frozen_partitions(original.bundle, groups, seed=17, val_frac=0.2, test_frac=0.2)


@pytest.mark.parametrize("frozen_bundle", ["three_peptides"], indirect=True)
def test_unknown_required_identity_fails_with_a_saved_report(frozen_bundle):
    with pytest.raises(ValueError, match="inconclusive under donor_disjoint; audit:") as exc:
        prepare(frozen_bundle, policy="donor_disjoint")
    from pathlib import Path

    report = json.loads(Path(str(exc.value).split("audit: ")[1]).read_text())
    assert report["verdict"] == "inconclusive"
    assert report["partitions"]["train"]["coverage"]["donor"]["n_unresolved_observations"] == 3


def test_shared_contributors_origins_and_aliases_form_one_component():
    registry = {
        "schema_version": 1,
        "entities": [
            dict(
                identity_id=name,
                kind="experimental_origin",
                namespace="fixture",
                reported_id=name,
                evidence_reference="fixture",
                evidence_note="reviewed",
            )
            for name in ("origin:a", "origin:b")
        ],
        "aliases": [
            dict(
                identity_id="origin:b",
                canonical_id="origin:a",
                evidence_reference="fixture",
                evidence_note="same deposited origin",
            )
        ],
    }
    identities = pd.DataFrame(
        [
            dict(
                evidence_row_id=str(i),
                peptide=str(i),
                provenance_id=str(i),
                experimental_origin_ids=json.dumps(["origin:a" if i == 0 else "origin:b"])
                if i in (0, 1)
                else "[]",
            )
            for i in range(4)
        ]
    )
    contributors = pd.DataFrame(
        [
            dict(provenance_id=str(i), source_dataset="iedb", source_record_id="shared")
            for i in (1, 2)
        ]
    )
    assert sorted(_components(identities, contributors, registry, "peptide_disjoint")) == [
        ["0", "1", "2"],
        ["3"],
    ]


@pytest.mark.parametrize("frozen_bundle", ["three_peptides"], indirect=True)
def test_local_generators_nested_parents_and_dropped_model_rows(frozen_bundle):
    partitions, groups = prepare(frozen_bundle)
    groups["binding_records"], stats = augment_within_partitions(
        augment_binding_records_with_synthetic_negatives,
        record_groups={"binding_records": groups["binding_records"]},
        enabled=True,
        mhc_sequences={"HLA-A*02:01": "ACDEFGHIKLMNPQRSTVWY" * 10},
        negative_ratio=1,
        weak_value_min_nM=20000,
        weak_value_max_nM=50000,
        seed=42,
        class_i_no_mhc_beta_ratio=0,
    )
    assert stats["added"] > 0
    groups["elution_records"], groups["tcell_records"], _ = augment_within_partitions(
        cascade_binding_negatives_to_downstream,
        record_groups={
            name: groups[name] for name in ("binding_records", "elution_records", "tcell_records")
        },
        enabled=True,
        elution_ratio=1,
        tcell_ratio=1,
        seed=42,
    )
    model = model_dataset(groups)
    # Keep a descendant while dropping its generated binding parent from the model.
    dropped = next(s for s in model.samples if s.generated_id and s.sample_id.startswith("bind:"))
    model.samples.remove(dropped)
    indices, receipt = finalize_frozen_membership(partitions, model, groups)
    assert receipt["coverage"]["n_records_without_model_sample"] == 1
    assert sum(map(len, indices.values())) == len(model)
    for sample in model.samples:
        if sample.generated_id:
            roots = json.loads(sample.root_observation_ids)
            assert roots and sample.generation_lineage_status == "resolved"
            assert not sample.evidence_row_id and not sample.assay_iri


@pytest.mark.parametrize("frozen_bundle", ["three_peptides"], indirect=True)
def test_forged_edges_and_actual_generated_sequence_overlap_fail(frozen_bundle):
    partitions, groups = prepare(frozen_bundle)
    a, b = groups["binding_records"][:2]
    generated = generated_record(
        BindingRecord(peptide=b.peptide, mhc_allele=a.mhc_allele, value=40000),
        parents=[a],
        kind="test-collision",
        seed=42,
        draw_index=0,
    )
    groups["binding_records"].append(generated)
    with pytest.raises(ValueError, match="failed qualification; audits:"):
        finalize_frozen_membership(partitions, model_dataset(groups), groups)
    forged = deepcopy(groups)
    forged["binding_records"][-1].root_observation_ids = json.dumps([b.evidence_row_id])
    with pytest.raises(ValueError, match="root observations disagree"):
        finalize_frozen_membership(partitions, model_dataset(forged), forged)
    generated.parent_ids = '["missing"]'
    with pytest.raises(ValueError, match="Missing or cyclic"):
        finalize_frozen_membership(partitions, model_dataset(groups), groups)


def test_partition_wrapper_does_not_supply_cross_partition_donors():
    rows = [SimpleNamespace(source_partition=partition) for partition in ("train", "val")]
    seen = []

    def transform(*, records):
        seen.append({r.source_partition for r in records})
        return records, {"added": 0}

    result, stats = augment_within_partitions(
        transform, record_groups={"records": rows}, enabled=True
    )
    assert result == rows and stats == {"added": 0}
    assert seen == [{"train"}, {"val"}]


@pytest.mark.parametrize("frozen_bundle", ["three_peptides"], indirect=True)
def test_prediction_exports_keep_generated_parents_and_partition(frozen_bundle, tmp_path):
    import csv
    from presto.data.collate import PrestoCollator
    from presto.data.source_lineage import source_lineage_fields
    from presto.models.presto import Presto
    from presto.training.holdout_eval import collect_holdout_predictions, write_holdout_artifacts
    from presto.scripts.train_synthetic import (
        LOSS_TASK_SPECS,
        _resolve_task_prediction,
        _get_batch_target,
        _get_batch_mask,
        _get_batch_qual,
    )

    _, groups = prepare(frozen_bundle)
    parent = groups["binding_records"][0]
    groups["binding_records"].append(
        generated_record(
            BindingRecord(
                peptide="VVVVVVVVV",
                mhc_allele=parent.mhc_allele,
                value=40000,
                source="synthetic_peptide_random",
            ),
            parents=[parent],
            kind="peptide_random",
            seed=42,
            draw_index=0,
        )
    )
    samples = model_dataset(groups).samples
    expected = {sample.sample_id: source_lineage_fields(sample) for sample in samples}
    model = Presto(d_model=32, n_layers=1, n_heads=4).eval()
    accumulators = collect_holdout_predictions(
        model,
        [PrestoCollator()(samples)],
        "cpu",
        LOSS_TASK_SPECS,
        forward_fn=lambda m, b: m(**b.model_inputs()),
        resolve_pred_fn=_resolve_task_prediction,
        get_target_fn=_get_batch_target,
        get_mask_fn=_get_batch_mask,
        get_qual_fn=_get_batch_qual,
        mil_chunk_size=1,
    )
    write_holdout_artifacts(tmp_path, accumulators, split="test")
    with (tmp_path / "test_predictions.csv").open() as handle:
        predictions = list(csv.DictReader(handle))
    assert {row["sample_id"] for row in predictions} == set(expected)
    assert any(row["generated_id"] and row["parent_ids"] for row in predictions)
    for row in predictions:
        for name, value in expected[row["sample_id"]].items():
            assert row[name] == value
