"""Merged publication metadata survives every supported training/export path."""

import csv
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from presto.data.collate import PrestoCollator
from presto.data.loaders import PrestoDataset
from presto.data.source_lineage import SOURCE_LINEAGE_FIELDS, source_lineage_fields
from presto.models.presto import Presto
from presto.scripts.focused_binding_probe import _load_binding_records_from_merged_tsv
from presto.scripts.train_iedb import (
    load_binding_records_for_alleles_from_merged_tsv,
    load_probe_allele_binding_bootstrap_from_merged_tsv,
    load_records_from_merged_tsv,
)
from presto.scripts.train_synthetic import (
    LOSS_TASK_SPECS,
    _get_batch_mask,
    _get_batch_qual,
    _get_batch_target,
    _resolve_task_prediction,
)
from presto.training.data_support import audit_split_support, validate_split_support
from presto.training.holdout_eval import collect_holdout_predictions, write_holdout_artifacts
from presto.training.output_contract import build_output_contract
from presto.training.output_coverage import OutputCoverageCensus

GROUPS = ("binding", "kinetics", "stability", "processing", "elution", "tcell", "tcr_evidence")
ALLELES = ["HLA-A*02:01", "HLA-A*03:01"]
PUBLICATION = {
    "pmid": "12345678",
    "doi": "10.1234/Test",
    "reference_text": 'A reference, with "quotes"\nand a second line',
}


def row(**kwargs):
    return (
        dict(
            peptide="ACDEFGHIK",
            mhc_allele=ALLELES[0],
            mhc_class="I",
            source="iedb",
            record_type="binding",
            value=100,
            value_type="IC50",
            qualifier=-1,
            **PUBLICATION,
        )
        | kwargs
    )


def write_source(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=sorted({k for r in rows for k in r}), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def load(path, cap=0, sampling="head"):
    return load_records_from_merged_tsv(
        path,
        **{f"max_{name}": cap for name in (*GROUPS[:-1], "vdjdb")},
        cap_sampling=sampling,
        sampling_seed=17,
    )


def all_modalities():
    rows = [row(value_type=measurement) for measurement in ("IC50", "kon", "koff", "t_half", "Tm")]
    rows += [
        row(record_type=kind, response="Positive", value_type="", value="")
        for kind in ("processing", "elution", "tcell", "tcr")
    ]
    # Multi-instance elution observations must retain their source row's metadata.
    rows[6]["mhc_allele_set"] = ";".join(ALLELES)
    for index, value in enumerate(rows):
        value.update(
            evidence_row_id=f"observation:{index}",
            assay_iri=f"assay:{index}",
            reference_iri="reference:shared",
        )
    return rows


def dataset(groups):
    groove_a, groove_b = "ACDEFGHIKLMNPQRS" * 4, "TVWYACDEFGHIKLMN" * 4
    return PrestoDataset(
        **{f"{name}_records": records for name, records in zip(GROUPS, groups, strict=True)},
        mhc_exact_inputs={
            allele: dict(
                allele=allele,
                sequence=groove_a + groove_b,
                groove1=groove_a,
                groove2=groove_b,
                mhc_class="I",
                chain="alpha",
                groove_status="complete",
                source="test",
            )
            for allele in ALLELES
        },
    )


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            equal_tree(left[name], right[name])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


def test_copy_keeps_missingness_and_publication_roles_separate():
    for value in (
        {"pmid": " 12345678 ", "doi": None},
        SimpleNamespace(pmid=" 12345678 ", doi=None),
    ):
        copied = source_lineage_fields(value)
        assert copied == {
            name: "12345678" if name == "pmid" else "" for name in SOURCE_LINEAGE_FIELDS
        }


def test_every_modality_preserves_source_fields_through_records_samples_and_batch(tmp_path):
    source = all_modalities()
    *groups, stats = load(write_source(tmp_path / "all.tsv", source))
    assert [len(records) for records in groups] == [1, 2, 2, 1, 1, 1, 1]
    expected = {r["evidence_row_id"]: source_lineage_fields(r) for r in source}
    for records in groups:
        for record in records:
            assert source_lineage_fields(record) == expected[record.evidence_row_id]
    samples = list(dataset(groups))
    assert len({sample.sample_id for sample in samples}) == len(source)
    assert all(sample.sample_id.endswith(":" + sample.evidence_row_id) for sample in samples)
    assert {sample.pmid for sample in samples} == {PUBLICATION["pmid"]}
    for sample in samples:
        assert source_lineage_fields(sample) == expected[sample.evidence_row_id]
    batch = PrestoCollator()(samples)
    assert batch.mil_instance_to_bag.numel() == 2
    moved = batch.to("cpu")
    for name in SOURCE_LINEAGE_FIELDS:
        assert moved.source_lineage[name] == [getattr(sample, name) for sample in samples]
    assert sum(stats["counts_before_cap"].values()) == len(source)


@pytest.mark.parametrize("sampling", ["head", "reservoir"])
def test_publications_do_not_change_caps_sampling_or_payloads(tmp_path, sampling):
    source = [row(value=i + 1, pmid=str(10000000 + i)) for i in range(20)]
    full = load(write_source(tmp_path / "full.tsv", source), cap=3, sampling=sampling)
    blank = load(
        write_source(
            tmp_path / "blank.tsv",
            [{k: v for k, v in r.items() if k not in SOURCE_LINEAGE_FIELDS} for r in source],
        ),
        cap=3,
        sampling=sampling,
    )
    assert full[-1] == blank[-1]
    assert len(full[0]) == 3
    for a, b in zip(full[0], blank[0], strict=True):
        assert a.pmid == str(10000000 + int(a.value) - 1)
        assert source_lineage_fields(b) == dict.fromkeys(SOURCE_LINEAGE_FIELDS, "")
        assert {k: v for k, v in asdict(a).items() if k not in SOURCE_LINEAGE_FIELDS} == {
            k: v for k, v in asdict(b).items() if k not in SOURCE_LINEAGE_FIELDS
        }
    assert [s.sample_id for s in dataset(full[:-1])] == [s.sample_id for s in dataset(blank[:-1])]


def test_metadata_does_not_change_actual_forward_inputs_targets_or_selectors(tmp_path):
    *groups, _ = load(write_source(tmp_path / "all.tsv", all_modalities()))
    samples = list(dataset(groups))
    blank = [replace(sample, **dict.fromkeys(PUBLICATION, "")) for sample in samples]
    batch, other = PrestoCollator()(samples), PrestoCollator()(blank)
    equal_tree(batch.model_inputs(), other.model_inputs())
    # Includes MIL tensors, masks, selectors, targets and sampler IDs, not just
    # the forward kwargs. Source lineage is the sole changed batch attribute.
    equal_tree(
        {k: v for k, v in vars(batch).items() if k != "source_lineage"},
        {k: v for k, v in vars(other).items() if k != "source_lineage"},
    )


@pytest.mark.parametrize("selector", ["panel", "bootstrap", "focused"])
def test_binding_selectors_preserve_existing_metadata(tmp_path, selector):
    rows = [
        row(mhc_allele=allele, evidence_row_id=f"obs:{index}", assay_iri=f"assay:{index}")
        for index, allele in enumerate(ALLELES)
    ]
    path = write_source(tmp_path / "binding.tsv", rows)
    if selector == "panel":
        records, _ = load_binding_records_for_alleles_from_merged_tsv(path, alleles=ALLELES)
    elif selector == "bootstrap":
        records, _ = load_probe_allele_binding_bootstrap_from_merged_tsv(
            path, probe_alleles=ALLELES, max_records=10, max_peptides=10
        )
    else:
        records, _ = _load_binding_records_from_merged_tsv(path, alleles=ALLELES)
    assert len(records) == len(rows)
    for record, source in zip(records, rows, strict=True):
        assert source_lineage_fields(record) == source_lineage_fields(source)


def test_publication_only_rows_remain_fallback_observations_and_do_not_pass_original_id_gate(
    tmp_path,
):
    *groups, _ = load(write_source(tmp_path / "binding.tsv", [row(), row(value=200)]))
    samples = list(dataset(groups))
    assert [s.sample_id for s in samples] == ["bind_0", "bind_1"]
    assert all(not s.evidence_row_id for s in samples)
    with OutputCoverageCensus(build_output_contract()) as census:
        census.add_batch("train", samples, PrestoCollator()(samples))
        assert (
            census.db.execute("SELECT COUNT(*) FROM samples WHERE traceable=1").fetchone()[0] == 0
        )
        assert census.counts("train", "assays.IC50_nM")["unique_observations"] == 2
    mapped = replace(samples[0], source_mapping_category="unmapped")
    audit = audit_split_support({"train": [mapped]})
    with pytest.raises(RuntimeError, match="source lineage is incomplete"):
        validate_split_support(audit, require_traceable_lineage=True)


def test_row_and_mil_predictions_export_the_source_publication_not_another_rows(tmp_path):
    source = all_modalities()
    # Distinct row metadata makes a wrong source-row join observable.
    for i, r in enumerate(source):
        r["doi"] += f"/{i}"
    *groups, _ = load(write_source(tmp_path / "all.tsv", source))
    samples = list(dataset(groups))
    expected = {s.sample_id: source_lineage_fields(s) for s in samples}
    model = Presto(d_model=32, n_layers=1, n_heads=4).eval()
    batch = PrestoCollator()(samples)
    accumulators = collect_holdout_predictions(
        model,
        [batch],
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
    with (tmp_path / "test_predictions.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {r["observation_kind"] for r in rows} >= {"row", "bag"}
    assert {r["sample_id"] for r in rows} == set(expected)
    for prediction in rows:
        assert {name: prediction[name] for name in SOURCE_LINEAGE_FIELDS} == expected[
            prediction["sample_id"]
        ]
