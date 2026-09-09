"""Source assay semantics must survive missing values and unsupported readouts."""

import csv

import pytest

from presto.data.cross_source_dedup import (
    UnifiedRecord,
    _build_elution_cell_hla_lookup,
    assay_measurement_label,
    classify_assay_type,
    write_assay_csvs,
)
from presto.scripts.train_iedb import (
    _record_source_loader_funnel,
    load_binding_records_for_alleles_from_merged_tsv,
    load_records_from_merged_tsv,
)


def record(**kwargs):
    return UnifiedRecord(
        **{
            "peptide": "ACDEFGHIK",
            "mhc_allele": "HLA-A*02:01",
            "record_type": "binding",
            "source": "iedb",
            **kwargs,
        }
    )


@pytest.mark.parametrize("value", [None, 2.7, 500.0])
@pytest.mark.parametrize(
    "measurement,method,bucket",
    [
        ("IC50", "targeted fluorescence", "binding_affinity"),
        ("dissociation constant KD (~EC50)", "purified MHC", "binding_affinity"),
        ("off rate", "purified MHC", "binding_koff"),
        ("dissociation rate", "", "binding_koff"),
        ("association rate", "", "binding_kon"),
        ("half-life", "", "binding_t_half"),
        ("50% dissociation temperature", "", "binding_tm"),
        ("  TM  ", "", "binding_tm"),
        ("qualitative binding", "purified MHC/direct/fluorescence", "binding_qualitative"),
        ("MHC binding", "High throughput multiplexed assay", "binding_qualitative"),
        ("qualitative binding", "mass spectrometry", "binding_qualitative"),
        ("3D structure", "x-ray crystallography", "binding_structure"),
        ("association constant KA", "", "binding_association_constant"),
        ("ka", "", "binding_association_constant"),
        ("unknown fluorescence measure", "radiation", "binding_unknown"),
        ("unknown measurement", "mass spectrometry", "binding_unknown"),
        ("", "radiation immunoassay", "binding_unknown"),
        ("", "targeted fluorescence", "binding_unknown"),
        ("ligand presentation", "Edman degradation", "presentation_non_ms"),
        ("ligand presentation", "T cell recognition", "presentation_non_ms"),
        ("ligand presentation", "coelution", "presentation_non_ms"),
        ("ligand presentation", "", "presentation_unknown_method"),
        ("ligand presentation", "cellular MHC/mass spectrometry", "elution_ms"),
        ("ligand presentation", "DIA", "elution_ms_dia"),
        ("ligand presentation", "LC-MS/MS DDA", "elution_ms_dda"),
        ("ligand presentation", "parallel reaction monitoring", "elution_ms_targeted"),
        ("", "mass spectrometry", "elution_ms"),
        ("TAP transport", "", "processing"),
    ],
)
def test_measurement_semantics_do_not_depend_on_scalar(value, measurement, method, bucket):
    assert (
        classify_assay_type(record(value=value, value_type=measurement, assay_method=method))
        == bucket
    )


def test_measurement_fallback_preserves_selected_label_and_explicit_type_priority():
    example = record(assay_type=" KD ", assay_method="purified MHC", value=42)
    assert classify_assay_type(example) == "binding_affinity"
    assert assay_measurement_label(example) == "KD"
    assert (
        classify_assay_type(record(value_type="3D structure", assay_type="KD", value=2.7))
        == "binding_structure"
    )
    assert classify_assay_type(record(assay_method="dissociation rate", value=42)) == "binding_koff"
    assert (
        classify_assay_type(record(record_type="elution", assay_method="DIA")) == "elution_ms_dia"
    )


def write_source(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def load_source(path, **kwargs):
    caps = {
        name: 0
        for name in (
            "max_binding",
            "max_kinetics",
            "max_stability",
            "max_processing",
            "max_elution",
            "max_tcell",
            "max_vdjdb",
        )
    }
    return load_records_from_merged_tsv(path, **{**caps, **kwargs})


def source_row(**kwargs):
    return {
        "peptide": "ACDEFGHIK",
        "mhc_allele": "HLA-A*02:01",
        "source": "iedb",
        "record_type": "binding",
        "qualifier": 0,
        **kwargs,
    }


def test_loader_partitions_unsupported_and_missing_labels_without_false_targets(tmp_path):
    path = tmp_path / "merged.tsv"
    rows = [
        source_row(value=42, value_type="IC50", qualifier=-1),
        source_row(value=7, assay_type="KD", assay_method="purified MHC", qualifier=1),
        source_row(value_type="IC50", value=""),
        source_row(value_type="qualitative binding", response="Positive", value=25),
        source_row(value_type="MHC binding", response="Negative"),
        source_row(value_type="3D structure", value=2.7),
        source_row(value_type="association constant KA", value=10000),
        source_row(
            value_type="ligand presentation", assay_method="Edman degradation", response="Positive"
        ),
        source_row(value_type="ligand presentation", assay_method="LC-MS/MS", response="Negative"),
        source_row(value_type="off rate", value=0.25, qualifier=1),
        source_row(value_type="half life", value=2, qualifier=-1),
        source_row(value_type="Tm", value=50, qualifier=1),
        source_row(value_type="unknown scalar", value=7),
    ]
    write_source(path, rows)
    binding, kinetics, stability, processing, elution, tcell, tcr, stats = load_source(path)
    assert [(r.value, r.unit, r.measurement_type, r.qualifier) for r in binding] == [
        (42.0, "nM", "IC50", -1),
        (7.0, "nM", "KD", 1),
    ]
    assert len(kinetics) == 1 and kinetics[0].koff == 0.25 and kinetics[0].koff_qualifier == 1
    assert [(r.t_half, r.tm, r.t_half_qualifier, r.tm_qualifier) for r in stability] == [
        (2, None, -1, 0),
        (None, 50, 0, 1),
    ]
    assert len(elution) == 1 and elution[0].detected is False
    assert not processing and not tcell and not tcr
    assert stats["skipped_by_reason"] == {
        "missing_binding_affinity_value": 1,
        "unsupported_binding_qualitative": 2,
        "unsupported_binding_structure": 1,
        "unsupported_binding_association_constant": 1,
        "unsupported_presentation_non_ms": 1,
        "unsupported_binding_unknown": 1,
    }
    assert sum(stats["counts_before_cap"].values()) + sum(
        stats["skipped_by_reason"].values()
    ) == len(rows)
    assert stats["skipped_unroutable_or_missing_label"] == 7
    funnel = {"stages": {}, "drop_reasons": {}}
    _record_source_loader_funnel(funnel, stats)
    skips = funnel["drop_reasons"]["source_ingest"]
    assert "skipped_unroutable_or_missing_label" not in skips
    assert sum(skips.values()) == 7
    # Panel selectors obey the same inclusion and measurement fallback policy.
    selected, _ = load_binding_records_for_alleles_from_merged_tsv(path, alleles=["HLA-A*02:01"])
    assert [(r.value, r.measurement_type) for r in selected] == [(42, "IC50"), (7, "KD")]
    from presto.scripts.focused_binding_probe import _load_binding_records_from_merged_tsv

    focused, _ = _load_binding_records_from_merged_tsv(path, alleles=["HLA-A*02:01"])
    assert [(r.value, r.measurement_type) for r in focused] == [(42, "IC50"), (7, "KD")]


def test_required_label_and_context_skip_reasons_are_disjoint(tmp_path):
    path = tmp_path / "merged.tsv"
    rows = [
        source_row(value_type=label) for label in ("IC50", "on rate", "off rate", "half life", "Tm")
    ]
    rows += [
        source_row(record_type="elution", mhc_allele=""),
        source_row(record_type="tcell", response="unknown"),
        source_row(record_type="tcr", mhc_allele=""),
        source_row(value_type="ligand presentation"),
        source_row(record_type="bcell"),
        source_row(peptide="invalid?", value_type="IC50", value=4),
    ]
    write_source(path, rows)
    *records, stats = load_source(path)
    assert all(not group for group in records)
    assert stats["skipped_by_reason"] == {
        "missing_binding_affinity_value": 1,
        "missing_binding_kon_value": 1,
        "missing_binding_koff_value": 1,
        "missing_binding_t_half_value": 1,
        "missing_binding_tm_value": 1,
        "missing_elution_alleles": 1,
        "missing_tcell_response": 1,
        "missing_tcr_allele": 1,
        "unsupported_presentation_unknown_method": 1,
        "unsupported_bcell_response": 1,
    }
    assert stats["rows_dropped_invalid_peptide"] == 1
    assert stats["skipped_unroutable_or_missing_label"] == 10


@pytest.mark.parametrize("sampling", ["head", "reservoir"])
def test_skip_counts_remain_precap_with_retained_caps(tmp_path, sampling):
    path = tmp_path / "merged.tsv"
    rows = [source_row(value_type="IC50", value=i + 1) for i in range(10)]
    rows += [source_row(value_type="qualitative binding", response="Positive") for _ in range(5)]
    write_source(path, rows)
    *_, stats = load_source(path, max_binding=2, cap_sampling=sampling, sampling_seed=17)
    assert stats["counts_before_cap"]["binding"] == 10
    assert stats["records_loaded"]["binding"] == 2
    assert stats["rows_dropped_by_cap"]["binding"] == 8
    assert stats["skipped_by_reason"] == {"unsupported_binding_qualitative": 5}


def test_unsupported_rows_remain_exportable_and_cannot_establish_elution_hla(tmp_path):
    examples = [
        record(value_type="qualitative binding", response="Positive", pmid="123", apc_name="PBMC"),
        record(value_type="3D structure", value=2.7, pmid="123"),
    ]
    lookup, stats = _build_elution_cell_hla_lookup(examples)
    assert lookup == {} and stats.get("elution_rows_total", 0) == 0
    paths = write_assay_csvs(examples, tmp_path)
    assert set(paths) == {"binding_qualitative", "binding_structure"}
    with open(paths["binding_structure"], newline="") as handle:
        exported = list(csv.DictReader(handle))
    assert len(exported) == 1 and exported[0]["value"] == "2.7" and exported[0]["pmid"] == "123"
