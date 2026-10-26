"""Artificial fixtures for source semantics; no biological performance claims."""

import pytest

from presto.data.modular_hitlist import AdmissionError, HumanMHCCatalog
from presto.data.modular_hitlist_ms import (
    class_ii_ms_observation,
    source_condition,
    study_factor_vocabulary,
)


@pytest.fixture
def catalog():
    rows = []
    for name, chain, sequence in (
        ("DPA1*01:01", "alpha", "ACDE" * 20),
        ("DPB1*01:01", "beta", "FGHI" * 20),
        ("DQA1*01:01", "alpha", "KLMN" * 20),
        ("DQB1*01:01", "beta", "PQRS" * 20),
    ):
        rows.append(
            dict(
                source="imgt",
                species="Homo sapiens",
                source_id="fixture:" + name,
                two_field_allele="HomoSapiens-" + name,
                is_functional="True",
                is_null="False",
                is_questionable="False",
                is_pseudogene="False",
                groove_status="ok",
                groove_flags="",
                mhc_class="II",
                chain=chain,
                sequence=sequence,
                groove1=sequence if chain == "alpha" else "",
                groove2=sequence if chain == "beta" else "",
                groove1_len="80" if chain == "alpha" else "",
                groove2_len="80" if chain == "beta" else "",
            )
        )
    return HumanMHCCatalog(rows)


def raw(**changes):
    return (
        dict(
            peptide="ACDEFGHIKLMN",
            mhc_class="II",
            is_ms_observation=True,
            assay_modality="ms",
            evidence_kind="ms",
            ms_outcome_status="positive_result",
            mhc_species="Homo sapiens",
            has_ptm=False,
            peptide_modifications="",
            is_non_peptide_ligand=False,
            evidence_row_id="fixture:observation",
            evidence_source_id="fixture:source",
            provenance_id="fixture:contributors",
            provenance_status="indexed",
            restriction_evidence="unknown",
            mhc_allele_provenance="exact",
            mhc_allele_set="HLA-DPA1*01:01/DPB1*01:01",
            pmid="fixture-study",
            condition_id="condition_a",
            condition_status="annotated",
            condition_evidence="primary_source",
            condition_reference="artificial software fixture",
            sample_attribution="elution_conditions",
            cell_type="fixture_cell",
            condition_knockdown_genes="FACTOR_A",
            condition_drugs="drug_a;drug_b",
            condition_cytokines="cytokine_a",
            condition_transfection="FACTOR_B",
        )
        | changes
    )


def admit(catalog, **changes):
    value = raw(**changes)
    return class_ii_ms_observation(
        value,
        catalog,
        source_sha256="a" * 64,
        ordinal=17,
        study_factors=study_factor_vocabulary([value]),
    )


def test_all_candidates_and_source_identity_survive(catalog):
    names = "HLA-DPA1*01:01/DPB1*01:01;HLA-DQA1*01:01/DQB1*01:01"
    row = admit(catalog, mhc_allele_set=names)
    assert len(row.mhc) == 2
    assert [p.provenance["reported_candidates"][0] for p in row.mhc] == names.split(";")
    assert row.observation_id == "fixture:observation"
    assert row.value == 1 and row.split == "predict" and row.group_id == "unpartitioned"
    assert row.provenance["source_ordinal"] == 17
    assert row.provenance["raw_record"] == raw(mhc_allele_set=names)
    assert row.context.cell_type == "fixture_cell"
    assert set(row.context.interventions) == {
        "knockdown:FACTOR_A",
        "transfection:FACTOR_B",
        "drug:drug_a",
        "drug:drug_b",
    }
    assert row.context.cytokines["cytokine_a"].state == "present"
    assert row.context.processing_pathways == {} and row.context.components == {}


def test_one_partial_candidate_defers_entire_observation(catalog):
    with pytest.raises(AdmissionError, match="missing_explicit_mhc_partner"):
        admit(catalog, mhc_allele_set="HLA-DPA1*01:01/DPB1*01:01;HLA-DRB1*01:01")
    duplicate = admit(catalog, mhc_allele_set="HLA-DPA1*01:01/DPB1*01:01;HLA-DPA1*01:01/DPB1*01:01")
    assert len(duplicate.mhc) == 1


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"restriction_evidence": "predicted"}, "predicted_restriction"),
        ({"mhc_allele_provenance": "pmid_class_pool"}, "unresolved_candidate_scope"),
        ({"has_ptm": None}, "unspecified_peptidoform"),
        ({"has_ptm": True}, "modified_peptidoform"),
        ({"peptide_modifications": "unlocalized modification"}, "modified_peptidoform"),
        ({"ms_outcome_status": "not_detected"}, "not_explicit_positive_ms"),
        ({"evidence_row_id": ""}, "missing_indexed_source_identity"),
        ({"provenance_id": None}, "missing_indexed_source_identity"),
        ({"is_ms_observation": False}, "outside_class_ii_ms"),
        ({"peptide": "ACDEFGHI"}, "unsupported_peptide"),
        ({"mhc_species": "unknown"}, "outside_human_mhc"),
    ],
)
def test_no_inferred_identity_chemistry_pair_or_negative(catalog, changes, reason):
    with pytest.raises(AdmissionError, match=reason):
        admit(catalog, **changes)


def test_pooled_condition_is_retained_as_provenance_only(catalog):
    row = admit(catalog, sample_attribution="pmid_ambiguous")
    assert row.context.interventions == {} and row.context.cytokines == {}
    assert row.context.cell_type is None
    assert row.context.provenance["condition_admitted"] is False
    assert row.provenance["raw_record"]["condition_knockdown_genes"] == "FACTOR_A"
    with pytest.raises(AdmissionError, match="unresolved_candidate_scope"):
        admit(
            catalog,
            sample_attribution="pmid_ambiguous",
            mhc_allele_provenance="sample_allele_match",
        )


def test_only_explicit_none_supplies_absence_within_study():
    treated = raw()
    control = raw(
        condition_id="control", condition_control="non_targeting", condition_knockdown_genes="none"
    )
    unknown = raw(
        condition_id="unknown", condition_control="non_targeting", condition_knockdown_genes=""
    )
    unrelated = raw(pmid="other-study", condition_knockdown_genes="OTHER")
    vocabulary = study_factor_vocabulary([treated, control, unknown, unrelated])
    context = source_condition(control, vocabulary["fixture-study"])
    assert context.interventions["knockdown:FACTOR_A"].state == "absent"
    assert "knockdown:OTHER" not in context.interventions
    assert (
        context.interventions["knockdown:FACTOR_A"].evidence["absence_basis"]
        == "explicit_none_in_same_family"
    )
    assert (
        "knockdown:FACTOR_A"
        not in source_condition(unknown, vocabulary["fixture-study"]).interventions
    )
    with pytest.raises(AdmissionError, match="contradictory_none_condition"):
        study_factor_vocabulary([raw(condition_drugs="none;drug_a")])
