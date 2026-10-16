"""Assay applicability is inspectable provenance, never a new model input."""

from dataclasses import replace
import json

import pytest
import torch

from presto.data.assay_context import ASSAY_CONTEXT_POLICY, assay_context_fields
from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.hitlist_source import _lineage_fields, training_columns
from presto.data.loaders import BindingRecord
from presto.data.source_lineage import generated_record
from presto.training.output_contract import build_output_contract
from presto.training.output_coverage import OutputCoverageCensus
from presto.training.supported_outputs import evaluate_supported_outputs
from test_output_coverage import manifest_for


def sample(method="", evidence="binding", **kwargs):
    return PrestoSample(
        peptide="ACDEFGHIK",
        mhc_a="ACDEFGHIK",
        primary_allele="HLA-A*02:01",
        source_assay_method=method,
        source_evidence_kind=evidence,
        species_context={"species_context_origin": "native_observation"},
        **kwargs,
    )


@pytest.mark.parametrize(
    "method,evidence,state,basis",
    [
        (
            "purified MHC/competitive/fluorescence",
            "binding",
            "not_applicable",
            "explicit_purified_mhc_preparation",
        ),
        (
            "cellular MHC/competitive/radioactivity",
            "binding",
            "applicable",
            "explicit_cellular_mhc_preparation",
        ),
        ("mass spectrometry", "ms", "applicable", "native_elution_evidence"),
        ("", "ms", "applicable", "native_elution_evidence"),
        ("IC50", "binding", "unknown", "unspecified_assay_preparation"),
        ("", "binding", "unknown", "unspecified_assay_preparation"),
        ("not purified MHC", "binding", "unknown", "unspecified_assay_preparation"),
        (
            "purified MHC/direct; cellular MHC/competitive",
            "binding",
            "unknown",
            "conflicting_assay_preparation",
        ),
        ("purified MHC/mass spectrometry", "ms", "unknown", "conflicting_assay_preparation"),
    ],
)
def test_explicit_preparation_and_elution_only(method, evidence, state, basis):
    row = sample(method, evidence)
    values = assay_context_fields(row)
    assert values["processing_context_applicability"] == state
    assert values["processing_context_applicability_basis"] == basis
    assert (
        values["processing_context_availability"]
        == {
            "not_applicable": "not_applicable",
            "applicable": "unresolved",
            "unknown": "applicability_unknown",
        }[state]
    )


@pytest.mark.parametrize(
    "method", ["purified MHC/direct or cellular MHC", "cellular MHC (purified MHC)"]
)
def test_mixed_preparation_prose_is_unknown(method):
    assert (
        assay_context_fields(sample(method))["processing_context_applicability_basis"]
        == "conflicting_assay_preparation"
    )


def test_known_presenter_does_not_override_purified_preparation():
    for method, availability in [
        ("purified MHC/direct", "not_applicable"),
        ("cellular MHC/direct", "resolved"),
    ]:
        row = sample(method)
        row.species_context["presenting_species"] = "Homo sapiens"
        assert assay_context_fields(row)["processing_context_availability"] == availability


def test_native_method_is_preserved_in_all_evidence_projections():
    for evidence in ("binding", "ms"):
        assert {"assay_method", "evidence_kind"} <= set(
            training_columns(evidence, include_flanks=False)
        )
    source = {"assay_method": "purified MHC/direct/fluorescence", "evidence_kind": "binding"}
    values = _lineage_fields(source)
    assert values["source_assay_method"] == source["assay_method"]
    assert values["source_evidence_kind"] == "binding"


def parent(method, identity):
    return BindingRecord(
        peptide="ACDEFGHIK",
        mhc_allele="HLA-A*02:01",
        value=10,
        evidence_row_id=identity,
        source_assay_method=method,
        source_evidence_kind="binding",
    )


def child(parents, index=0):
    return generated_record(
        parent("", ""), parents=parents, kind="peptide_scramble", seed=1, draw_index=index
    )


def test_generated_controls_keep_parent_methods_without_forging_measured_assays():
    parents = [
        parent("purified MHC/competitive/radioactivity", "a"),
        parent("purified MHC/competitive/fluorescence", "b"),
    ]
    first = child(parents)
    assert first.source_assay_method == first.source_evidence_kind == ""
    assert len(json.loads(first.parent_assay_contexts)) == 2
    assert assay_context_fields(first)["processing_context_applicability"] == "not_applicable"
    assert assay_context_fields(first)["assay_context_origin"] == "generated_parent_evidence"
    second = child([first], 1)
    assert second.parent_assay_contexts == first.parent_assay_contexts
    assert assay_context_fields(second) == assay_context_fields(first)


@pytest.mark.parametrize("method", ["", "cellular MHC/direct"])
def test_generated_mixed_or_unknown_preparations_stay_unknown(method):
    row = child([parent("purified MHC/direct", "a"), parent(method, "b")])
    assert assay_context_fields(row)["processing_context_applicability"] == "unknown"


def test_malformed_generated_context_is_rejected():
    row = child([parent("purified MHC/direct", "a")])
    row.parent_assay_contexts = '["purified MHC"]'
    with pytest.raises(ValueError, match="method/evidence-kind"):
        assay_context_fields(row)


def test_inspection_metadata_changes_no_predictive_tensor_or_target():
    row = sample("purified MHC/direct", bind_value=15, binding_assay_type="IC50")
    other = replace(row, source_assay_method="cellular MHC/direct")
    a, b = [PrestoCollator()([s]).to("cpu") for s in (row, other)]
    for key, value in a.model_inputs().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, b.model_inputs()[key], equal_nan=True)
        else:
            assert value == b.model_inputs()[key]
    assert a.bind_target.tolist() == b.bind_target.tolist()
    assert a.source_lineage["source_assay_method"] == ["purified MHC/direct"]
    assert a.source_lineage["processing_context_applicability"] == ["not_applicable"]
    assert b.source_lineage["processing_context_availability"] == ["unresolved"]


def test_coverage_fingerprints_and_facets_record_applicability():
    contract = build_output_contract()
    assert contract.to_dict()["assay_context_policy"] == ASSAY_CONTEXT_POLICY
    hashes = []
    for method in ("purified MHC/direct", "cellular MHC/direct"):
        row = sample(method, sample_id="one")
        with OutputCoverageCensus(contract) as census:
            census.add_batch("train", [row], PrestoCollator()([row]))
            assert (
                census.db.execute(
                    "SELECT value FROM facets WHERE field='source_assay_method'"
                ).fetchone()[0]
                == method
            )
            hashes.append(census.sha256)
    assert hashes[0] != hashes[1]


@pytest.mark.parametrize("policy", [None, "old-policy"])
def test_support_contract_rejects_missing_or_wrong_applicability_policy(policy):
    contract = build_output_contract()
    manifest = manifest_for(contract)
    if policy is None:
        del manifest["source_contract"]["assay_context_policy"]
    else:
        manifest["source_contract"]["assay_context_policy"] = policy
    with OutputCoverageCensus(contract) as census:
        with pytest.raises(ValueError, match="assay context applicability policy"):
            evaluate_supported_outputs(
                census, manifest, source_contract=manifest["source_contract"]
            )
