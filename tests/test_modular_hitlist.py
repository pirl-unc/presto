"""Source semantics and original-outcome conservation for the actual IC50 route."""

from dataclasses import replace
import json

import pytest
import torch

from presto.data.modular import Observation, fit_target_scales, observation_target
from presto.data.modular_hitlist import (
    AdmissionError,
    HumanMHCCatalog,
    MHCReferenceAssumption,
    purified_ic50_observation,
)
from presto.data.modular_preparation import prepare_observations
from presto.inference.modular import ModularPredictor
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import save_checkpoint, train_step


def reference(name="A*02:01", cls="I", chain="alpha", first="A" * 70, second="C" * 70):
    return {
        "two_field_allele": "HomoSapiens-" + name,
        "species": "Homo sapiens",
        "source": "imgt",
        "source_id": name,
        "mhc_class": cls,
        "chain": chain,
        "gene": name.split("*")[0],
        "sequence": "M" + first + second + "G",
        "groove1": first,
        "groove2": second,
        "groove1_len": str(len(first)),
        "groove2_len": str(len(second)),
        "groove_status": "ok",
        "groove_flags": "",
        "is_functional": "True",
        "is_null": "False",
        "is_questionable": "False",
        "is_pseudogene": "False",
    }


def raw(**changes):
    return {
        "source": "iedb",
        "response_measured": "half maximal inhibitory concentration (IC50)",
        "assay_method": "purified MHC/competitive/radioactivity",
        "has_ptm": False,
        "peptide_modifications": None,
        "is_non_peptide_ligand": False,
        "peptide": "ACDEFGHIKL",
        "mhc_class": "I",
        "mhc_restriction": "HLA-A*02:01",
        "measurement_units": "nM",
        "measurement_inequality": "=",
        "quantitative_value": 10.0,
        "quantitative_measurement": "10",
        "assay_iri": "https://example.org/assays/1",
        **changes,
    }


def convert(value, ordinal=1, catalog=None):
    catalog = catalog or HumanMHCCatalog([reference()])
    return purified_ic50_observation(value, catalog, source_sha256="0" * 64, ordinal=ordinal)


@pytest.mark.parametrize("qualifier", [None, "", "=", "<", "<=", ">", ">="])
def test_physical_values_and_original_qualifiers_survive(qualifier):
    value = raw(
        measurement_inequality=qualifier, quantitative_value=100000, quantitative_measurement="1e5"
    )
    row = convert(value)
    assert row.value == 100000
    assert observation_target(row) == 5
    assert row.qualifier == (qualifier or "=")
    assert row.provenance["raw_record"] == value
    assert not row.identity_labels
    assert row.mhc[0].species1 == row.mhc[0].species2 == "human"
    assert not row.group_id  # The adapter cannot choose a split from labels.


@pytest.mark.parametrize(
    "unit,value", [("pM", 10000), ("nM", 10), ("uM", 0.01), ("mM", 1e-5), ("M", 1e-8)]
)
def test_concentration_units_are_preserved_before_canonical_transform(unit, value):
    row = convert(
        raw(measurement_units=unit, quantitative_value=value, quantitative_measurement=str(value))
    )
    assert row.value == value and row.unit == unit
    assert observation_target(row) == pytest.approx(1)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"peptide": float("nan")}, "unsupported_peptide_or_class"),
        ({"quantitative_value": 0}, "nonpositive_or_nonfinite_measurement"),
        ({"quantitative_value": float("inf")}, "nonpositive_or_nonfinite_measurement"),
        ({"measurement_inequality": 0}, "unsupported_measurement_qualifier"),
        ({"measurement_inequality": "~"}, "unsupported_measurement_qualifier"),
        ({"quantitative_measurement": "<10"}, "raw_numeric_or_qualifier_disagreement"),
        ({"quantitative_measurement": "10-20"}, "raw_numeric_or_qualifier_disagreement"),
        ({"quantitative_measurement": "11"}, "raw_numeric_or_qualifier_disagreement"),
        ({"has_ptm": None}, "modified_or_unspecified_peptidoform"),
        ({"peptide_modifications": "phosphorylation"}, "modified_or_unspecified_peptidoform"),
        ({"assay_method": "cellular MHC/binding"}, "not_explicit_purified_mhc"),
        ({"assay_iri": None}, "missing_original_assay_iri"),
        (
            {"response_measured": "dissociation constant KD (~IC50)"},
            "outside_direct_iedb_cedar_ic50",
        ),
    ],
)
def test_unsupported_source_semantics_are_not_reinterpreted(changes, reason):
    with pytest.raises(AdmissionError, match=reason):
        convert(raw(**changes))


def test_matching_bound_text_is_not_an_exact_value():
    row = convert(raw(measurement_inequality=">=", quantitative_measurement=">= 10"))
    assert row.qualifier == ">="


def test_catalog_aliases_do_not_choose_an_arbitrary_assignment():
    original = reference()
    catalog = HumanMHCCatalog([original, {**original, "source_id": "copy"}])
    assert len(catalog.aliases(used_only=False)["HLA-A*02:01"]) == 2
    assert catalog.resolve("HLA-A*02:01", "I").groove1 == original["groove1"]
    ambiguous = HumanMHCCatalog([original, reference(first="D" * 70)])
    with pytest.raises(AdmissionError, match="ambiguous_catalog_name") as error:
        ambiguous.resolve("HLA-A*02:01", "I")
    assert error.value.disposition == "deferred"


def test_class_ii_requires_actual_order_independent_alpha_beta_pair():
    catalog = HumanMHCCatalog(
        [
            reference("DRA*01:01", "II", "alpha", second=""),
            reference("DRB1*01:01", "II", "beta", first=""),
        ]
    )
    row = convert(raw(mhc_class="II", mhc_restriction="HLA-DRB1*01:01/DRA*01:01"), catalog=catalog)
    assert row.mhc[0].groove1 == "A" * 70 and row.mhc[0].groove2 == "C" * 70
    assert row.mhc[0].provenance["partner_imputation"] is False
    with pytest.raises(AdmissionError, match="missing_explicit_mhc_partner"):
        convert(raw(mhc_class="II", mhc_restriction="HLA-DRB1*01:01"), catalog=catalog)
    with pytest.raises(AdmissionError, match="class_ii_not_alpha_beta_pair"):
        convert(raw(mhc_class="II", mhc_restriction="HLA-DRB1*01:01/DRB1*01:01"), catalog=catalog)


@pytest.mark.parametrize("mhc_class", ["I", "II"])
def test_explicit_reference_retains_unknown_typing_through_checkpoint_inference(
    tmp_path, mhc_class
):
    catalog = HumanMHCCatalog(
        [
            reference(),
            reference("DRA*01:01", "II", "alpha", second=""),
            reference("DRB1*01:01", "II", "beta", first=""),
        ]
    )
    reported = None if mhc_class == "I" else "HLA-DRB1*01:01"
    selected = "HLA-A*02:01" if mhc_class == "I" else "HLA-DRA*01:01/DRB1*01:01"
    assumption = MHCReferenceAssumption(selected, "software fixture reference", "fixture:v1")
    pair = catalog.resolve(reported, mhc_class, reference_assumption=assumption)
    assert pair.provenance["reported_restriction"] == reported
    assert pair.provenance["assumed_alleles"] == [
        "HLA-A*02:01" if mhc_class == "I" else "HLA-DRA*01:01"
    ]
    assert pair.provenance["partner_imputation"] is (mhc_class == "II")
    assert pair.provenance["reference_assumption"]["reference"] == "fixture:v1"
    # Strict resolution still fails: explicitly modeling a reference is not typing evidence.
    with pytest.raises(AdmissionError):
        catalog.resolve(reported, mhc_class)
    row = replace(convert(raw()), mhc=(pair,))
    row = Observation.from_dict(json.loads(json.dumps(row.to_dict())))
    torch.manual_seed(101)
    model = ModularPresto(ModularConfig(mhc_classes=(mhc_class,), dropout=0))
    scales = fit_target_scales([row, replace(row, value=100)])
    train_step(model, torch.optim.AdamW(model.parameters(), lr=0.001), [row], scales)
    expected = model.eval().predict([row])[0]
    checkpoint = tmp_path / "reference.pt"
    save_checkpoint(checkpoint, model, scales)
    result = ModularPredictor(checkpoint).predict([replace(row, value=None, split="predict")])[0]
    assert result["diagnostics"]["candidates"][0]["mhc"]["provenance"] == pair.provenance
    assert result["transformed_value"] == scales["ic50"].decode(float(expected.value))
    # Evidence is retained for audit, never encoded as a predictive source identifier.
    alternate = replace(pair, provenance={"source": "other", "assumed_alleles": []})
    torch.testing.assert_close(
        model.predict([replace(row, mhc=(alternate,))])[0].value, expected.value, rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "reported,selected,reason",
    [
        ("HLA-A*03:01", "HLA-A*02:01", "changes_reported_typing"),
        ("HLA-DRB1*02:01", "HLA-DRA*01:01/DRB1*01:01", "changes_reported_typing"),
        ("HLA-DRB1*01:01/DRB1*01:01", "HLA-DRA*01:01/DRB1*01:01", "changes_reported_typing"),
        ("", "HLA-A*02:01", "not_explicit_human"),
        (None, "HLA-A*03:01", "unresolved_catalog_name"),
        (None, "HLA-DRB1*01:01", "missing_explicit_mhc_partner"),
        (None, "HLA-DRB1*01:01/DRB1*01:01", "class_ii_not_alpha_beta_pair"),
    ],
)
def test_reference_assumptions_cannot_bypass_source_or_catalog_checks(reported, selected, reason):
    catalog = HumanMHCCatalog([reference(), reference("DRB1*01:01", "II", "beta", first="")])
    with pytest.raises(AdmissionError, match=reason):
        catalog.resolve(
            reported,
            "II" if "DR" in selected else "I",
            reference_assumption=MHCReferenceAssumption(
                selected, "explicit reference", "fixture:v1"
            ),
        )
    assert not catalog.used


@pytest.mark.parametrize("field", ["restriction", "reason", "reference"])
@pytest.mark.parametrize("value", [None, "", " ", 1])
def test_reference_assumption_requires_explicit_typed_metadata(field, value):
    supplied = dict(restriction="HLA-A*02:01", reason="explicit reference", reference="fixture:v1")
    supplied[field] = value
    with pytest.raises(AdmissionError, match="requires_restriction_reason_reference"):
        MHCReferenceAssumption(**supplied)


def test_untyped_assumption_is_rejected_and_strict_provenance_is_unchanged():
    catalog = HumanMHCCatalog([reference()])
    with pytest.raises(AdmissionError, match="untyped_reference_assumption"):
        catalog.resolve(None, "I", reference_assumption={"restriction": "HLA-A*02:01"})
    assert catalog.resolve("HLA-A*02:01", "I").provenance == {
        "reported_restriction": "HLA-A*02:01",
        "catalog_alleles": ["HLA-A*02:01"],
        "partner_imputation": False,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"groove_flags": "ambiguous"},
        {"is_null": "True"},
        {"groove1_len": "69"},
        {"sequence": "M" * 200},
    ],
)
def test_invalid_reference_cannot_resolve_an_assay(change):
    catalog = HumanMHCCatalog([{**reference(), **change}])
    with pytest.raises(AdmissionError, match="unresolved_catalog_name"):
        convert(raw(), catalog=catalog)


def test_canonical_preparation_retains_independent_disagreement_and_deduplicates_copies(tmp_path):
    values = [
        raw(),
        raw(source="cedar"),  # Same original assay re-exported, not a second vote.
        raw(
            assay_iri="https://example.org/assays/2",
            quantitative_value=1000,
            quantitative_measurement="1000",
        ),
        raw(assay_iri="https://example.org/assays/3"),
        raw(
            assay_iri="https://example.org/assays/3",
            quantitative_value=20,
            quantitative_measurement="20",
        ),
    ]
    rows = [
        replace(convert(value, i), group_id="peptide:ACDEFGHIKL") for i, value in enumerate(values)
    ]
    report = prepare_observations(
        rows,
        tmp_path / "prepared",
        population="test",
        label_policy="raw IC50",
        split_policy="fixed",
    )
    assert report["dispositions"] == {"admitted": 2, "duplicate": 1, "deferred": 2}
    assert report["independent_input_contradictions"] == 1
    admitted = [json.loads(line) for source in report["sources"] for line in open(source["path"])]
    assert {row["value"] for row in admitted} == {10, 1000}
    assert len(admitted[0]["provenance"]["export_aliases"]) == 2
