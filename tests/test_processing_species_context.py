"""Native processing context follows the presenter, independently of host/MHC."""

from dataclasses import replace

import pytest

from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.loaders import BindingRecord, PrestoDataset
from presto.data.source_lineage import generated_record
from presto.data.species_context import native_species_context, processing_species_context
from presto.data.vocab import normalize_single_species, normalize_species
from presto.legacy.presto import _processing_species_idx_tensor
from presto.training.output_contract import build_output_contract
from presto.training.output_coverage import OutputCoverageCensus
from presto.training.supported_outputs import evaluate_supported_outputs
from test_output_coverage import manifest_for


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Homo sapiens (human)", "human"),
        ("Mus musculus", "mouse"),
        ("Rattus norvegicus (brown rat)", "rat"),
        ("Danio rerio (zebrafish)", "zebrafish"),
        ("Homo sapiens;Mus musculus", None),
        ("human (mouse)", None),
        ("human or unknown", None),
        ("unknown", None),
        (None, None),
    ],
)
def test_single_declared_species_rejects_ambiguity(raw, expected):
    assert normalize_single_species(raw) == expected


def test_legacy_parser_contract_is_not_silently_rewritten():
    assert normalize_species("Homo sapiens;Mus musculus") == "human"
    assert normalize_single_species("Homo sapiens;Mus musculus") is None


@pytest.mark.parametrize(
    "presenter,expected,status",
    [
        ("Rattus norvegicus", "Rattus norvegicus", "resolved"),
        ("Homo sapiens", "Homo sapiens", "resolved"),
        ("", None, "unknown"),
        ("human;mouse", None, "unresolved"),
        ("unrecognized", None, "unresolved"),
    ],
)
def test_presenter_controls_row_and_both_mil_paths(presenter, expected, status):
    sample = PrestoSample(
        peptide="ACDEFGHIK",
        mhc_a="ACDEFGHIK",
        mhc_class="I",
        primary_allele="HLA-A*02:01",
        species="Mus musculus",
        elution_label=1,
        tcell_label=1,
        use_tcell_pathway_mil=True,
        mil_mhc_a_list=["ACDEFGHIK", "LMNPQRSTV"],
        tcell_mil_mhc_a_list=["ACDEFGHIK", "LMNPQRSTV"],
        mil_species_list=["human", "murine"],
        tcell_mil_species_list=["human", "murine"],
        species_context=native_species_context(
            {
                "host": "Mus musculus",
                "presenting_species": presenter,
                "in_vivo_host_species": "Mus musculus",
                "source_organism": "Homo sapiens",
                "introduced_mhc_species": "Homo sapiens",
            }
        ),
    )
    batch = PrestoCollator()([sample]).to("cpu")
    assert batch.processing_species == [expected]
    assert batch.mil_species == batch.tcell_mil_species == [expected or ""] * 2
    assert batch.molecular_mhc_species == ["human"]
    assert batch.source_lineage["reported_host_species"] == ["Mus musculus"]
    assert batch.source_lineage["processing_species_status"] == [status]
    row_ids = _processing_species_idx_tensor(batch.model_inputs()["species"], 1, "cpu")
    mil_ids = _processing_species_idx_tensor(batch.mil_species, 2, "cpu")
    assert mil_ids.tolist() == row_ids.tolist() * 2


def test_legacy_explicit_instance_species_are_retained_and_disclosed():
    sample = PrestoSample(
        peptide="ACDEFGHIK",
        species="human",
        elution_label=1,
        mil_mhc_a_list=["ACDEFGHIK", "LMNPQRSTV"],
        mil_species_list=["human", "murine"],
    )
    batch = PrestoCollator()([sample])
    assert batch.processing_species == ["human"]
    assert batch.mil_species == ["human", "murine"]
    assert batch.source_lineage["processing_species_basis"] == ["legacy_explicit_species"]


def test_generated_context_uses_parent_consensus_not_swapped_mhc_or_host():
    parent = BindingRecord(
        peptide="ACDEFGHIK",
        mhc_allele="HLA-A*02:01",
        value=50,
        species="human",
        evidence_row_id="source:1",
        species_context=native_species_context(
            {"host": "human", "presenting_species": "Rattus norvegicus"}
        ),
    )
    other = replace(
        parent,
        evidence_row_id="source:2",
        species_context=native_species_context({"presenting_species": "Homo sapiens"}),
    )
    for parents, expected in [([parent], "Rattus norvegicus"), ([parent, other], "")]:
        derived = generated_record(
            replace(parent, mhc_allele="H2-K*b", species="mouse", evidence_row_id=""),
            parents=parents,
            kind="test",
            seed=1,
            draw_index=0,
        )
        context = processing_species_context(derived)
        assert context["processing_species"] == expected
        assert context["processing_species_basis"] == "generated_parent_consensus"
        assert derived.species_context["parent_processing_species"] == expected
        dataset = PrestoDataset(binding_records=[derived], strict_mhc_resolution=False)
        assert processing_species_context(dataset[0]) == context


def test_unknown_parent_prevents_invented_generated_context():
    parent = BindingRecord(
        peptide="ACDEFGHIK",
        mhc_allele="HLA-A*02:01",
        value=50,
        species="human",
        evidence_row_id="source:1",
        species_context=native_species_context({"host": "human"}),
    )
    other = replace(parent, evidence_row_id="source:2", species_context={})
    derived = generated_record(
        replace(parent, evidence_row_id=""),
        parents=[parent, other],
        kind="test",
        seed=1,
        draw_index=0,
    )
    assert processing_species_context(derived)["processing_species"] == ""
    legacy = generated_record(
        replace(other, evidence_row_id=""), parents=[other], kind="test", seed=1, draw_index=1
    )
    assert processing_species_context(legacy)["processing_species"] == "human"


def test_stale_or_omitted_processing_policy_fails_qualification():
    with OutputCoverageCensus(build_output_contract()) as census:
        for value in (None, "legacy_host_context"):
            manifest = manifest_for(census.contract)
            if value is None:
                del manifest["source_contract"]["processing_species_policy"]
            else:
                manifest["source_contract"]["processing_species_policy"] = value
            with pytest.raises(ValueError, match="current processing species policy"):
                evaluate_supported_outputs(
                    census, manifest, source_contract=manifest["source_contract"]
                )


def test_presenter_changes_support_fingerprint_with_constant_host_and_mhc():
    sample = PrestoSample(
        peptide="ACDEFGHIK", species="human", primary_allele="HLA-A*02:01", sample_id="source-1"
    )
    fingerprints = set()
    for presenter in ("Homo sapiens", "Rattus norvegicus", ""):
        row = replace(
            sample, species_context=native_species_context({"presenting_species": presenter})
        )
        with OutputCoverageCensus(build_output_contract()) as census:
            census.add_batch("train", [row], PrestoCollator()([row]))
            fingerprints.add(census.sha256)
    assert len(fingerprints) == 3
