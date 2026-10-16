"""Molecular MHC labels and independent native species roles (#115)."""

import csv
from dataclasses import replace

import pytest
import torch

from presto.data.allele_resolver import PROCESSING_SPECIES_TO_IDX
from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.hitlist_source import load_records_from_hitlist
from presto.data.loaders import BindingRecord, PrestoDataset
from presto.data.source_lineage import generated_record
from presto.data.species_context import (
    MOLECULAR_SPECIES_POLICY,
    molecular_mhc_species,
    native_species_context,
)
from presto.training.holdout_eval import collect_holdout_predictions, write_holdout_artifacts
from presto.training.output_contract import build_output_contract
from presto.training.output_coverage import OutputCoverageCensus
from presto.training.supervision import (
    ROW_TASK_SPECS,
    _get_batch_mask,
    _get_batch_qual,
    _get_batch_target,
    _resolve_task_prediction,
    resolve_row_targets,
)
from test_hitlist_source import _binding_row, _install_stub_hitlist, _ms_row, TEST_MHC_SEQUENCE


@pytest.mark.parametrize(
    "alleles,bucket,status",
    [
        (["HLA-B*27:05"], "human", "resolved"),
        (["HLA-A*02:01", "HLA-B*07:02"], "human", "resolved"),
        (["H2-K*b"], "murine", "resolved"),
        (["HLA-A*02:01", "H2-K*b"], "", "mixed"),
        (["HLA-A*02:01", "not-an-allele"], "", "unresolved"),
        ([], "", "unknown"),
    ],
)
def test_molecular_identity_consensus(alleles, bucket, status):
    result = molecular_mhc_species(alleles)
    assert result["molecular_mhc_species_bucket"] == bucket
    assert result["molecular_mhc_species_status"] == status


def test_host_never_supplies_an_unknown_mhc_label():
    row = PrestoSample(peptide="ACDEFGHIK", mhc_a="ACDEFGHIK", species="human")
    batch = PrestoCollator()([row])
    target = resolve_row_targets(batch)["mhc_species"]
    assert target.mask.tolist() == [0]
    assert batch.processing_species == ["human"]
    assert batch.molecular_mhc_species == [None]


def test_old_or_omitted_policy_cannot_pass_a_new_qualification():
    from presto.training.supported_outputs import evaluate_supported_outputs
    from test_output_coverage import manifest_for

    for policy in (None, "processing_host_species_v0"):
        with OutputCoverageCensus(build_output_contract()) as census:
            manifest = manifest_for(census.contract)
            if policy is None:
                del manifest["source_contract"]["molecular_mhc_species_policy"]
            else:
                manifest["source_contract"]["molecular_mhc_species_policy"] = policy
            with pytest.raises(ValueError, match="molecular MHC species policy"):
                evaluate_supported_outputs(
                    census, manifest, source_contract=manifest["source_contract"]
                )


class SpeciesAndElutionModel(torch.nn.Module):
    def forward(self, **inputs):
        n_samples = len(inputs["pep_tok"])
        return {
            "mhc_species_logits": torch.zeros(n_samples, 6),
            "elution_logit": torch.zeros(n_samples, 1),
            "presentation_logit": torch.zeros(n_samples, 1),
            "ms_logit": torch.zeros(n_samples, 1),
        }


def test_rat_host_hla_roles_survive_adapter_device_mil_and_predictions(monkeypatch, tmp_path):
    fields = {
        "host": "Rattus norvegicus (brown rat)",
        "source_organism": "Homo sapiens",
        "mhc_species": "Homo sapiens",
        "mhc_species_source": "allele",
        "species_context_status": "partial",
        "species_context_reference": "https://doi.org/10.1002/pmic.201700249",
        "presenting_species": "Rattus norvegicus",
        "in_vivo_host_species": "Rattus norvegicus",
        "introduced_mhc_species": "Homo sapiens",
        "species_context_note": "A human transgene does not establish peptide source.",
    }
    _install_stub_hitlist(monkeypatch, [_binding_row(**fields)], [_ms_row(**fields)])
    loaded = load_records_from_hitlist(include_flanks=False)
    dataset = PrestoDataset(
        binding_records=loaded[0],
        elution_records=loaded[4],
        mhc_sequences={name: TEST_MHC_SEQUENCE for name in ("HLA-A*02:01", "HLA-B*07:02")},
        strict_mhc_resolution=False,
    )
    assert len(dataset) == 2
    for row in dataset:
        assert row.species_context["reported_host_species"] == fields["host"]
        assert row.species_context["presenting_species"] == fields["presenting_species"]
        assert row.species_context["reported_antigen_species"] == "Homo sapiens"
    batch = PrestoCollator()(list(dataset)).to("cpu")
    assert batch.molecular_mhc_species == ["human", "human"]
    assert batch.processing_species == [fields["presenting_species"]] * 2
    target = resolve_row_targets(batch)["mhc_species"]
    assert target.target.tolist() == [PROCESSING_SPECIES_TO_IDX["human"]] * 2
    assert target.mask.tolist() == [1, 1]
    assert "species_context" not in batch.model_inputs()
    assert "molecular_mhc_species" not in batch.model_inputs()
    assert batch.mil_species == [fields["presenting_species"]] * 2
    predictions = collect_holdout_predictions(
        SpeciesAndElutionModel(),
        [batch],
        "cpu",
        ROW_TASK_SPECS,
        forward_fn=lambda model, moved: model(**moved.model_inputs()),
        resolve_pred_fn=_resolve_task_prediction,
        get_target_fn=_get_batch_target,
        get_mask_fn=_get_batch_mask,
        get_qual_fn=_get_batch_qual,
        mil_chunk_size=1,
    )
    write_holdout_artifacts(tmp_path, predictions)
    rows = list(csv.DictReader((tmp_path / "val_predictions.csv").open()))
    assert any(row["observation_kind"] == "bag" for row in rows)
    assert any(row["task"] == "mhc_species" for row in rows)
    for row in rows:
        assert row["reported_host_species"] == fields["host"]
        assert row["presenting_species"] == fields["presenting_species"]
        assert row["introduced_mhc_species"] == "Homo sapiens"
        assert row["molecular_mhc_species_bucket"] == "human"
        assert row["species_context_reference"] == fields["species_context_reference"]
        assert row["processing_species"] == fields["presenting_species"]
        assert row["processing_species_bucket"] == "murine"
        assert row["processing_species_basis"] == "native_presenting_species"
        assert row["processing_species_status"] == "resolved"


def test_generated_context_preserves_consensus_without_relabeling_molecular_identity():
    parent = BindingRecord(
        peptide="ACDEFGHIK",
        mhc_allele="HLA-A*02:01",
        value=50,
        evidence_row_id="binding:1",
        species_context=native_species_context({"host": "rat", "mhc_species": "Homo sapiens"}),
    )
    generated = generated_record(
        BindingRecord(peptide="LMNPQRSTV", mhc_allele="H2-K*b", value=50000),
        parents=[parent],
        kind="test",
        seed=3,
        draw_index=0,
    )
    assert generated.species_context["reported_host_species"] == "rat"
    assert generated.species_context["reported_mhc_species"] == "Homo sapiens"
    assert generated.species_context["species_context_origin"] == "generated_parent_consensus"
    assert molecular_mhc_species([generated.mhc_allele])["molecular_mhc_species_bucket"] == "murine"
    assert generated.species_context is not parent.species_context


def test_unknown_native_axes_and_conflicting_parent_contexts_stay_empty():
    context = native_species_context({"host": "rat", "source_organism": None})
    assert context["presenting_species"] == context["reported_antigen_species"] == ""
    parents = [
        BindingRecord(
            peptide="ACDEFGHIK",
            mhc_allele="HLA-A*02:01",
            value=50,
            evidence_row_id=str(index),
            species_context={"reported_host_species": host},
        )
        for index, host in enumerate(("rat", "human"))
    ]
    result = generated_record(
        BindingRecord(peptide="LMNPQRSTV", mhc_allele="HLA-A*02:01", value=50000),
        parents=parents,
        kind="test",
        seed=3,
        draw_index=0,
    )
    assert result.species_context["reported_host_species"] == ""


def test_role_change_alters_coverage_fingerprint_and_policy_is_declared():
    contract = build_output_contract()
    assert contract.to_dict()["schema_version"] == 4
    assert contract.to_dict()["molecular_mhc_species_policy"] == MOLECULAR_SPECIES_POLICY
    sample = PrestoSample(
        peptide="ACDEFGHIK",
        mhc_a="ACDEFGHIK",
        primary_allele="HLA-A*02:01",
        species_context={"reported_host_species": "rat"},
        sample_id="source-1",
    )
    hashes = []
    for row in (sample, replace(sample, species_context={"reported_host_species": "human"})):
        with OutputCoverageCensus(contract) as census:
            census.add_batch("train", [row], PrestoCollator()([row]))
            hashes.append(census.sha256)
    assert hashes[0] != hashes[1]
