"""Observed assay descriptors survive quantitative merged-TSV ingestion."""

import csv

import pytest
import torch

from presto.data.collate import PrestoCollator
from presto.data.loaders import PrestoDataset
from presto.data.vocab import (
    BINDING_ASSAY_GEOMETRY_TO_IDX,
    BINDING_ASSAY_PREP_TO_IDX,
    BINDING_ASSAY_READOUT_TO_IDX,
    BINDING_ASSAY_TYPE_TO_IDX,
)
from presto.models.presto import Presto
from presto.scripts.train_iedb import load_records_from_merged_tsv
from presto.scripts.train_synthetic import compute_loss


def _load(tmp_path, **fields):
    row = {
        "peptide": "SIINFEKL",
        "mhc_allele": "HLA-A*02:01",
        "mhc_class": "I",
        "source": "iedb",
        "record_type": "binding",
        "value": "100",
        "value_type": "IC50",
        "qualifier": "-1",
        "assay_type": "IC50",
        "assay_method": "purified MHC/direct/fluorescence",
        **fields,
    }
    path = tmp_path / "merged.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    return load_records_from_merged_tsv(
        path,
        max_binding=0,
        max_kinetics=0,
        max_stability=0,
        max_processing=0,
        max_elution=0,
        max_tcell=0,
        max_vdjdb=0,
    )


@pytest.mark.parametrize(
    "measurement,assay_type,group,value_field,qualifier_field",
    [
        ("IC50", "dissociation constant KD (~IC50)", 0, "value", "qualifier"),
        ("kon", "association rate", 1, "kon", "kon_qualifier"),
        ("koff", "dissociation rate", 1, "koff", "koff_qualifier"),
        ("t_half", "half life", 2, "t_half", "t_half_qualifier"),
        ("Tm", "50% dissociation temperature", 2, "tm", "tm_qualifier"),
    ],
)
def test_quantitative_adapter_preserves_observed_descriptors(
    tmp_path, measurement, assay_type, group, value_field, qualifier_field
):
    records = _load(tmp_path, value_type=measurement, assay_type=f" {assay_type} ")[group]
    assert len(records) == 1
    record = records[0]
    assert record.assay_type == assay_type
    assert record.assay_method == "purified MHC/direct/fluorescence"
    assert getattr(record, value_field) == 100.0
    assert getattr(record, qualifier_field) == -1


@pytest.mark.parametrize("measurement,group", [("IC50", 0), ("kon", 1), ("t_half", 2)])
def test_missing_descriptors_retain_existing_type_fallback(tmp_path, measurement, group):
    record = _load(tmp_path, value_type=measurement, assay_type=" ", assay_method=" ")[group][0]
    assert record.assay_method is None
    assert record.assay_type == (None if group == 0 else measurement)


def test_binding_cultures_and_conflicting_type_reach_output_selectors(tmp_path):
    record = _load(
        tmp_path,
        assay_type="dissociation constant KD (~IC50)",
        effector_culture_condition=" Direct ex vivo ",
        apc_culture_condition=" cultured ",
    )[0][0]
    assert record.measurement_type == "IC50"
    assert record.unit == "nM"
    assert record.effector_culture_condition == "Direct ex vivo"
    assert record.apc_culture_condition == "cultured"
    sample = _dataset_sample(record)
    assert sample.binding_effector_culture == "Direct ex vivo"
    assert sample.binding_apc_culture == "cultured"
    batch = PrestoCollator()([sample])
    assert (
        batch.binding_context["assay_type_idx"].item() == BINDING_ASSAY_TYPE_TO_IDX["KD_PROXY_IC50"]
    )
    assert batch.target_masks["binding_kd_proxy_ic50"].item() == 1.0
    assert "binding_ic50" not in batch.targets
    assert batch.bind_target.item() == 100.0
    assert batch.bind_qual.item() == -1


def _dataset_sample(record, record_group="binding"):
    # Supply exact groove metadata so optional external registry state cannot
    # determine whether this adapter test has a resolvable input.
    groove_a = "ACDEFGHIKLMNPQRS" * 4
    groove_b = "TVWYACDEFGHIKLMN" * 4
    dataset = PrestoDataset(
        **{f"{record_group}_records": [record]},
        mhc_exact_inputs={
            "HLA-A*02:01": {
                "allele": "HLA-A*02:01",
                "sequence": groove_a + groove_b,
                "groove1": groove_a,
                "groove2": groove_b,
                "mhc_class": "I",
                "chain": "alpha",
                "groove_status": "complete",
                "source": "test",
            }
        },
    )
    assert len(dataset) == 1
    return dataset[0]


@pytest.mark.parametrize(
    "measurement,group,index", [("koff", "kinetics", 1), ("t_half", "stability", 2)]
)
def test_other_quantitative_methods_do_not_create_binding_targets(
    tmp_path, measurement, group, index
):
    record = _load(tmp_path, value_type=measurement, assay_type=measurement)[index][0]
    batch = PrestoCollator()([_dataset_sample(record, group)])
    assert batch.binding_context["assay_prep_idx"].item() == BINDING_ASSAY_PREP_TO_IDX["PURIFIED"]
    assert (
        batch.binding_context["assay_readout_idx"].item()
        == BINDING_ASSAY_READOUT_TO_IDX["FLUORESCENCE"]
    )
    assert batch.bind_mask is None or batch.bind_mask.sum().item() == 0.0
    assert "binding_ic50" not in batch.targets


def test_observed_binding_method_selects_loss_without_changing_fixed_predictions(tmp_path):
    import dataclasses

    record = _load(tmp_path, qualifier="0")[0][0]
    sample = _dataset_sample(record)
    collator = PrestoCollator()
    batch = collator([sample])
    expected = {
        "assay_prep_idx": BINDING_ASSAY_PREP_TO_IDX["PURIFIED"],
        "assay_geometry_idx": BINDING_ASSAY_GEOMETRY_TO_IDX["DIRECT"],
        "assay_readout_idx": BINDING_ASSAY_READOUT_TO_IDX["FLUORESCENCE"],
    }
    for name, index in expected.items():
        assert batch.binding_context[name].item() == index

    unknown_batch = collator([dataclasses.replace(sample, binding_assay_method=None)])
    for name, value in batch.model_inputs().items():
        other = unknown_batch.model_inputs()[name]
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, other), name
        else:
            assert value == other, name
    torch.manual_seed(49)
    model = Presto(d_model=32, n_layers=1, n_heads=4).eval()
    with torch.no_grad():
        observed = model(**batch.model_inputs())
        unknown = model(**unknown_batch.model_inputs())
    for axis in ("assay_prep", "assay_geometry", "assay_readout"):
        key = f"binding_assay_panel_{axis}"
        assert torch.equal(observed[key], unknown[key])
    loss, terms, _ = compute_loss(model, batch, "cpu")
    assert "binding_assay_panel" in terms
    loss.backward()
    # Check the actual selected table rows below, not a whole parameter norm.
    for axis in ("assay_prep", "assay_readout"):
        index = expected[f"{axis}_idx"]
        table = model.affinity_predictor.assay_panel_embed[axis]
        assert table.weight.grad[index].abs().sum().item() > 0.0
        assert table.weight.grad[0].abs().sum().item() == 0.0
