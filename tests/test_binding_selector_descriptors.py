"""Alternate merged selectors must train the same observed columns as canonical ingestion."""

import csv
from dataclasses import asdict, replace

import pytest
import torch

from presto.data.collate import PrestoCollator
from presto.data.loaders import PrestoDataset
from presto.data.vocab import BINDING_ASSAY_PREP_TO_IDX, BINDING_ASSAY_READOUT_TO_IDX
from presto.models.presto import Presto
from presto.scripts.focused_binding_probe import _load_binding_records_from_merged_tsv
from presto.scripts.train_iedb import (
    load_binding_records_for_alleles_from_merged_tsv,
    load_probe_allele_binding_bootstrap_from_merged_tsv,
    load_records_from_merged_tsv,
)
from presto.scripts.train_synthetic import compute_loss

ALLELES = ["HLA-A*02:01", "HLA-A*03:01"]
FIELDS = ("assay_type", "assay_method", "effector_culture_condition", "apc_culture_condition")
DESCRIPTORS = [
    dict(
        value_type="IC50",
        assay_type="dissociation constant KD (~IC50)",
        assay_method="purified MHC/direct/fluorescence",
        effector_culture_condition="Direct ex vivo",
        apc_culture_condition="cultured",
    ),
    dict(value_type="", assay_type=" KD ", assay_method="purified MHC/competitive/fluorescence"),
    dict(value_type="", assay_type="", assay_method=" IC50 "),
    dict(value_type="IC50", assay_type="", assay_method=""),
    dict(
        value_type="IC50",
        assay_type="IC50",
        assay_method=" cellular MHC/competitive/radioactivity ",
        effector_culture_condition=" Direct ex vivo ",
        apc_culture_condition=" cultured ",
    ),
]


def write_source(path, descriptors, *, exact=False):
    rows = [
        dict(
            peptide="ACDEFGHIK",
            mhc_allele=allele,
            mhc_class="I",
            source="iedb",
            species="human",
            record_type="binding",
            value=100 * (index + 1),
            qualifier=0 if exact else 2 * index - 1,
            evidence_row_id=f"observation:{index}",
            assay_iri=f"assay:{index}",
            pmid="12345678",
            doi="10.1234/example",
            reference_text="Reference text",
            **descriptors,
        )
        for index, allele in enumerate(ALLELES)
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def load(path, selector):
    if selector == "canonical":
        return load_records_from_merged_tsv(
            path,
            max_binding=0,
            max_kinetics=0,
            max_stability=0,
            max_processing=0,
            max_elution=0,
            max_tcell=0,
            max_vdjdb=0,
        )[0]
    if selector == "panel":
        return load_binding_records_for_alleles_from_merged_tsv(path, alleles=ALLELES)[0]
    if selector == "bootstrap":
        return load_probe_allele_binding_bootstrap_from_merged_tsv(
            path, probe_alleles=ALLELES, max_records=10, max_peptides=10
        )[0]
    if selector == "focused":
        return _load_binding_records_from_merged_tsv(path, alleles=ALLELES)[0]
    raise ValueError(selector)


def samples(records):
    groove1, groove2 = "ACDEFGHIKLMNPQRS" * 4, "TVWYACDEFGHIKLMN" * 4
    return list(
        PrestoDataset(
            binding_records=records,
            mhc_exact_inputs={
                allele: dict(
                    allele=allele,
                    sequence=groove1 + groove2,
                    groove1=groove1,
                    groove2=groove2,
                    mhc_class="I",
                    chain="alpha",
                    groove_status="complete",
                    source="test",
                )
                for allele in ALLELES
            },
        )
    )


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            equal_tree(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("selector", ["canonical", "panel", "bootstrap", "focused"])
@pytest.mark.parametrize("descriptors", DESCRIPTORS)
def test_all_paths_preserve_observations_and_select_identical_targets(
    tmp_path, selector, descriptors
):
    path = write_source(tmp_path / "binding.tsv", descriptors)
    expected = load(path, "canonical")
    actual = load(path, selector)
    assert len(actual) == 2
    for index, record in enumerate(actual):
        for name in FIELDS:
            assert getattr(record, name) == (descriptors.get(name, "").strip() or None)
        assert record.value == 100 * (index + 1)
        assert record.unit == "nM"
        assert record.qualifier == 2 * index - 1
        assert record.evidence_row_id == f"observation:{index}"
    assert [asdict(record) for record in actual] == [asdict(record) for record in expected]
    collator = PrestoCollator()
    actual_batch, expected_batch = collator(samples(actual)), collator(samples(expected))
    # Covers actual inputs, target/mask/qualifier tensors, selector columns,
    # stable IDs and all publication/mapping lineage.
    equal_tree(vars(actual_batch), vars(expected_batch))


@pytest.mark.parametrize("selector", ["panel", "bootstrap"])
def test_observed_method_selects_gradient_without_changing_predictions(tmp_path, selector):
    path = write_source(tmp_path / "binding.tsv", DESCRIPTORS[0], exact=True)
    observed_samples = samples(load(path, selector))
    collator = PrestoCollator()
    batch = collator(observed_samples)
    unknown = collator([replace(sample, binding_assay_method=None) for sample in observed_samples])
    equal_tree(batch.model_inputs(), unknown.model_inputs())
    torch.manual_seed(49)
    model = Presto(d_model=32, n_layers=1, n_heads=4).eval()
    with torch.no_grad():
        observed_outputs = model(**batch.model_inputs())
        unknown_outputs = model(**unknown.model_inputs())
    for axis in ("assay_prep", "assay_geometry", "assay_readout"):
        assert torch.equal(
            observed_outputs[f"binding_assay_panel_{axis}"],
            unknown_outputs[f"binding_assay_panel_{axis}"],
        )
    loss, terms, _ = compute_loss(model, batch, "cpu")
    assert "binding_assay_panel" in terms
    loss.backward()
    for axis, index in (
        ("assay_prep", BINDING_ASSAY_PREP_TO_IDX["PURIFIED"]),
        ("assay_readout", BINDING_ASSAY_READOUT_TO_IDX["FLUORESCENCE"]),
    ):
        gradient = model.affinity_predictor.assay_panel_embed[axis].weight.grad
        assert gradient[index].abs().sum().item() > 0
        assert gradient[0].abs().sum().item() == 0
