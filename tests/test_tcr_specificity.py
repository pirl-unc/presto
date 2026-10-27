"""Contract checks use artificial sequences, never released receptor records."""

import csv

import pytest
import torch

from presto.data.tcr_specificity import TCRSpecificityRecord, load_tcr_specificity_csv
from presto.legacy.presto import Presto
from presto.legacy.tcr_specificity import TCRSpecificityModel, functional_response_loss


@pytest.fixture
def pair():
    torch.manual_seed(17)
    model = TCRSpecificityModel(
        Presto(d_model=16, n_layers=1, n_heads=4, binding_n_queries=2),
        d_model=16,
        n_layers=1,
        dropout=0,
    ).eval()
    inputs = dict(
        tcr_alpha=["ACDEFGHIKLMNPQRSTVWY" * 12],
        tcr_beta=["YWVTSRQPNMLKIHGFEDCA" * 13],
        pep_tok=torch.tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12]]),
        mhc_a_tok=torch.arange(4, 24).repeat(1, 3),
        mhc_b_tok=torch.arange(4, 24).repeat(1, 3),
        mhc_class="I",
    )
    return model, inputs


def test_chain_and_pmhc_sensitivity_and_padding(pair):
    model, inputs = pair
    reference = model(**inputs)["tcr_functional_logit"]
    assert reference.shape == (1,)
    assert torch.isfinite(reference).all()
    for key in ("tcr_alpha", "tcr_beta"):
        # Changing the end of a long chain must survive tokenization.
        changed = {**inputs, key: [inputs[key][0][:-10] + "A" * 10]}
        assert not torch.equal(reference, model(**changed)["tcr_functional_logit"])
    changed = {**inputs, "pep_tok": torch.tensor([[22] * 9])}
    assert not torch.equal(reference, model(**changed)["tcr_functional_logit"])
    changed = {**inputs, "mhc_a_tok": torch.tensor([[5] * 60])}
    assert not torch.equal(reference, model(**changed)["tcr_functional_logit"])
    batched = {
        **inputs,
        "tcr_alpha": inputs["tcr_alpha"] + ["ACDE" * 90],
        "tcr_beta": inputs["tcr_beta"] + ["ACDE" * 95],
        "pep_tok": inputs["pep_tok"].repeat(2, 1),
        "mhc_a_tok": inputs["mhc_a_tok"].repeat(2, 1),
        "mhc_b_tok": inputs["mhc_b_tok"].repeat(2, 1),
    }
    torch.testing.assert_close(reference, model(**batched)["tcr_functional_logit"][:1])


def test_freezing_modes_gradients_and_checkpoint(pair, tmp_path):
    model, inputs = pair
    model.train()
    assert not model.backbone.training
    loss = functional_response_loss(model(**inputs)["tcr_functional_logit"], torch.ones(1))
    loss.backward()
    assert all(p.grad is None for p in model.backbone.parameters())
    for module in (model.alpha_encoder, model.beta_encoder, model.response_head):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    model.set_backbone_frozen(False)
    assert model.backbone.training
    model.zero_grad()
    functional_response_loss(model(**inputs)["tcr_functional_logit"], torch.zeros(1)).backward()
    assert model.backbone.aa_embedding.weight.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.backbone.aa_embedding.weight.detach().clone()
    model.set_backbone_frozen(True)
    assert all(p.grad is None for p in model.backbone.parameters())
    optimizer.step()
    torch.testing.assert_close(model.backbone.aa_embedding.weight, before)
    model.eval()
    path = tmp_path / "specificity.pt"
    model.save_checkpoint(path, metadata={"assay_scope": "synthetic unit test"})
    loaded, metadata = TCRSpecificityModel.load_checkpoint(path)
    assert loaded.backbone.binding_n_queries == 2
    assert metadata == {"assay_scope": "synthetic unit test"}
    torch.testing.assert_close(
        model(**inputs)["tcr_functional_logit"], loaded(**inputs)["tcr_functional_logit"]
    )


@pytest.mark.parametrize("alpha", [[], [""], ["ACX"], ["A" * 513], "ACDE"])
def test_reject_invalid_chain(pair, alpha):
    model, inputs = pair
    with pytest.raises(ValueError, match="chain|paired|batches"):
        model(**{**inputs, "tcr_alpha": alpha})


def test_reject_unpaired_batch_and_targets(pair):
    model, inputs = pair
    with pytest.raises(ValueError, match="pep_tok batch"):
        model(**{**inputs, "pep_tok": inputs["pep_tok"].repeat(2, 1)})
    for target in (torch.tensor([float("nan")]), torch.tensor([-1.0]), torch.ones(1, 1)):
        with pytest.raises(ValueError, match="Require|outcomes|Targets"):
            functional_response_loss(torch.zeros(1), target)


def row(**overrides):
    return (
        dict(
            record_id="test:1",
            study_id="artificial-test",
            assay_id="functional-test",
            peptide="ACDEFGHIK",
            mhc_a="HLA-A*02:01",
            mhc_b="",
            mhc_class="I",
            tcr_alpha="ACDE" * 60,
            tcr_beta="FGHI" * 65,
            sequence_provenance="observed_full_length",
            sequence_reference="artificial fixture; not an observed biological receptor",
            outcome="validated",
            quality="pass",
        )
        | overrides
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TCRSpecificityRecord.__dataclass_fields__)
        writer.writeheader()
        writer.writerows(rows)


def test_loader_preserves_measured_outcomes_and_excludes_unknown(tmp_path):
    path = tmp_path / "normalized.csv"
    write_csv(
        path,
        [
            row(),
            row(record_id="2", outcome="not_validated", sequence_provenance="reconstructed"),
            row(record_id="3", quality="fail"),
            row(record_id="4", outcome="indeterminate"),
            row(record_id="5", outcome="unassayed"),
        ],
    )
    loaded = load_tcr_specificity_csv(path)
    assert [r.target for r in loaded.records] == [1, 0]
    assert loaded.records[1].sequence_provenance == "reconstructed"
    assert len(loaded.source_sha256) == 64
    assert loaded.exclusions == {"quality_fail": 1, "indeterminate": 1, "unassayed": 1}


@pytest.mark.parametrize(
    "change",
    [
        {"tcr_beta": ""},
        {"mhc_class": "II"},
        {"outcome": "positive"},
        {"sequence_reference": ""},
        {"sequence_provenance": "cdr3_only"},
    ],
)
def test_loader_rejects_unqualified_schema(tmp_path, change):
    path = tmp_path / "normalized.csv"
    write_csv(path, [row(**change)])
    with pytest.raises(ValueError, match="chain|Class II|outcome|sequence_reference|Declare"):
        load_tcr_specificity_csv(path)


def test_loader_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "normalized.csv"
    write_csv(path, [row(), row()])
    with pytest.raises(ValueError, match="duplicate"):
        load_tcr_specificity_csv(path)


@pytest.mark.parametrize("change", [{"outcome": "unassayed"}, {"quality": "fail"}])
def test_direct_record_cannot_turn_unknown_into_negative(change):
    record = TCRSpecificityRecord(**row(**change))
    with pytest.raises(ValueError, match="Only measured"):
        _ = record.target
