"""Row, class, component and selected-column identities survive loss and export."""

import csv
import json
import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from presto.data.collate import PrestoCollator, PrestoSample, TCR_EVIDENCE_METHOD_BINS
from presto.models.presto import Presto
from presto.scripts.train_synthetic import compute_loss, evaluate
from presto.training.data_support import audit_split_support, write_split_support_artifacts
from presto.training.evaluation_ledger import reconcile_prediction_artifact
from presto.training.holdout_eval import collect_holdout_predictions, write_holdout_artifacts
from presto.training.supervision import (
    LOSS_TASK_SPECS,
    PANEL_TASK_SPECS,
    _get_batch_mask,
    _get_batch_qual,
    _get_batch_target,
    _resolve_task_prediction,
    make_row_target,
    pair_row_prediction,
    resolve_row_targets,
)


def sample(**kwargs):
    values = dict(
        peptide="ACDEFGHIKLM",
        mhc_a="ACDEFGHIK",
        mhc_b="LMNPQRSTV",
        mhc_class="I",
        species="human",
        primary_allele="HLA-A*02:01",
        sample_id="first",
        sample_source="measured",
        bind_value=25.0,
        binding_assay_type="KD",
        elution_label=1.0,
        tcell_label=1.0,
        tcell_assay_method="ELISPOT",
        core_start=0,
        species_of_origin="human",
        tcr_evidence_label=1.0,
        tcr_evidence_method_bins=(TCR_EVIDENCE_METHOD_BINS[0],),
        kon=1000.0,
        koff=0.01,
        t_half=2.0,
        tm=65.0,
    )
    values.update(kwargs)
    return PrestoSample(**values)


def collect(model, batches):
    return collect_holdout_predictions(
        model,
        batches,
        "cpu",
        LOSS_TASK_SPECS,
        forward_fn=lambda m, b: m(**b.model_inputs()),
        resolve_pred_fn=_resolve_task_prediction,
        get_target_fn=_get_batch_target,
        get_mask_fn=_get_batch_mask,
        get_qual_fn=_get_batch_qual,
        mil_chunk_size=1,
    )


@pytest.fixture
def model():
    torch.manual_seed(17)
    return Presto(d_model=32, n_layers=1, n_heads=4).eval()


def test_categorical_and_vector_observations_keep_source_lineage(model, tmp_path):
    samples = [
        sample(evidence_row_id="source-A"),
        sample(
            sample_id="second",
            evidence_row_id="source-B",
            mhc_class="II",
            primary_allele="HLA-DRA*01:01",
            core_start=1,
            tcr_evidence_method_bins=(TCR_EVIDENCE_METHOD_BINS[2],),
        ),
    ]
    batch = PrestoCollator()(samples)
    collection = collect(model, [batch])
    rows = collection["tcr_evidence_method"].rows()
    assert [row["sample_id"] for row in rows] == ["first"] * 3 + ["second"] * 3
    assert [row["evidence_row_id"] for row in rows] == ["source-A"] * 3 + ["source-B"] * 3
    assert [row["source_row_index"] for row in rows] == [0] * 3 + [1] * 3
    assert [row["component_name"] for row in rows] == list(TCR_EVIDENCE_METHOD_BINS) * 2
    assert [row["y_true"] for row in rows] == [1, 0, 0, 0, 0, 1]
    for name in (
        "mhc_class",
        "mhc_species",
        "mhc_a_fine_type",
        "mhc_b_fine_type",
        "species_of_origin",
        "core_start",
    ):
        records = collection[name].rows()
        assert len(records) == 2, name
        for row in records:
            probabilities = json.loads(row["class_probabilities"])
            assert sum(probabilities) == pytest.approx(1)
            assert row["observation_loss"] == pytest.approx(
                -math.log(probabilities[int(row["y_true"])])
            )
        assert "cross_entropy" in collection[name].metrics()
    audit = audit_split_support({"test": samples})["splits"]["test"]["row_targets"]
    assert audit["tcr_evidence_method"]["count"] == 6
    assert audit["tcr_evidence_method"]["sources"]["measured"]["count"] == 6
    assert all(
        counts["count"] == 2 for counts in audit["tcr_evidence_method"]["components"].values()
    )
    write_holdout_artifacts(tmp_path, collection)


def test_raw_quantitative_targets_survive_collation_and_device_transfer(model):
    batch = PrestoCollator()([sample(kon=0.0)]).to("cpu")
    assert "raw_targets" not in batch.model_inputs()
    records = collect(model, [batch])
    for task, raw, transformed, unit in (
        ("binding", 25, math.log10(25), "nM"),
        ("kon", 0, -12, "1/(M*s)"),
        ("koff", 0.01, -2, "1/s"),
        ("t_half", 2, math.log10(120), "h"),
        ("tm", 65, 1, "degC"),
    ):
        row = records[task].rows()[0]
        assert row["raw_target"] == pytest.approx(raw)
        assert row["y_true"] == pytest.approx(transformed)
        assert row["raw_unit"] == unit


def test_binding_family_export_preserves_source_precision(model):
    value = 12.345678901234
    batch = PrestoCollator()([sample(bind_value=value, binding_assay_type="IC50")])
    collection = collect(model, [batch])
    assert collection["binding"].rows()[0]["raw_target"] == value
    assert collection["binding_ic50"].rows()[0]["raw_target"] == value
    moved = batch.to("meta")
    assert moved.pep_tok.device.type == "meta"
    assert moved.raw_targets["binding"].device.type == "cpu"
    assert moved.raw_targets["binding"].dtype == torch.float64


def test_invalid_active_observations_fail_explicitly():
    spec = next(spec for spec in LOSS_TASK_SPECS if spec.name == "binding")
    target = make_row_target(spec, torch.tensor([100.0]), torch.ones(1))
    with pytest.raises(ValueError, match="missing qualifier"):
        pair_row_prediction(target, torch.tensor([2.0]))
    with pytest.raises(ValueError, match="expected 1 scalar predictions"):
        pair_row_prediction(target, torch.zeros(1, 2))
    with pytest.raises(ValueError, match="selector outside declared columns"):
        make_row_target(
            PANEL_TASK_SPECS[0], torch.ones(1), torch.ones(1), selectors=torch.tensor([999])
        )


@pytest.mark.parametrize("aggregation", ["task_mean", "sample_weighted"])
def test_mixed_row_bag_and_panel_dump_reconstructs_canonical_evaluation(
    model, tmp_path, aggregation
):
    samples = [
        sample(bind_qual=-1),
        sample(
            sample_id="other",
            bind_value=500,
            bind_qual=1,
            elution_label=0,
            tcell_label=0,
            binding_assay_type="IC50",
        ),
    ]
    bag = sample(
        sample_id="bag",
        bind_value=None,
        elution_label=0,
        mil_mhc_a_list=["ACDEFGHIK"] * 3,
        mil_mhc_b_list=["LMNPQRSTV"] * 3,
        mil_mhc_class_list=["I"] * 3,
    )
    batches = [PrestoCollator()(samples), PrestoCollator()([bag])]
    receipts = []
    total, metrics = evaluate(
        model,
        batches,
        "cpu",
        show_progress=False,
        supervised_loss_aggregation=aggregation,
        regularization={"consistency_cascade_weight": 0.1},
        mil_chunk_size=1,
        batch_receipts=receipts,
    )
    collection = collect(model, batches)
    payload = write_holdout_artifacts(tmp_path, collection, expected_batches=receipts)
    ledger = json.loads((tmp_path / "val_loss_ledger.json").read_text())
    assert ledger["canonical_evaluation_compared"]
    assert ledger["overall_loss"] == pytest.approx(total, rel=2e-5, abs=2e-6)
    for name, loss in ledger["task_losses"].items():
        assert loss == pytest.approx(metrics[f"loss_{name}"], rel=2e-5, abs=2e-6)
    assert payload["output_support"]["ms"]["alias_of"] == "elution"
    assert any(
        counts["exported"] == 0
        for counts in payload["output_support"]["binding_assay_panel_assay_type"][
            "columns"
        ].values()
    )
    assert len(collection["excision_panel_apm"]) == 3


def test_panel_axes_preserve_censor_loss_and_selected_gradients(model):
    batch = PrestoCollator()(
        [sample(bind_qual=-1), sample(sample_id="second", bind_value=500, bind_qual=1)]
    )
    outputs = model(**batch.model_inputs())
    for spec in PANEL_TASK_SPECS:
        outputs[spec.name].retain_grad()

    class FixedOutputs(torch.nn.Module):
        def forward(self, **kwargs):
            return outputs

    _, losses, _ = compute_loss(FixedOutputs(), batch, "cpu")
    targets = resolve_row_targets(batch, PANEL_TASK_SPECS)
    axis_losses = []
    for spec in PANEL_TASK_SPECS[:4]:
        selected = outputs[spec.name][torch.arange(2), targets[spec.name].selectors]
        target = torch.log10(torch.tensor([25.0, 500.0]))
        axis_losses.append(
            (F.relu(selected[0] - target[0]).square() + F.relu(target[1] - selected[1]).square())
            / 2
        )
    assert float(losses["binding_assay_panel"]) == pytest.approx(
        float(torch.stack(axis_losses).mean())
    )
    losses["binding_assay_panel"].backward()
    for spec in PANEL_TASK_SPECS[:4]:
        gradient = outputs[spec.name].grad.clone()
        gradient[torch.arange(2), targets[spec.name].selectors] = 0
        assert torch.count_nonzero(gradient) == 0


@pytest.mark.parametrize("corruption", ["drop", "duplicate", "prediction", "class", "missing"])
def test_artifact_reconciliation_rejects_missing_or_corrupt_observations(
    model, tmp_path, corruption
):
    collection = collect(model, [PrestoCollator()([sample()])])
    write_holdout_artifacts(tmp_path, collection)
    path = tmp_path / "val_predictions.csv"
    if corruption == "missing":
        path.unlink()
    else:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        if corruption == "drop":
            rows.pop()
        elif corruption == "duplicate":
            rows.append(rows[0])
        elif corruption == "prediction":
            rows[0]["y_pred"] = "999"
        else:
            next(row for row in rows if row["loss_type"] == "ce")["class_probabilities"] = "[1.0]"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    with pytest.raises((ValueError, FileNotFoundError)):
        reconcile_prediction_artifact(path, collection)


def test_masked_nonfinite_values_do_not_poison_active_loss_or_gradient():
    spec = next(spec for spec in LOSS_TASK_SPECS if spec.name == "excision")
    target = make_row_target(spec, torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 0.0]))
    logits = torch.tensor([0.0, float("nan")], requires_grad=True)
    prediction = pair_row_prediction(target, logits)
    assert float(prediction.loss()) == pytest.approx(math.log(2))
    prediction.loss().backward()
    assert logits.grad.tolist() == [-0.5, 0.0]
    with pytest.raises(ValueError, match="nonfinite active observation loss"):
        pair_row_prediction(replace(target, mask=torch.ones(2), target=torch.ones(2)), logits)


def test_declared_zero_support_and_empty_artifacts(model, tmp_path):
    audit = audit_split_support({"test": []})
    assert audit["splits"]["test"]["row_targets"]["mhc_class"]["count"] == 0
    paths = write_split_support_artifacts(tmp_path, audit)
    assert "binding_assay_panel_assay_type" in paths["row_csv"].read_text()
    collection = collect(model, [])
    write_holdout_artifacts(tmp_path, collection, expected_batches=[])
    assert (tmp_path / "val_predictions.csv").read_text().startswith("task,sample_id")
    assert json.loads((tmp_path / "val_loss_ledger.json").read_text())["overall_loss"] == 0
