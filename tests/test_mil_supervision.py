"""Bag response supervision must agree across loss, census and exported rows."""

import csv
import json
import math

import pytest
import torch

from presto.data.collate import PrestoCollator, PrestoSample
from presto.data.vocab import TCELL_ASSAY_METHOD_TO_IDX
from presto.models.presto import Presto
from presto.scripts.train_synthetic import (
    LOSS_TASK_SPECS,
    _get_batch_mask,
    _get_batch_qual,
    _get_batch_target,
    _resolve_task_prediction,
    compute_loss,
    evaluate,
)
from presto.training.data_support import audit_split_support, write_split_support_artifacts
from presto.training.holdout_eval import collect_holdout_predictions, write_holdout_artifacts
from presto.training.losses import mil_bag_loss
from presto.training.mil import MIL_TASKS, get_mil_channel, predict_mil_channel, resolve_mil_targets


def sample(*, tcell=False, n=2, label=1.0, **kwargs):
    values = dict(
        peptide="SIINFEKL", mhc_a="ACDEFGHIK", mhc_b="LMNPQRSTV", mhc_class="I", sample_id="bag"
    )
    prefix = "tcell_mil" if tcell else "mil"
    values.update(
        {
            f"{prefix}_mhc_a_list": ["ACDEFGHIK"] * n,
            f"{prefix}_mhc_b_list": ["LMNPQRSTV"] * n,
            f"{prefix}_mhc_class_list": ["I", "II"] + ["I"] * (n - 2)
            if tcell and n >= 2
            else ["I"] * n,
        }
    )
    if tcell:
        values.update(tcell_label=label, use_tcell_pathway_mil=True, tcell_assay_method="ELISPOT")
    else:
        values.update(elution_label=label)
    values.update(kwargs)
    return PrestoSample(**values)


class PanelModel(torch.nn.Module):
    """Independent output columns make accidentally trained columns observable."""

    def __init__(self, scalar=0.0):
        super().__init__()
        self.scalar = torch.nn.Parameter(torch.tensor(scalar))
        self.columns = torch.nn.ParameterDict(
            {
                spec.axis: torch.nn.Parameter(torch.linspace(-0.7, 0.8, len(spec.columns)))
                for spec in MIL_TASKS["tcell_mil"]
                if spec.axis
            }
        )
        self.seen = []

    def forward(self, **kwargs):
        assert "tcell_context" not in kwargs
        n = len(kwargs["pep_tok"])
        self.seen.append(kwargs)
        scalar = self.scalar.expand(n, 1)
        return {
            **{
                key: scalar
                for key in (
                    "elution_logit",
                    "presentation_logit",
                    "ms_logit",
                    "tcell_logit",
                    "immunogenicity_logit",
                )
            },
            "tcell_panel_logits": {key: value.expand(n, -1) for key, value in self.columns.items()},
        }


def collect(model, batch, chunk_size=1):
    return collect_holdout_predictions(
        model,
        [batch],
        "cpu",
        LOSS_TASK_SPECS,
        forward_fn=lambda m, b: m(**b.model_inputs()),
        resolve_pred_fn=_resolve_task_prediction,
        get_target_fn=_get_batch_target,
        get_mask_fn=_get_batch_mask,
        get_qual_fn=_get_batch_qual,
        mil_chunk_size=chunk_size,
    )


@pytest.mark.parametrize("label", [0.0, 1.0, 0.3])
def test_exact_noisy_or_loss_dump_and_support_agree(label, tmp_path):
    samples = [sample(label=label), sample(n=3, label=1 - label, sample_id="other")]
    batch = PrestoCollator()(samples)
    model = PanelModel()
    _, losses, metrics = compute_loss(model, batch, "cpu", mil_chunk_size=1)
    accumulators = collect(model, batch)
    rows = accumulators["elution"].rows()
    assert [row["y_prob"] for row in rows] == pytest.approx([0.75, 0.875])
    assert [row["instance_count"] for row in rows] == [2, 3]
    assert all(row["observation_kind"] == "bag" for row in rows)
    for task in ("elution", "presentation", "ms"):
        task_rows = accumulators[task].rows()
        reconstructed = sum(row["observation_loss"] for row in task_rows) / len(task_rows)
        assert reconstructed == pytest.approx(float(losses[task]))
        assert metrics[f"batch_support_{task}"] == len(task_rows)
    expected = -label * math.log(0.75) - (1 - label) * math.log(0.25)
    assert rows[0]["observation_loss"] == pytest.approx(expected)
    audit = audit_split_support({"test": samples})
    assert audit["splits"]["test"]["mil_targets"]["elution"]["count"] == len(rows)
    assert audit["splits"]["test"]["mil_targets"]["ms"]["alias_of"] == "elution"
    write_holdout_artifacts(tmp_path, accumulators, split="test")
    with (tmp_path / "test_predictions.csv").open() as handle:
        exported = list(csv.DictReader(handle))
    first = next(row for row in exported if row["task"] == "elution")
    assert json.loads(first["bag_instance_indices"]) == [0, 1]


@pytest.mark.parametrize("label", [0.0, 1.0])
def test_selected_tcell_bag_column_receives_response_gradient(label):
    torch.manual_seed(13)
    batch = PrestoCollator()([sample(tcell=True, label=label)])
    model = Presto(d_model=32, n_layers=2, n_heads=4)
    model.eval()
    _, losses, _ = compute_loss(model, batch, "cpu", max_mil_instances=1)
    assert batch.target_masks["tcell_assay_method"].sum() == 0
    assert "tcell_assay_method" not in losses
    losses["tcell_assay_method_mil"].backward()
    grad = model.tcell_assay_head.assay_method_embed.weight.grad
    selected = TCELL_ASSAY_METHOD_TO_IDX["ELISPOT"]
    assert grad[selected].abs().sum() > 0
    others = torch.arange(grad.shape[0]) != selected
    assert torch.count_nonzero(grad[others]) == 0


def test_mixed_row_bags_unknown_selectors_and_column_census(tmp_path):
    samples = [
        sample(tcell=True, n=3, sample_id="positive", sample_source="iedb"),
        sample(
            tcell=True,
            label=0,
            tcell_assay_method="ICS",
            sample_id="negative",
            sample_source="cedar",
        ),
        sample(tcell=True, tcell_assay_method=None, sample_id="unknown"),
        sample(tcell=True, use_tcell_pathway_mil=False, sample_id="row"),
    ]
    batch = PrestoCollator()(samples)
    model = PanelModel()
    _, losses, metrics = compute_loss(model, batch, "cpu", max_mil_instances=1)
    assert "tcell_assay_method" in losses and "tcell_assay_method_mil" in losses
    assert metrics["batch_support_tcell_assay_method_mil"] == 2
    assert batch.target_masks["tcell_assay_method"].tolist() == [0, 0, 0, 1]
    rows = collect(model, batch)
    panel_rows = rows["tcell_assay_method_mil"].rows()
    assert [row["selector_name"] for row in panel_rows] == ["ELISPOT", "ICS"]
    assert [row["source"] for row in panel_rows] == ["iedb", "cedar"]
    assert rows["tcell_assay_method"].rows()[0]["sample_id"] == "row"
    assert len(rows["tcell_mil"]) == 3
    audit = audit_split_support({"train": samples, "test": []}, chunk_size=2)
    counts = audit["splits"]["train"]["mil_targets"]["tcell_assay_method_mil"]
    assert counts["count"] == 2 and counts["unknown_selector"] == 1
    assert counts["columns"]["ELISPOT"]["positive"] == 1
    assert counts["columns"]["ICS"]["negative"] == 1
    assert counts["columns"]["ELISA"]["count"] == 0
    assert counts["sources"]["cedar"]["negative"] == 1
    assert counts["instances"] == 5
    assert audit["splits"]["test"]["mil_targets"]["tcell_assay_method_mil"]["count"] == 0
    paths = write_split_support_artifacts(tmp_path, audit)
    assert "ELISPOT,1,1,0" in paths["mil_csv"].read_text()


def test_duplicate_sample_ids_and_class_split_bags_keep_source_positions():
    samples = [
        sample(
            sample_id="duplicate",
            sample_source="iedb",
            evidence_row_id="first",
            mil_mhc_class_list=["I", "II"],
        ),
        sample(sample_id="duplicate", sample_source="cedar", evidence_row_id="second"),
    ]
    batch = PrestoCollator()(samples).to("cpu")
    assert batch.mil_bag_sample_indices == [0, 0, 1]
    rows = collect(PanelModel(), batch)["elution"].rows()
    assert [row["bag_id"] for row in rows] == ["duplicate:I", "duplicate:II", "duplicate"]
    assert [row["source"] for row in rows] == ["iedb", "iedb", "cedar"]
    assert [row["evidence_row_id"] for row in rows] == ["first", "first", "second"]
    batch.mil_bag_sample_indices = []
    with pytest.raises(ValueError, match="source-row positions"):
        collect(PanelModel(), batch)


@pytest.mark.parametrize("chunk_size", [1, 7, 128, 0])
def test_complete_chunked_bags_match_dense_loss_and_preserve_inputs(chunk_size):
    batch = PrestoCollator()(
        [
            sample(n=259, flank_n_is_terminus=True, processing_stimulus="ifn_gamma"),
            sample(n=3, label=0.0, sample_id="negative", flank_n_is_terminus=True),
        ]
    )
    model = PanelModel(scalar=-6.0)
    channel = get_mil_channel(batch, "mil")
    targets = resolve_mil_targets(channel, MIL_TASKS["mil"])
    with torch.no_grad():
        predictions = predict_mil_channel(
            model, channel=channel, targets=targets, device="cpu", chunk_size=chunk_size
        )
    p = model.scalar.detach().sigmoid()
    dense = p.expand(2, 259)
    mask = torch.zeros_like(dense)
    mask[0, :] = 1
    mask[1, :3] = 1
    expected_loss, expected_probs = mil_bag_loss(dense, torch.tensor([1.0, 0.0]), mask)
    assert predictions["elution"].probabilities == pytest.approx(expected_probs, abs=2e-6)
    assert float(predictions["elution"].loss()) == pytest.approx(float(expected_loss), abs=2e-6)
    assert predictions["elution"].evaluated_counts.tolist() == [259, 3]
    assert all(call["flank_n_is_terminus"].all() for call in model.seen)
    assert all(
        len(value) == len(call["pep_tok"])
        for call in model.seen
        for value in call["provenance"].values()
    )
    if chunk_size:
        assert max(len(call["pep_tok"]) for call in model.seen) <= chunk_size


def test_training_sampling_does_not_change_complete_eval_or_export():
    batch = PrestoCollator()([sample(n=3)])
    model = PanelModel()
    _, capped, _ = compute_loss(model, batch, "cpu", max_mil_instances=1)
    _, full = evaluate(
        model, [batch], "cpu", show_progress=False, max_mil_instances=0, mil_chunk_size=1
    )
    row = collect(model, batch)["elution"].rows()[0]
    assert float(capped["elution"]) == pytest.approx(-math.log(0.5))
    assert full["loss_elution"] == pytest.approx(-math.log(0.875))
    assert row["observation_loss"] == pytest.approx(full["loss_elution"])
    assert row["evaluated_instance_count"] == 3


def test_inconsistent_or_invalid_selectors_fail_before_training_cap():
    batch = PrestoCollator()([sample(tcell=True)])
    batch.tcell_mil_context["assay_method_idx"][1] = TCELL_ASSAY_METHOD_TO_IDX["ICS"]
    with pytest.raises(ValueError, match="Inconsistent assay_method_idx"):
        compute_loss(PanelModel(), batch, "cpu", max_mil_instances=1)
    batch.tcell_mil_context["assay_method_idx"][1] = 100
    with pytest.raises(ValueError, match="Invalid MIL selector"):
        compute_loss(PanelModel(), batch, "cpu", max_mil_instances=1)


def test_all_six_selected_axes_share_row_column_contract_and_train_only_selected_columns():
    batch = PrestoCollator()([sample(tcell=True)])
    # Isolate selection from categorization: choose a distinct declared column
    # on every axis, then exercise actual loss and export consumers.
    for spec in MIL_TASKS["tcell_mil"]:
        if spec.axis:
            batch.tcell_mil_context[spec.selector_key].fill_(2)
            row_spec = next(
                row for row in LOSS_TASK_SPECS if row.name == spec.name.removesuffix("_mil")
            )
            assert row_spec.selector_key == spec.selector_key
            assert row_spec.pred_paths == (spec.output_path,)
    model = PanelModel()
    _, losses, _ = compute_loss(model, batch, "cpu", mil_chunk_size=1)
    panel_specs = [spec for spec in MIL_TASKS["tcell_mil"] if spec.axis]
    sum(losses[spec.name] for spec in panel_specs).backward()
    rows = collect(model, batch)
    for spec in panel_specs:
        grad = model.columns[spec.axis].grad
        assert grad[2].abs() > 0
        assert torch.count_nonzero(grad) == 1
        exported = rows[spec.name].rows()[0]
        assert exported["selector_name"] == spec.columns[2]
        assert exported["observation_loss"] == pytest.approx(float(losses[spec.name]))


def test_actual_model_full_and_chunked_bag_predictions_agree():
    torch.manual_seed(21)
    model = Presto(d_model=32, n_layers=2, n_heads=4).eval()
    batch = PrestoCollator()([sample(tcell=True, n=3), sample(tcell=True, label=0.0)])
    channel = get_mil_channel(batch, "tcell_mil")
    targets = resolve_mil_targets(channel, MIL_TASKS["tcell_mil"])
    with torch.no_grad():
        full = predict_mil_channel(model, channel=channel, targets=targets, device="cpu")
        chunked = predict_mil_channel(
            model, channel=channel, targets=targets, device="cpu", chunk_size=1
        )
    assert full.keys() == chunked.keys()
    for name in full:
        torch.testing.assert_close(
            full[name].probabilities, chunked[name].probabilities, atol=1e-6, rtol=1e-5
        )


def test_incomplete_channel_is_an_error_instead_of_silent_missing_predictions():
    batch = PrestoCollator()([sample()])
    batch.mil_pep_tok = None
    with pytest.raises(ValueError, match="Incomplete MIL channel"):
        collect(PanelModel(), batch)


def test_canonical_selected_checkpoint_evaluation_has_no_training_cap():
    import ast
    import inspect

    from presto.scripts import train_iedb

    tree = ast.parse(inspect.getsource(train_iedb))
    final_loss = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Tuple)
        and any(
            isinstance(name, ast.Name) and name.id == "heldout_loss"
            for name in node.targets[0].elts
        )
    )
    kwargs = {keyword.arg: keyword.value for keyword in final_loss.keywords}
    assert ast.literal_eval(kwargs["max_mil_instances"]) == 0
    assert ast.literal_eval(kwargs["max_val_batches"]) == 0
    assert ast.literal_eval(kwargs["use_amp"]) is False
    assert kwargs["mil_chunk_size"].id == "DEFAULT_MIL_EVAL_CHUNK_SIZE"
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "collect_holdout_predictions"
    ]
    assert len(calls) == 1
    export_kwargs = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert export_kwargs["mil_chunk_size"].id == kwargs["mil_chunk_size"].id
