"""Coverage instrumentation observes actual optimizer behavior without changing it."""

import copy
import json

import pytest
import torch

from presto.data.collate import PrestoCollator
from presto.data.vocab import TCELL_ASSAY_METHODS
from presto.models.presto import Presto
from presto.scripts.train_synthetic import LOSS_TASK_NAMES, train_epoch
from presto.training.losses import PCGrad, UncertaintyWeighting
from presto.training.output_updates import OutputUpdateTracker
from test_output_coverage import sample


@pytest.fixture(scope="module", autouse=True)
def single_thread_checks():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def batch():
    return PrestoCollator()(
        [
            sample(
                bind_value=25,
                binding_assay_type="KD",
                bind_measurement_type="KD",
                tcell_label=1,
                tcell_assay_method="ELISPOT",
                elution_label=1,
                peptide_source="mhc",
                use_tcell_pathway_mil=True,
                tcell_mil_mhc_a_list=["ACDEFGHIK", "ACDEFGHIK"],
                tcell_mil_mhc_b_list=["LMNPQRSTV", "LMNPQRSTV"],
                tcell_mil_mhc_class_list=["I", "II"],
            ),
            sample(
                sample_id="second",
                evidence_row_id="assay-2",
                peptide="LMNPQRSTV",
                bind_value=1000,
                binding_assay_type="IC50",
                tcell_label=0,
                tcell_assay_method="ICS",
                elution_label=0,
                peptide_source="mhc",
            ),
        ]
    )


def cells(report):
    return {(row["endpoint"], row["column"]): row for row in report["outputs"]}


@pytest.mark.parametrize("topology", ["collapsed", "expanded"])
@pytest.mark.parametrize("weighting", ["plain", "uncertainty", "pcgrad"])
@pytest.mark.parametrize("aggregation", ["task_mean", "sample_weighted"])
def test_tracker_preserves_losses_all_parameter_updates_and_optimizer_state(
    topology, weighting, aggregation
):
    baseline = Presto(d_model=32, n_layers=1, n_heads=4, latent_topology=topology)
    observed = copy.deepcopy(baseline)
    batch_value = batch()
    optimizers = [
        torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.1)
        for model in (baseline, observed)
    ]
    uncertainty = [
        UncertaintyWeighting(n_tasks=len(LOSS_TASK_NAMES)) if weighting == "uncertainty" else None
        for _ in optimizers
    ]
    for optimizer, weights in zip(optimizers, uncertainty):
        if weights is not None:
            optimizer.add_param_group({"params": weights.parameters()})
    kwargs = dict(show_progress=False, supervised_loss_aggregation=aggregation)
    torch.manual_seed(45)
    loss, metrics = train_epoch(
        baseline,
        [batch_value],
        optimizers[0],
        "cpu",
        pcgrad=PCGrad(optimizers[0]) if weighting == "pcgrad" else None,
        uncertainty_weighting=uncertainty[0],
        **kwargs,
    )
    torch.manual_seed(45)
    with OutputUpdateTracker(observed, optimizers[1]) as tracker:
        tracked_loss, tracked_metrics = train_epoch(
            observed,
            [batch_value],
            optimizers[1],
            "cpu",
            output_tracker=tracker,
            pcgrad=PCGrad(optimizers[1]) if weighting == "pcgrad" else None,
            uncertainty_weighting=uncertainty[1],
            **kwargs,
        )
        report = tracker.report()
    assert tracked_loss == loss
    assert {k: v for k, v in metrics.items() if k.startswith("loss_")} == {
        k: v for k, v in tracked_metrics.items() if k.startswith("loss_")
    }
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(observed.state_dict()[name], value, rtol=0, atol=0)
    if weighting == "uncertainty":
        torch.testing.assert_close(uncertainty[0].log_vars, uncertainty[1].log_vars, rtol=0, atol=0)
    left, right = (optimizer.state_dict()["state"] for optimizer in optimizers)
    assert left.keys() == right.keys()
    for index, state in left.items():
        for key, value in state.items():
            torch.testing.assert_close(right[index][key], value, rtol=0, atol=0)
    assert report["optimizer_steps"] == report["batches_completed"] == 1
    method = cells(report)[("tcell_panel_logits.assay_method", "ELISPOT")]
    assert method["batches_with_direct_labels"] == method["label_observations_seen"] == 1
    assert method["batches_with_output_gradient"] == 1
    assert method["parameter_rows"][0]["steps_with_gradient"] == 1
    # The repeated MIL ms loss must not count the same elution label twice.
    assert cells(report)[("elution_logit", "")]["label_observations_seen"] == 2


def test_weight_decay_does_not_make_unused_columns_supervised():
    model = Presto(d_model=32, n_layers=1, n_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.1)
    with OutputUpdateTracker(model, optimizer) as tracker:
        train_epoch(model, [batch()], optimizer, "cpu", show_progress=False, output_tracker=tracker)
        unused = next(
            name for name in TCELL_ASSAY_METHODS if name not in {"unknown", "ELISPOT", "ICS"}
        )
        row = cells(tracker.report())[("tcell_panel_logits.assay_method", unused)]
        assert row["label_observations_seen"] == row["batches_with_output_gradient"] == 0
        assert row["parameter_rows"][0]["steps_with_gradient"] == 0
        assert row["parameter_rows"][0]["steps_updated_without_gradient"] == 1


def test_frozen_rows_zero_initialization_and_fixed_rules_are_distinguished(tmp_path):
    model = Presto(d_model=32, n_layers=1, n_heads=4)
    model.tcell_assay_head.assay_method_embed.weight.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    with OutputUpdateTracker(model, optimizer, output_dir=tmp_path) as tracker:
        tracker.epoch = 2
        train_epoch(model, [batch()], optimizer, "cpu", show_progress=False, output_tracker=tracker)
    report = json.loads((tmp_path / "output_updates.json").read_text())
    row = cells(report)[("tcell_panel_logits.assay_method", "ELISPOT")]
    assert row["batches_with_output_gradient"] == 1
    assert row["parameter_rows"][0]["initial_requires_grad"] is False
    assert row["parameter_rows"][0]["steps_frozen"] == 1
    assert row["parameter_rows"][0]["steps_updated"] == 0
    assert all(
        p["initial_nonzero_elements"] == 0
        for row in report["outputs"]
        if row["endpoint"] == "excision_panel_apm"
        for p in row["parameter_rows"]
    )
    assert any(row["replaced_by_fixed_rule"] for row in report["additional_parameter_rows"])
    assert report["epochs_observed"] == [2]


def test_aborted_batch_removes_tensor_hooks_and_does_not_observe_evaluation():
    model = Presto(d_model=32, n_layers=1, n_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    with OutputUpdateTracker(model, optimizer) as tracker:

        def abort():
            with tracker.observe_batch(batch()):
                model(**batch().model_inputs())
                raise RuntimeError("deliberate abort")

        with pytest.raises(RuntimeError, match="deliberate abort"):
            abort()
        assert tracker.report()["batches_aborted"] == 1
        assert not tracker._tensor_hooks
        before = tracker.report()
        with torch.no_grad():
            model(**batch().model_inputs())
        assert tracker.report() == before
    assert not model._forward_hooks


def test_nonfinite_gradients_are_recorded_without_invalid_json(tmp_path):
    model = Presto(d_model=32, n_layers=1, n_heads=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    with OutputUpdateTracker(model, optimizer, output_dir=tmp_path) as tracker:
        with tracker.observe_batch(batch()):
            optimizer.zero_grad()
            (model.tcell_assay_head.assay_method_embed.weight.sum() * float("nan")).backward()
            optimizer.step()
    text = (tmp_path / "output_updates.json").read_text()
    assert "NaN" not in text
    row = cells(json.loads(text))[("tcell_panel_logits.assay_method", "ELISPOT")]
    assert row["parameter_rows"][0]["nonfinite_gradient_elements"] > 0
    assert row["parameter_rows"][0]["nonfinite_update_elements"] > 0


def test_host_evidence_survives_device_transfer():
    original = batch()
    moved = original.to("meta")
    assert moved.sample_evidence == original.sample_evidence
    assert "sample_evidence" not in moved.model_inputs()


def test_tracker_accepts_compiled_wrapper():
    raw = Presto(d_model=32, n_layers=1, n_heads=4)
    model = torch.compile(raw, backend="eager")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    try:
        with OutputUpdateTracker(model, optimizer) as tracker:
            train_epoch(
                model, [batch()], optimizer, "cpu", show_progress=False, output_tracker=tracker
            )
            assert tracker.report()["optimizer_steps"] == 1
    finally:
        torch._dynamo.reset()
