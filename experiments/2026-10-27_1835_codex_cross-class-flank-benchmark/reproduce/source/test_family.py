"""Software-only fixtures for orchestration; never score biological holdouts."""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

import dashboard
import launch
import support
from presto.data.modular import AA, MHCInput, Observation, TargetScale
from presto.models import Presto, PrestoConfig
from presto.training.modular import train_step
from presto.training.modular_evaluation import summarize


def fixture():
    return Observation(
        observation_id="software",
        source="fixture",
        endpoint="ic50",
        peptide=AA[:10],
        mhc=(MHCInput(AA, AA[::-1], "I"),),
        value=100.0,
        unit="nM",
        group_id="software",
        split="train",
    )


def test_instrumentation_preserves_step_and_all_weights(tmp_path):
    torch.manual_seed(31)
    a = Presto(PrestoConfig(mhc_classes=("I",), dropout=0))
    b = Presto(a.config)
    b.load_state_dict(a.state_dict())
    scales = {"ic50": TargetScale(1.0, 1.0, 2)}
    plain = train_step(a, torch.optim.AdamW(a.parameters()), [fixture()], scales)
    runner = SimpleNamespace(train_step=train_step)
    support.instrument_steps(runner, tmp_path / "progress.json")
    observed = runner.train_step(b, torch.optim.AdamW(b.parameters()), [fixture()], scales)
    assert plain["loss"] == observed["loss"]
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])
    assert json.loads((tmp_path / "progress.json").read_text())["steps"] == 1
    view, contract = support.diagnostic_view(b)
    assert contract["all_weights_preserved"] and not contract["new_parameters"]
    for key, value in b.state_dict().items():
        assert torch.equal(value, view.state_dict()[key])
    assert view([replace(fixture(), mhc=(MHCInput(AA, AA[::-1], "II"),))])[0].value.isfinite()


def test_failed_child_stops_controller_and_preserves_failure(tmp_path, monkeypatch):
    (tmp_path / "reproduce").mkdir()
    monkeypatch.setattr(launch, "FAMILY", tmp_path)
    calls = []

    def child(action, label, condition=None):
        calls.append((action, label, condition))
        if action == "fit":
            raise RuntimeError("fixture failure")

    monkeypatch.setattr(launch, "child", child)
    with pytest.raises(RuntimeError, match="fixture failure"):
        launch.run_all()
    receipt = support.read(tmp_path / "reproduce" / "controller.json")
    assert receipt["status"] == "failed" and receipt["completed"] == []
    assert len([x for x in calls if x[0] == "fit"]) == 1
    with pytest.raises(ValueError, match="already attempted"):
        launch.run_all()


def test_live_snapshot_never_invents_cross_class_or_test_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "FAMILY", tmp_path)
    monkeypatch.setattr(dashboard, "RAW", tmp_path / "raw")
    initial = dashboard.snapshot()["rows"][0]
    assert initial["cells"]["II"]["validation"] is None
    support.atomic(
        tmp_path / "raw" / "shared-I-s42" / "history.json",
        [
            dict(
                epoch=1,
                steps=10,
                validation={"ic50": {"by_class": {"I": {"spearman": 0.2, "rmse": 1.0}}}},
            )
        ],
    )
    changed = dashboard.snapshot()["rows"][0]
    assert changed["cells"]["I"]["validation"]["spearman"] == 0.2
    assert changed["cells"]["I"]["test"] is None
    assert changed["cells"]["II"]["validation"] is None


def test_training_mean_baseline_keeps_censor_loss_and_metrics():
    records = [
        dict(
            endpoint="ic50",
            observation_id=str(i),
            qualifier=q,
            target=y,
            prediction=9.0,
            loss=99.0,
            mhc_classes=["I"],
            group_id="fixture",
        )
        for i, (q, y) in enumerate([("=", 1.0), ("<=", 2.0), (">=", 4.0)])
    ]
    metrics, values = launch.summarize_mean(records, 3.0, 2.0)
    assert [x["loss"] for x in values] == [1.0, 0.25, 0.25]
    assert metrics == summarize(values)
