"""Artificial software checks for the registered fixed-update comparison."""

from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch

from presto.data.modular import AA, MHCInput, Observation, TargetScale
from presto.data.modular_sources import (
    MANIFEST_VERSION,
    ObservationIndex,
    file_sha256,
    write_observations,
)
from presto.models.modular import ModularConfig, ModularPresto
from presto.training.modular import load_checkpoint, save_checkpoint, train_step
from presto.training.modular_run import fit_index


CODE = (
    Path(__file__).resolve().parents[1]
    / "experiments/2026-10-26_1929_codex_mhc2-censor-refinement/code"
)


@pytest.fixture
def scripts(monkeypatch):
    monkeypatch.syspath_prepend(str(CODE))
    modules = {}
    for name in ("launch", "verify", "continue"):
        spec = importlib.util.spec_from_file_location(name, CODE / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    return modules


def test_source_checker_preserves_bounds_and_rejects_changed_provenance(scripts):
    original = Observation(
        "fixture",
        "software",
        "ic50",
        AA,
        (MHCInput(AA, AA[::-1], "I"),),
        value=100,
        unit="nM",
        qualifier=">",
        split="test",
        group_id="fixture-group",
        provenance={"original_measurement_id": "only"},
    ).to_dict()
    # The source predates the optional context fields; only absent empty fields
    # may be removed for compatibility.
    for key in (
        "processing_pathways",
        "pathway_overrides",
        "interventions",
        "intervention_overrides",
    ):
        original["context"].pop(key)
    observed = Observation.from_dict(original).to_dict()
    row = dict(
        observation_id="fixture",
        observation=observed,
        qualifier=">",
        raw_value=100,
        peptide=AA,
        group_id="fixture-group",
        mhc_classes=["I"],
        target=2.0,
        prediction=1.0,
        loss=0.25,
    )
    check = scripts["verify"].verify_rows
    check([row], {"fixture": original}, "test", 2.0)
    with pytest.raises(AssertionError):
        check([{**row, "loss": 0.0}], {"fixture": original}, "test", 2.0)
    changed = deepcopy(row)
    changed["observation"]["provenance"]["original_measurement_id"] = "other"
    with pytest.raises(AssertionError):
        check([changed], {"fixture": original}, "test", 2.0)
    changed = deepcopy(row)
    changed["observation"]["context"]["interventions"] = {"invented": {"state": "present"}}
    with pytest.raises(AssertionError):
        check([changed], {"fixture": original}, "test", 2.0)


def test_better_class_ii_rmse_cannot_hide_bound_or_class_i_failure(scripts):
    metrics = {
        "by_class": {
            cls: {
                "rmse": 1.0,
                "binding_at_500_nM": {
                    "auroc": 0.8,
                    "balanced_accuracy": 0.7,
                    "recall": 0.6,
                },
            }
            for cls in ("I", "II")
        },
        "class_ii_bounds": {"mean_loss": 1.0, "satisfaction": 0.2},
    }
    improved = deepcopy(metrics)
    improved["by_class"]["II"]["rmse"] = 0.9
    improved["class_ii_bounds"] = {"mean_loss": 0.8, "satisfaction": 0.3}
    screen = scripts["verify"].primary_screen
    assert screen(metrics, improved)["passed"]
    missing_bounds = deepcopy(improved)
    missing_bounds["class_ii_bounds"] = deepcopy(metrics["class_ii_bounds"])
    assert not screen(metrics, missing_bounds)["gates"]["class_ii_bounds"]
    class_i_regression = deepcopy(improved)
    class_i_regression["by_class"]["I"]["rmse"] = 1.1
    assert not screen(metrics, class_i_regression)["gates"]["retained_class_i"]


def test_controller_stops_after_failure_and_refuses_reentry(scripts, monkeypatch, tmp_path):
    controller = scripts["continue"]
    state = tmp_path / "continuation-state.json"
    monkeypatch.setattr(controller, "FAMILY", tmp_path)
    monkeypatch.setattr(controller, "STATE", state)
    calls = []

    class FailedChild:
        pid = 123

        def __init__(self, command, **kwargs):
            calls.append(command)

        def wait(self):
            return 17

    monkeypatch.setattr(controller.subprocess, "Popen", FailedChild)
    with pytest.raises(RuntimeError, match="no next condition"):
        controller.main()
    assert len(calls) == 1
    assert json.loads(state.read_text())["status"] == "stopped_on_failure"
    with pytest.raises(FileExistsError):
        controller.main()
    assert len(calls) == 1


def test_fixed_updates_preserve_parent_scale_and_record_partial_epochs(tmp_path):
    config = ModularConfig(endpoints=("ic50",), mhc_classes=("I",), dropout=0.0)
    model = ModularPresto(config)
    scale = {"ic50": TargetScale(2.0, 1.0, 2)}

    def row(i, **kwargs):
        values = dict(
            observation_id=f"fixture-{i}",
            source="software",
            endpoint="ic50",
            peptide=AA[i:] + AA[:i],
            mhc=(MHCInput(AA, AA[::-1], "I"),),
            value=10.0 if i % 2 else 1000.0,
            unit="nM",
            group_id=f"g{i}",
            provenance={"original_measurement_id": f"m{i}"},
        )
        values.update(kwargs)
        return Observation(**values)

    train_step(model, torch.optim.AdamW(model.parameters(), lr=0.001), [row(0), row(1)], scale)
    parent = tmp_path / "parent.pt"
    save_checkpoint(parent, model, scale)
    for with_bound, exposure_count in ((False, 10), (True, 9)):
        rows = [row(i) for i in range(6)]
        if with_bound:
            rows.append(row(6, value=1e8, qualifier=">"))
        rows.extend([row(7, split="validation"), row(8, split="test")])
        source = write_observations(tmp_path / f"source-{with_bound}.jsonl", rows)
        manifest = dict(
            schema=MANIFEST_VERSION,
            model=asdict(config),
            pretrained={"path": str(parent), "sha256": file_sha256(parent), "freeze": []},
            training={
                "seed": 101,
                "batch_size": 2,
                "max_steps": 5,
                "max_epochs": 3,
                "min_epochs": 3,
                "patience": 100,
                "identity_weight": 0,
            },
            sources=[
                {
                    **source,
                    "source": "software",
                    "population": "artificial",
                    "label_policy": "fixture values",
                    "split_policy": "distinct artificial peptides",
                }
            ],
        )
        path = tmp_path / f"manifest-{with_bound}.json"
        path.write_text(json.dumps(manifest))
        output = tmp_path / f"fit-{with_bound}"
        result = fit_index(ObservationIndex(path), output)
        _, restored_scale, final = load_checkpoint(output / "last.pt")
        initial = torch.load(output / "initial.pt", weights_only=True)
        assert restored_scale == scale
        assert final["metadata"]["steps"] == 5
        assert result["training_exposures"] == {"ic50": exposure_count}
        assert all(
            torch.equal(initial["state_dict"][key], value)
            for key, value in model.state_dict().items()
        )
        history = json.loads((output / "history.json").read_text())
        assert len(history) == 2
        assert history[0]["complete_epoch"] and not history[1]["complete_epoch"]
