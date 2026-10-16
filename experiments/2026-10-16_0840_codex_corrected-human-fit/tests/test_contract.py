"""Exercise experiment evidence safeguards without fitting a biological model."""

import importlib.util
import inspect
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

CODE = Path(__file__).resolve().parents[1] / "code"


@pytest.fixture
def modules(monkeypatch):
    loaded = {}
    for name in ("baselines", "diagnostic"):
        spec = importlib.util.spec_from_file_location(name, CODE / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return SimpleNamespace(**loaded)


def rows():
    return pd.DataFrame(
        {
            "peptide": ["ACDEFGHI", "KLMNPQRST"] * 30,
            "native_mhc_class": ["I", "II"] * 30,
            "source_partition": ["train"] * 60,
            "qualifier": [0] * 60,
            "target_log10_nM": [1.0, 3.0] * 30,
        }
    )


def test_baselines_reject_heldout_and_ignore_censored_targets(modules):
    frame = rows()
    state = modules.baselines.fit_baselines(frame)
    heldout = frame.copy()
    heldout.loc[0, "source_partition"] = "val"
    with pytest.raises(ValueError, match="training observations only"):
        modules.baselines.fit_baselines(heldout)
    censored = frame.iloc[:1].copy()
    censored["qualifier"], censored["target_log10_nM"] = 1, 1000.0
    assert modules.baselines.fit_baselines(pd.concat([frame, censored])) == state
    assert state["intercept"] == 2.0
    prediction = modules.baselines.predict_baselines(frame, state)
    assert (
        np.mean((prediction["length_composition_class_ridge"] - frame.target_log10_nM) ** 2) < 0.01
    )
    assert np.array_equal(prediction["train_mean"], np.full(60, 2.0))


def test_metrics_censor_direction_ambiguous_bounds_and_undefined(modules):
    metrics = modules.baselines.metrics([2.0, 4.0, 2.0, 4.0], [1.0, 5.0, 5.0, 1.0], [-1, 1, 1, -1])
    assert metrics["rmse"] is None
    assert metrics["censor_aware_mse"] == 0.0
    assert metrics["censor_satisfaction_fraction"] == 1.0
    binary = metrics["binding_500nM"]
    assert binary["n_eligible_observations"] == 2
    assert binary["n_ambiguous_bounds"] == 2
    assert binary["accuracy"] == 1.0
    violated = modules.baselines.metrics([2.0, 4.0], [3.0, 3.0], [-1, 1])
    assert violated["censor_aware_mse"] == 1.0
    assert violated["censor_satisfaction_fraction"] == 0.0
    constant = modules.baselines.metrics([1.0, 2.0, 4.0], [3.0, 3.0, 3.0], [0, 0, 0])
    assert constant["spearman"] is None and constant["pearson"] is None
    assert constant["binding_500nM"]["precision"] is None
    json.dumps(constant, allow_nan=False)


def test_bootstrap_keeps_peptide_clusters_and_is_reproducible(modules):
    frame = rows()
    values = {"presto": frame.target_log10_nM.to_numpy() + 0.5, "train_mean": np.full(60, 2.0)}
    first = modules.baselines.paired_rmse_bootstrap(frame, values, n_resamples=50)
    second = modules.baselines.paired_rmse_bootstrap(frame, values, n_resamples=50)
    assert first == second
    assert first["n_peptides"] == 2 and first["n_exact_observations"] == 60
    assert first["paired_presto_minus_baseline_rmse_95_intervals"]["train_mean"] == [-0.5, -0.5]


def test_prediction_join_rejects_changed_identity_target_and_censoring(modules, tmp_path):
    expected = pd.DataFrame(
        {
            "sample_id": ["s1", "s2"],
            "evidence_row_id": ["o1", "o2"],
            "peptide": ["ACDE", "FGHI"],
            "source_partition": ["val", "val"],
            "target_log10_nM": [2.0, 3.0],
            "qualifier": [0, 1],
        }
    )
    raw = expected.rename(columns={"target_log10_nM": "y_true"}).copy()
    raw["task"], raw["y_pred"] = "binding_ic50", [2.2, 3.1]
    path = tmp_path / "prediction.csv"
    raw.iloc[::-1].to_csv(path, index=False)
    assert np.allclose(modules.baselines.verified_predictions(path, expected), [2.2, 3.1])
    for column, bad in (
        ("sample_id", "s2"),
        ("evidence_row_id", "o2"),
        ("source_partition", "train"),
        ("y_true", 200.0),
        ("qualifier", -1),
    ):
        changed = raw.copy()
        changed.loc[0, column] = bad
        changed.to_csv(path, index=False)
        with pytest.raises(ValueError):
            modules.baselines.verified_predictions(path, expected)


def test_diagnostic_restores_all_cpu_rngs_on_failure(modules):
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
    with pytest.raises(RuntimeError):
        with modules.diagnostic.preserve_rng():
            random.random()
            np.random.random()
            torch.rand(4)
            raise RuntimeError("exercise exceptional exit")
    assert random.getstate() == before[0]
    after_np = np.random.get_state()
    assert after_np[0] == before[1][0] and np.array_equal(after_np[1], before[1][1])
    assert after_np[2:] == before[1][2:]
    assert torch.equal(torch.random.get_rng_state(), before[2])


def test_training_hook_preserves_signature_and_rejects_heldout(modules, tmp_path):
    def epoch(model, train_loader, optimizer, device, *, output_tracker=None):
        raise AssertionError("Optimization must not run on invalid diagnostic membership")

    trainer = SimpleNamespace(train_epoch=epoch)
    sample = SimpleNamespace(
        source_endpoint=modules.baselines.ENDPOINT, sample_id="s1", source_partition="val"
    )
    modules.diagnostic.install(trainer, tmp_path, ["s1"], {})
    assert inspect.signature(trainer.train_epoch) == inspect.signature(epoch)
    with pytest.raises(ValueError, match="frozen training IC50 cohort"):
        trainer.train_epoch(None, SimpleNamespace(dataset=[sample]), None, "cpu")
