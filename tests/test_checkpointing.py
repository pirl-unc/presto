"""Tests for checkpoint serialization helpers."""

import torch


def test_save_model_checkpoint_contains_model_config(tmp_path):
    from presto.models.presto import Presto
    from presto.training.checkpointing import save_model_checkpoint

    model = Presto(
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_affinity_nM=100000.0,
        binding_midpoint_nM=800.0,
        binding_log10_scale=0.5,
    )
    path = tmp_path / "checkpoint.pt"
    save_model_checkpoint(path, model=model, epoch=2, step=10)

    payload = torch.load(path, map_location="cpu")
    assert payload["checkpoint_format"] == "presto.v2"
    assert payload["model_class"] == "presto.models.presto.Presto"
    assert payload["model_config"]["d_model"] == 64
    assert payload["model_config"]["n_layers"] == 2
    assert payload["model_config"]["n_heads"] == 4
    assert payload["model_config"]["max_affinity_nM"] == 100000.0
    assert payload["model_config"]["binding_midpoint_nM"] == 800.0
    assert payload["model_config"]["binding_log10_scale"] == 0.5


def test_load_model_from_checkpoint_uses_embedded_config(tmp_path):
    from presto.models.presto import Presto
    from presto.training.checkpointing import load_model_from_checkpoint, save_model_checkpoint

    model = Presto(
        d_model=64,
        n_layers=1,
        n_heads=4,
        max_affinity_nM=120000.0,
        binding_midpoint_nM=1200.0,
        binding_log10_scale=0.45,
    )
    path = tmp_path / "checkpoint.pt"
    save_model_checkpoint(path, model=model)

    loaded, payload = load_model_from_checkpoint(path, map_location="cpu")
    assert loaded.d_model == 64
    assert loaded.max_affinity_nM == 120000.0
    assert loaded.binding_midpoint_nM == 1200.0
    assert loaded.binding_log10_scale == 0.45
    assert "model_state_dict" in payload
