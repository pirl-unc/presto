"""Tests for checkpoint serialization helpers."""

import pytest
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


def test_legacy_dead_module_keys_are_no_longer_dropped(tmp_path):
    """The `presentation.` key-dropping pass is gone, deliberately.

    `training/checkpointing.py` carried its own migration layer --
    `_drop_legacy_dead_keys` and `_migrate_chain_type_heads` -- separate from
    the one in `Presto._load_from_state_dict`. Removing only the latter left
    checkpoint compat half-alive in a second file, so both are gone now and a
    stale key is an error like any other.
    """
    from presto.models.presto import Presto
    from presto.training.checkpointing import load_model_from_checkpoint

    model = Presto(d_model=64, n_layers=1, n_heads=4)
    state = dict(model.state_dict())
    state["presentation.weight"] = torch.zeros(4, 4)

    payload = {
        "checkpoint_format": "presto.v2",
        "checkpoint_format_version": 2,
        "model_class": "presto.models.presto.Presto",
        "model_config": {"d_model": 64, "n_layers": 1, "n_heads": 4},
        "model_state_dict": state,
    }
    path = tmp_path / "legacy_dead.pt"
    torch.save(payload, path)

    with pytest.raises(RuntimeError) as excinfo:
        load_model_from_checkpoint(path, map_location="cpu")
    assert "presentation.weight" in str(excinfo.value)


def test_renamed_head_keys_are_no_longer_remapped(tmp_path):
    """The legacy key remap is gone, deliberately.

    A checkpoint written before the head modules were renamed used to be
    rewritten on load: `processing_class1_head.*` became
    `class1_processing_predictor.head.*`, and so on for eight prefixes. That
    layer was removed with the rest of the checkpoint-compat machinery, because
    silently reshaping weights across a semantic change produces a model that
    loads cleanly and means something different.

    A stale key is now an error, which is the honest outcome.
    """
    from presto.models.presto import Presto
    from presto.training.checkpointing import load_model_from_checkpoint

    model = Presto(d_model=64, n_layers=1, n_heads=4)
    state = model.state_dict()
    state["processing_class1_head.weight"] = state.pop("class1_processing_predictor.head.weight")
    state["processing_class1_head.bias"] = state.pop("class1_processing_predictor.head.bias")

    payload = {
        "checkpoint_format": "presto.v2",
        "checkpoint_format_version": 2,
        "model_class": "presto.models.presto.Presto",
        "model_config": {"d_model": 64, "n_layers": 1, "n_heads": 4},
        "model_state_dict": state,
    }
    path = tmp_path / "legacy_heads.pt"
    torch.save(payload, path)

    with pytest.raises(RuntimeError) as excinfo:
        load_model_from_checkpoint(path, map_location="cpu")
    message = str(excinfo.value)
    assert "processing_class1_head" in message or "class1_processing_predictor" in message
