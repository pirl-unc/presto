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


def test_load_model_from_checkpoint_drops_legacy_dead_module_keys(tmp_path):
    from presto.models.presto import Presto
    from presto.training.checkpointing import load_model_from_checkpoint

    model = Presto(d_model=64, n_layers=1, n_heads=4)
    state = model.state_dict()
    state["presentation.w_proc"] = torch.tensor(0.8)
    payload = {
        "checkpoint_format": "presto.v2",
        "checkpoint_format_version": 2,
        "model_class": "presto.models.presto.Presto",
        "model_config": {"d_model": 64, "n_layers": 1, "n_heads": 4},
        "model_state_dict": state,
    }
    path = tmp_path / "legacy.pt"
    torch.save(payload, path)

    loaded, raw = load_model_from_checkpoint(path, map_location="cpu")

    assert isinstance(loaded, Presto)
    assert "model_state_dict" in raw


def test_load_model_from_checkpoint_remaps_legacy_head_keys(tmp_path):
    from presto.models.presto import Presto
    from presto.training.checkpointing import load_model_from_checkpoint

    model = Presto(d_model=64, n_layers=1, n_heads=4)
    state = model.state_dict()
    state["processing_class1_head.weight"] = state.pop(
        "class1_processing_predictor.head.weight"
    )
    state["processing_class1_head.bias"] = state.pop(
        "class1_processing_predictor.head.bias"
    )
    state["presentation_class2_latent_head.weight"] = state.pop(
        "class2_presentation_predictor.head.weight"
    )
    state["presentation_class2_latent_head.bias"] = state.pop(
        "class2_presentation_predictor.head.bias"
    )
    state["binding_probe_mix_logit"] = state.pop(
        "affinity_predictor.binding_probe_mix_logit"
    )
    state["kd_assay_bias_scale"] = state.pop(
        "affinity_predictor.kd_assay_bias_scale"
    )

    payload = {
        "checkpoint_format": "presto.v2",
        "checkpoint_format_version": 2,
        "model_class": "presto.models.presto.Presto",
        "model_config": {"d_model": 64, "n_layers": 1, "n_heads": 4},
        "model_state_dict": state,
    }
    path = tmp_path / "legacy_heads.pt"
    torch.save(payload, path)

    loaded, _ = load_model_from_checkpoint(path, map_location="cpu")
    assert isinstance(loaded, Presto)
    assert torch.allclose(
        loaded.class1_processing_predictor.head.weight,
        model.class1_processing_predictor.head.weight,
    )
    assert torch.allclose(
        loaded.class2_presentation_predictor.head.bias,
        model.class2_presentation_predictor.head.bias,
    )
    assert torch.allclose(
        loaded.affinity_predictor.binding_probe_mix_logit,
        model.affinity_predictor.binding_probe_mix_logit,
    )
