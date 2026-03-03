"""Tests for evaluation CLI metrics."""

import json
from types import SimpleNamespace

import torch


def test_evaluate_synthetic_outputs_retrieval_metrics(tmp_path, capsys):
    from presto.cli.evaluate import cmd_evaluate_synthetic
    from presto.models.presto import Presto

    checkpoint = tmp_path / "model.pt"
    model = Presto(d_model=64, n_layers=2, n_heads=4)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"d_model": 64, "n_layers": 2, "n_heads": 4},
        },
        checkpoint,
    )

    args = SimpleNamespace(
        checkpoint=str(checkpoint),
        batch_size=8,
        d_model=None,
        n_layers=None,
        n_heads=None,
        n_binding=20,
        n_elution=20,
        n_tcr=200,
        data_dir=str(tmp_path / "data"),
        seed=11,
        device="cpu",
        json=True,
    )

    code = cmd_evaluate_synthetic(args)
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "val_loss" in payload
    assert "tcell_auroc" in payload
    assert "elution_auroc" in payload
    assert "retrieval_n" in payload
    assert "recall_at_1" in payload
    assert "recall_at_5" in payload
    assert "recall_at_10" in payload

