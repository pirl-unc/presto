"""The canonical zero-prior contract must control actual loss and saved metadata."""

import argparse
import json

import pytest
import torch

from presto.cli.main import create_parser
from presto.legacy.scripts import train_iedb
from presto.legacy.scripts.train_synthetic import (
    _compute_consistency_losses,
    _regularization_config_from_args,
)


@pytest.mark.parametrize("value", [0.0, 0.37])
def test_historical_binding_prior_config_and_rejected_new_cli_flag(monkeypatch, tmp_path, value):
    flags = ["--binding-orthogonality-weight", str(value)]
    with pytest.raises(SystemExit, match="2"):
        create_parser().parse_args(["train", "--manifest", "fixture.json", *flags])
    captured = []
    monkeypatch.setattr(train_iedb, "run", captured.append)
    train_iedb.main(flags)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"train": {"unified": {"binding_orthogonality_weight": value}}}))
    resolved = train_iedb._resolve_run_args(argparse.Namespace(config=str(config)))
    for args in (captured[0], resolved):
        assert _regularization_config_from_args(args)["binding_orthogonality_weight"] == value
    with pytest.raises(SystemExit, match="2"):
        create_parser().parse_args(["train", "synthetic", "--manifest", "fixture.json", *flags])


def test_regularization_keys_are_declared_and_default_behavior_is_preserved():
    args = train_iedb._resolve_run_args(argparse.Namespace())
    effective = _regularization_config_from_args(args)
    assert effective["binding_orthogonality_weight"] == 0.01
    assert set(effective) <= set(train_iedb.IEDB_DEFAULTS)
    assert args.balanced_batches is True


@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf")])
def test_invalid_prior_fails_before_any_data_or_model_work(monkeypatch, value):
    def forbidden():
        raise AssertionError("Dependency/data work reached before weight validation")

    monkeypatch.setattr(train_iedb, "_assert_mhcseqs_importable", forbidden)
    with pytest.raises(ValueError, match="binding_orthogonality_weight"):
        train_iedb.run(argparse.Namespace(binding_orthogonality_weight=value))


def test_explicit_zero_removes_the_actual_orthogonality_loss():
    args = train_iedb._resolve_run_args(argparse.Namespace())
    regularization = _regularization_config_from_args(args)
    regularization.update({key: 0.0 for key in regularization if key.endswith("_weight")})
    outputs = {
        "latent_vecs": {
            "binding_affinity": torch.tensor([[1.0, 0.0]], requires_grad=True),
            "binding_stability": torch.tensor([[1.0, 0.0]], requires_grad=True),
        }
    }
    batch = argparse.Namespace(pep_tok=torch.zeros((1, 1), dtype=torch.long))
    assert _compute_consistency_losses(outputs, batch, regularization) == {}
    regularization["binding_orthogonality_weight"] = 0.37
    loss = _compute_consistency_losses(outputs, batch, regularization)
    assert set(loss) == {"consistency_binding_orthogonality"}
    assert loss["consistency_binding_orthogonality"].item() == pytest.approx(0.37)


@pytest.mark.parametrize("supplied", [None, "0", "0.01"])
def test_historical_explicit_default_over_config_and_rejected_new_cli_config(
    monkeypatch, tmp_path, supplied
):
    from presto.cli import main as cli_main

    config = tmp_path / "prior.json"
    config.write_text(json.dumps({"train": {"unified": {"binding_orthogonality_weight": 0.37}}}))
    flags = ["--config", str(config)]
    if supplied is not None:
        flags.append("--binding-orthogonality-weight=" + supplied)
    resolved = []
    monkeypatch.setattr(
        train_iedb, "run", lambda args: resolved.append(train_iedb._resolve_run_args(args))
    )
    train_iedb.main(flags)
    with pytest.raises(SystemExit, match="2"):
        cli_main(["train", "--manifest", "fixture.json", *flags])
    assert len(resolved) == 1
    expected = 0.37 if supplied is None else float(supplied)
    assert all(args.binding_orthogonality_weight == expected for args in resolved)
