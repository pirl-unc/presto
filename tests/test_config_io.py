"""Tests for CLI/config merge helpers."""

from argparse import Namespace


def test_merge_namespace_with_config_uses_config_for_defaults():
    from presto.training.config_io import merge_namespace_with_config

    args = Namespace(epochs=10, batch_size=16, lr=1e-4)
    defaults = {"epochs": 10, "batch_size": 16, "lr": 1e-4}
    config = {"epochs": 3, "batch_size": 8}

    merged = merge_namespace_with_config(args, defaults, config)
    assert merged.epochs == 3
    assert merged.batch_size == 8
    assert merged.lr == 1e-4


def test_merge_namespace_with_config_cli_values_win():
    from presto.training.config_io import merge_namespace_with_config

    args = Namespace(epochs=2, batch_size=16, lr=1e-4)
    defaults = {"epochs": 10, "batch_size": 16, "lr": 1e-4}
    config = {"epochs": 3, "batch_size": 8, "lr": 5e-4}

    merged = merge_namespace_with_config(args, defaults, config)
    assert merged.epochs == 2
    assert merged.batch_size == 8
    assert merged.lr == 5e-4


def test_pick_train_section_prefers_nested_train_subcommand():
    from presto.training.config_io import pick_train_section

    config = {
        "train": {
            "synthetic": {"epochs": 2, "batch_size": 4},
            "shared_flag": True,
        }
    }
    section = pick_train_section(config, "synthetic")
    assert section["epochs"] == 2
    assert section["batch_size"] == 4
