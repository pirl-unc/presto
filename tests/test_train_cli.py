"""One manifest CLI contract; historical architecture flags must fail."""

import pytest

from presto.cli import train as train_cli
from presto.cli.main import create_parser


@pytest.mark.parametrize("mode", [None, "unified", "synthetic", "pretrain", "finetune"])
def test_every_training_alias_has_one_contract(mode):
    argv = ["train", *([mode] if mode else []), "--manifest", "fixture.json", "--preflight"]
    args = create_parser().parse_args(argv)
    assert args.func is train_cli.cmd_train
    assert args.observations_manifest == "fixture.json"
    assert args.modular_preflight is True
    assert args.model_kind == "shared"
    assert not hasattr(args, "latent_topology")
    assert not hasattr(args, "synthetic_negative_ratio")


@pytest.mark.parametrize(
    "flags",
    [
        ["--model-kind", "legacy"],
        ["--d_model", "256"],
        ["--epochs", "5"],
        ["--profile", "canary"],
        ["--data-source", "hitlist"],
        ["--config", "old.yml"],
        ["--latent-topology", "expanded"],
        ["--use-pcgrad"],
        ["--batch_size", "16"],
        ["--no-filter-unresolved-mhc"],
        ["--synthetic-pmhc-negative-ratio", "1"],
    ],
)
def test_old_training_flags_fail_instead_of_selecting_another_path(flags):
    with pytest.raises(SystemExit, match="2"):
        create_parser().parse_args(["train", "--manifest", "fixture.json", *flags])


def test_manifest_is_required_and_resume_is_explicit():
    with pytest.raises(SystemExit, match="2"):
        create_parser().parse_args(["train", "unified"])
    args = create_parser().parse_args(
        [
            "train",
            "--observations-manifest",
            "fixture.json",
            "--resume",
            "--run-dir",
            "run",
        ]
    )
    assert args.modular_resume and args.run_dir == "run"
    assert train_cli.cmd_train_unified is train_cli.cmd_train_synthetic is train_cli.cmd_train
