"""Training CLI commands."""

from typing import Any

from ..scripts import train_iedb, train_synthetic


def cmd_train_synthetic(args: Any) -> int:
    """Run synthetic-data training."""
    train_synthetic.run(args)
    return 0


def cmd_train_unified(args: Any) -> int:
    """Run unified multi-source training."""
    train_iedb.run(args)
    return 0


def cmd_train_iedb(args: Any) -> int:
    """Backward-compatible alias for unified multi-source training."""
    return cmd_train_unified(args)
