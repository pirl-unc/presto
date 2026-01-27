"""Training CLI commands."""

from typing import Any

from ..scripts import train_curriculum, train_synthetic


def cmd_train_synthetic(args: Any) -> int:
    """Run synthetic-data training."""
    train_synthetic.run(args)
    return 0


def cmd_train_curriculum(args: Any) -> int:
    """Run curriculum training (synthetic curriculum demo)."""
    train_curriculum.run(args)
    return 0
