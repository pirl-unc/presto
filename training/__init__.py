"""The common observation-based training API for every new Presto run."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "Trainer",
    "train_step",
    "fit_index",
    "preflight",
    "run_manifest",
    "save_checkpoint",
    "load_checkpoint",
    "initialize_from_checkpoint",
]

_EXPORTS = {
    "Trainer": (".trainer", "Trainer"),
    "train_step": (".modular", "train_step"),
    "save_checkpoint": (".modular", "save_checkpoint"),
    "load_checkpoint": (".modular", "load_checkpoint"),
    "initialize_from_checkpoint": (".modular", "initialize_from_checkpoint"),
    "fit_index": (".modular_run", "fit_index"),
    "preflight": (".modular_run", "preflight"),
    "run_manifest": (".modular_run", "run_manifest"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
