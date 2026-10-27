"""Compatibility names for the single canonical Presto implementation."""

from .presto import (
    MODEL_VERSION,
    Presto,
    PrestoConfig,
    Prediction,
    ObservationOffset,
    candidates,
)

ModularPresto = Presto
ModularConfig = PrestoConfig

__all__ = [
    "MODEL_VERSION",
    "Presto",
    "PrestoConfig",
    "ModularPresto",
    "ModularConfig",
    "Prediction",
    "ObservationOffset",
    "candidates",
]
