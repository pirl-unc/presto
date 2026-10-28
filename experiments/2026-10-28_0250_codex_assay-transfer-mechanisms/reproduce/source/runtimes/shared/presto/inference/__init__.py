"""Observation prediction for current models and explicit historical API support."""

from .modular import ModularPredictor as ObservationPredictor

from .predictor import (
    Predictor,
    PresentationResult,
    RecognitionResult,
)
