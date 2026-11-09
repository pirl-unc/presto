"""Manifest trainer facade; the optimizer and model live in one implementation."""

from ..data.modular_sources import ObservationIndex
from .modular import train_step
from .modular_run import fit_index


class Trainer:
    """Execute a normalized manifest through the canonical trainer."""

    def __init__(self, manifest, *, device="cpu"):
        self.index = (
            manifest if isinstance(manifest, ObservationIndex) else ObservationIndex(manifest)
        )
        self.device = device

    def fit(self, output_dir, *, resume=False):
        return fit_index(self.index, output_dir, device=self.device, resume=resume)


__all__ = ["Trainer", "fit_index", "train_step"]
