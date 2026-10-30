"""Fixed factorized output tracks on the common biological representation."""

import math

import torch
from torch import nn


class AssayMeasurementLayer(nn.Module):
    def __init__(self, width, tracks, mode, rank=8):
        super().__init__()
        self.tracks = tracks
        self.mode = mode
        self.vocabulary = {
            axis: tuple(sorted({getattr(track, axis) for track in tracks}))
            for axis in ("endpoint", "preparation", "mechanism", "detector")
        }
        if mode == "calibration":
            self.offset = nn.Parameter(torch.zeros(len(tracks)))
        elif mode == "factorized":
            self.factors = nn.ModuleDict(
                {axis: nn.Embedding(len(values), rank) for axis, values in self.vocabulary.items()}
            )
            self.projection = nn.Linear(width, rank, bias=True)
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
        elif mode != "pooled":
            raise ValueError("unsupported assay measurement mode")

    def forward(self, hidden, endpoint):
        projected = self.projection(hidden) if self.mode == "factorized" else None
        values = {}
        for i, track in enumerate(self.tracks):
            if track.endpoint != endpoint:
                continue
            if self.mode == "pooled":
                residual = hidden.new_zeros(hidden.shape[:-1])
            elif self.mode == "calibration":
                residual = self.offset[i].expand(hidden.shape[:-1])
            else:
                embedding = sum(
                    self.factors[axis].weight[self.vocabulary[axis].index(getattr(track, axis))]
                    for axis in self.vocabulary
                )
                residual = 0.1 * (projected * embedding).sum(-1) / math.sqrt(projected.shape[-1])
            values[track.key] = residual
        return values

    def trained(self, track, evidence):
        """Require actual gradient-and-update evidence for every requested factor."""
        if self.mode == "pooled":
            return True
        if self.mode == "calibration":
            mask = evidence.get("assay_layer.offset")
            return mask is not None and bool(mask[self.tracks.index(track)])
        for axis in self.vocabulary:
            mask = evidence.get(f"assay_layer.factors.{axis}.weight")
            index = self.vocabulary[axis].index(getattr(track, axis))
            if mask is None or not bool(mask[index].any()):
                return False
        return all(
            key in evidence and bool(evidence[key].any())
            for key in ("assay_layer.projection.weight", "assay_layer.projection.bias")
        )
