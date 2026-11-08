"""Factorized fixed-panel T-cell outputs downstream of cellular pMHC biology."""

from __future__ import annotations

import torch
from torch import nn

from ..data.tcell_assay import AXES


class TCellAssayLayer(nn.Module):
    """Emit positivity and conditional ordered intensity for every declared track."""

    def __init__(self, input_width, tracks, embedding_dim=16, hidden_width=None):
        super().__init__()
        self.tracks = tracks
        self.embedding_dim = embedding_dim
        hidden_width = hidden_width or input_width
        self.vocabulary = {
            axis: tuple(sorted({getattr(track, axis) for track in tracks})) for axis in AXES
        }
        self.factors = nn.ModuleDict(
            {
                axis: nn.ParameterDict(
                    {value: nn.Parameter(torch.randn(embedding_dim) * 0.02) for value in values}
                )
                for axis, values in self.vocabulary.items()
            }
        )
        decoder_width = input_width + len(AXES) * embedding_dim
        self.positivity_decoder = nn.Sequential(
            nn.Linear(decoder_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )
        self.intensity_decoder = nn.Sequential(
            nn.Linear(decoder_width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, 1),
        )
        # threshold[1] is strictly greater than threshold[0] by construction.
        self.ordinal_threshold_base = nn.Parameter(torch.tensor(-0.5))
        self.ordinal_threshold_delta = nn.Parameter(torch.tensor(0.541324854612918))

    @property
    def ordinal_thresholds(self):
        first = self.ordinal_threshold_base
        return torch.stack(
            (
                first,
                first + torch.nn.functional.softplus(self.ordinal_threshold_delta) + 1e-4,
            )
        )

    def _features(self, hidden, track):
        vectors = [hidden]
        for axis in AXES:
            factor = self.factors[axis][getattr(track, axis)]
            vectors.append(factor.expand(*hidden.shape[:-1], self.embedding_dim))
        return torch.cat(vectors, dim=-1)

    def score(self, hidden, track):
        return self.positivity_decoder(self._features(hidden, track)).squeeze(-1)

    def intensity_logits(self, hidden, track):
        location = self.intensity_decoder(self._features(hidden, track)).squeeze(-1)
        return location.unsqueeze(-1) - self.ordinal_thresholds

    def forward(self, hidden):
        return {
            track.key: {
                "positivity": self.score(hidden, track),
                "conditional_intensity": self.intensity_logits(hidden, track),
            }
            for track in self.tracks
        }

    def dependencies(self, track=None, *, ordinal=True):
        selected = self.tracks if track is None else (track,)
        names = {
            f"tcell_assay_layer.positivity_decoder.{name}"
            for name, _ in self.positivity_decoder.named_parameters()
        }
        if ordinal:
            names.update(
                f"tcell_assay_layer.intensity_decoder.{name}"
                for name, _ in self.intensity_decoder.named_parameters()
            )
            names.update(
                {
                    "tcell_assay_layer.ordinal_threshold_base",
                    "tcell_assay_layer.ordinal_threshold_delta",
                }
            )
        for candidate in selected:
            names.update(
                f"tcell_assay_layer.factors.{axis}.{getattr(candidate, axis)}" for axis in AXES
            )
        return names

    def trained(self, track, evidence, *, ordinal=False):
        names = self.dependencies(track, ordinal=ordinal)
        return all(name in evidence and bool(evidence[name].any()) for name in names)
