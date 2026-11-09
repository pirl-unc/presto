"""One nonlinear measurement head over biology and separate categorical vectors."""

import torch
from torch import nn


AFFINITY_ENDPOINTS = frozenset(("ic50", "kd", "ec50", "kd_via_ic50", "kd_via_ec50"))
TRACKED_ENDPOINTS = AFFINITY_ENDPOINTS | {"qualitative_binding"}
AXES = ("endpoint", "preparation", "mechanism", "detector")
OUTPUT_POLARITY = {"qualitative_binding": -1}


class AssayMeasurementLayer(nn.Module):
    def __init__(self, width, endpoints, tracks, embedding_dim=16):
        super().__init__()
        self.tracks = tracks
        self.embedding_dim = embedding_dim
        self.vocabulary = {
            axis: tuple(
                sorted(
                    set(endpoints)
                    if axis == "endpoint"
                    else {getattr(track, axis) for track in tracks}
                )
            )
            for axis in AXES
        }
        # Individually named rows preserve transfer/freeze semantics when quantities
        # are added. Unknown slots are zeros, not untrained embedding parameters.
        self.factors = nn.ModuleDict(
            {
                axis: nn.ParameterDict(
                    {value: nn.Parameter(torch.randn(embedding_dim) * 0.02) for value in values}
                )
                for axis, values in self.vocabulary.items()
            }
        )
        self.decoder = nn.Sequential(
            nn.Linear(width + len(AXES) * embedding_dim, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def score(self, hidden, endpoint, track=None):
        vectors = [hidden]
        for axis in AXES:
            value = endpoint if axis == "endpoint" else getattr(track, axis, None)
            vector = (
                self.factors[axis][value]
                if value is not None
                else hidden.new_zeros(self.embedding_dim)
            )
            vectors.append(vector.expand(*hidden.shape[:-1], self.embedding_dim))
        value = self.decoder(torch.cat(vectors, dim=-1)).squeeze(-1)
        # The shared decoder is concentration-oriented: lower means stronger.
        # A positive qualitative-binding label instead requires a higher logit.
        return OUTPUT_POLARITY.get(endpoint, 1) * value

    def forward(self, hidden, endpoint):
        return {
            track.key: self.score(hidden, endpoint, track)
            for track in self.tracks
            if track.endpoint == endpoint
        }

    def dependencies(self, endpoint):
        names = {f"assay_layer.decoder.{name}" for name, _ in self.decoder.named_parameters()}
        names.add(f"assay_layer.factors.endpoint.{endpoint}")
        for track in self.tracks:
            if track.endpoint == endpoint:
                names.update(f"assay_layer.factors.{axis}.{getattr(track, axis)}" for axis in AXES)
        return names

    def trained(self, track, evidence):
        names = {f"assay_layer.decoder.{name}" for name, _ in self.decoder.named_parameters()}
        names.update(f"assay_layer.factors.{axis}.{getattr(track, axis)}" for axis in AXES)
        return all(name in evidence and bool(evidence[name].any()) for name in names)
