"""Assay context encoder and integration modules.

Factorized embedding of assay metadata (type, prep, geometry, readout) →
context vector used by heads to adjust predictions per-assay.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from presto.data.vocab import (
    BINDING_ASSAY_TYPES,
    BINDING_ASSAY_PREP,
    BINDING_ASSAY_GEOMETRY,
    BINDING_ASSAY_READOUT,
)


class AssayContextEncoder(nn.Module):
    """Factorized assay metadata → context vector.

    Embeds type, prep, geometry, readout separately, concatenates, then
    projects through a small MLP.

    When ``repr_dim > 0``, the encoder also accepts a detached binding
    logit scalar and a detached molecular representation vector. These
    let the assay bias depend on *what* is being measured (content-
    conditioned) without giving the assay pathway gradient access to the
    encoder or binding head.
    """

    def __init__(self, factor_dim: int = 8, ctx_dim: int = 32, repr_dim: int = 0):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.repr_dim = repr_dim
        self.type_embed = nn.Embedding(len(BINDING_ASSAY_TYPES), factor_dim)
        self.prep_embed = nn.Embedding(len(BINDING_ASSAY_PREP), factor_dim)
        self.geom_embed = nn.Embedding(len(BINDING_ASSAY_GEOMETRY), factor_dim)
        self.read_embed = nn.Embedding(len(BINDING_ASSAY_READOUT), factor_dim)

        in_dim = 4 * factor_dim
        if repr_dim > 0:
            in_dim += 1 + repr_dim  # binding logit scalar + mol repr

        self.proj = nn.Sequential(
            nn.Linear(in_dim, ctx_dim),
            nn.GELU(),
            nn.Linear(ctx_dim, ctx_dim),
        )

    def forward(
        self,
        assay_type_idx: torch.Tensor,
        assay_prep_idx: torch.Tensor,
        assay_geometry_idx: torch.Tensor,
        assay_readout_idx: torch.Tensor,
        binding_logit: Optional[torch.Tensor] = None,
        mol_repr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return (B, ctx_dim) assay context vector.

        Args:
            binding_logit: (B,) scalar binding prediction (detached).
            mol_repr: (B, repr_dim) mean-pooled pep/mhc representations (detached).
        """
        parts = [
            self.type_embed(assay_type_idx),
            self.prep_embed(assay_prep_idx),
            self.geom_embed(assay_geometry_idx),
            self.read_embed(assay_readout_idx),
        ]
        if self.repr_dim > 0:
            if binding_logit is not None:
                parts.append(binding_logit.unsqueeze(-1))
            if mol_repr is not None:
                parts.append(mol_repr)
        cat = torch.cat(parts, dim=-1)
        return self.proj(cat)


# ---------------------------------------------------------------------------
# Integration modules — modify head predictions based on assay context
# ---------------------------------------------------------------------------

class AdditiveIntegration(nn.Module):
    """Additive bias: pred = base + bias(ctx)."""

    def __init__(self, ctx_dim: int):
        super().__init__()
        self.bias = nn.Linear(ctx_dim, 1)

    def forward(self, base: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return base + self.bias(ctx).squeeze(-1)


class AffineIntegration(nn.Module):
    """Affine: pred = softplus(scale(ctx)) * base + bias(ctx)."""

    def __init__(self, ctx_dim: int):
        super().__init__()
        self.scale = nn.Linear(ctx_dim, 1)
        self.bias = nn.Linear(ctx_dim, 1)

    def forward(self, base: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        s = F.softplus(self.scale(ctx).squeeze(-1))
        b = self.bias(ctx).squeeze(-1)
        return s * base + b


class D1AffineIntegration(nn.Module):
    """D1-affine: shifts/scales bin *centers* per assay context.

    adjusted_centers = softplus(scale(ctx)) * centers + bias(ctx)
    Logits unchanged. For censored thresholds, must also adjust edges.
    """

    def __init__(self, ctx_dim: int):
        super().__init__()
        self.scale = nn.Linear(ctx_dim, 1)
        self.bias = nn.Linear(ctx_dim, 1)

    def forward(
        self,
        centers: torch.Tensor,
        edges: torch.Tensor,
        ctx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return adjusted centers and edges.

        Args:
            centers: (K,) canonical bin centers.
            edges: (K+1,) canonical bin edges.
            ctx: (B, ctx_dim) assay context.

        Returns:
            dict with ``centers`` (B, K) and ``edges`` (B, K+1).
        """
        s = F.softplus(self.scale(ctx))   # (B, 1)
        b = self.bias(ctx)                # (B, 1)
        adj_centers = s * centers.unsqueeze(0) + b  # (B, K)
        adj_edges = s * edges.unsqueeze(0) + b      # (B, K+1)
        return {"centers": adj_centers, "edges": adj_edges}


class D2LogitIntegration(nn.Module):
    """D2-logit: per-bin logit shift from assay context.

    adjusted_logits = base_logits + shift(ctx)
    Bins stay fixed.
    """

    def __init__(self, ctx_dim: int, n_bins: int):
        super().__init__()
        self.shift = nn.Linear(ctx_dim, n_bins)

    def forward(self, logits: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """Return adjusted logits (B, K)."""
        return logits + self.shift(ctx)


INTEGRATION_CLASSES = {
    "additive": AdditiveIntegration,
    "affine": AffineIntegration,
    "d1_affine": D1AffineIntegration,
    "d2_logit": D2LogitIntegration,
}
