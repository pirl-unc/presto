"""Abstract base class for binding affinity prediction heads."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


class AffinityHead(nn.Module, ABC):
    """Base interface for all BA output heads.

    Every head:
    - Receives encoder representation h (B, D) and assay embedding (B, ctx_dim).
    - Produces at least ``pred_ic50_nM`` in forward().
    - Owns its own loss (target space may differ between heads).
    - Optionally exposes predicted distribution.
    """

    @abstractmethod
    def forward(
        self,
        h: torch.Tensor,
        assay_emb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict binding affinity.

        Returns dict with at least ``pred_ic50_nM`` (B,).
        """
        ...

    @abstractmethod
    def compute_loss(
        self,
        h: torch.Tensor,
        assay_emb: torch.Tensor,
        ic50_nM: torch.Tensor,
        qual: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute loss in the head's native target space.

        Args:
            h: encoder output (B, D)
            assay_emb: assay context embedding (B, ctx_dim)
            ic50_nM: raw IC50 values in nM (B,)
            qual: censor qualifier -1/0/1 (B,)
            mask: valid-sample mask (B,)

        Returns:
            (scalar loss, metrics dict)
        """
        ...

    def predict_distribution(
        self,
        h: torch.Tensor,
        assay_emb: torch.Tensor,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Return distributional prediction (only for distributional heads).

        Returns None for regression heads. For distributional heads returns
        dict with ``probs``, ``bin_centers``, ``bin_edges``, ``entropy``.
        """
        return None
