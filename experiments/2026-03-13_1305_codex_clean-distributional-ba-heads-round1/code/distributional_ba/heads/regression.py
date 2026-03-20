"""Regression heads: MHCflurry bounded and Log-MSE."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .base import AffinityHead
from ..assay_context import AdditiveIntegration, AffineIntegration
from ..losses import mhcflurry_censored_loss, log_censored_mse, _masked_mean


class MHCflurryHead(AffinityHead):
    """MHCflurry-style bounded regression in [0,1].

    target = 1 - log10(ic50) / log10(MAX)
    Stronger binders → higher target.
    Decode: ic50 = MAX^(1-pred).
    """

    def __init__(
        self,
        in_dim: int,
        ctx_dim: int,
        max_nM: float = 50_000.0,
        assay_mode: str = "affine",
    ):
        super().__init__()
        self.max_nM = float(max_nM)
        self._log10_max = math.log10(max(self.max_nM, 1.0))
        self.assay_mode = assay_mode

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
        )

        if assay_mode == "additive":
            self.integration = AdditiveIntegration(ctx_dim)
        elif assay_mode == "affine":
            self.integration = AffineIntegration(ctx_dim)
        else:
            raise ValueError(f"MHCflurryHead only supports additive/affine, got {assay_mode}")

    def _nm_to_bounded(self, ic50_nM: torch.Tensor) -> torch.Tensor:
        log10_val = torch.log10(ic50_nM.clamp(min=1.0, max=self.max_nM))
        return (1.0 - log10_val / self._log10_max).clamp(0.0, 1.0)

    def _bounded_to_nM(self, bounded: torch.Tensor) -> torch.Tensor:
        return torch.pow(self.max_nM, (1.0 - bounded).clamp(0.0, 1.0))

    def forward(self, h: torch.Tensor, assay_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.mlp(h).squeeze(-1)  # (B,)
        # Apply integration in pre-sigmoid (logit) space to preserve gradient flow
        adjusted_raw = self.integration(raw, assay_emb)
        bounded = torch.sigmoid(adjusted_raw)  # always in (0,1) with gradient
        return {"pred_ic50_nM": self._bounded_to_nM(bounded), "pred_bounded": bounded}

    def compute_loss(
        self,
        h: torch.Tensor,
        assay_emb: torch.Tensor,
        ic50_nM: torch.Tensor,
        qual: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(h, assay_emb)
        pred_bounded = out["pred_bounded"]
        target_bounded = self._nm_to_bounded(ic50_nM)
        # In bounded space, larger = stronger binding → flip qual direction
        qual_flipped = -qual
        loss = mhcflurry_censored_loss(pred_bounded, target_bounded, qual_flipped, mask)
        return loss, {"loss_mhcflurry": float(loss.detach())}


class LogMSEHead(AffinityHead):
    """Log-MSE regression in y = log(1 + ic50) space.

    Decode: ic50 = exp(y) - 1.
    """

    def __init__(
        self,
        in_dim: int,
        ctx_dim: int,
        max_nM: float = 50_000.0,
        assay_mode: str = "affine",
    ):
        super().__init__()
        self.max_nM = float(max_nM)
        self._y_max = math.log(1.0 + self.max_nM)
        self.assay_mode = assay_mode

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, 1),
        )

        if assay_mode == "additive":
            self.integration = AdditiveIntegration(ctx_dim)
        elif assay_mode == "affine":
            self.integration = AffineIntegration(ctx_dim)
        else:
            raise ValueError(f"LogMSEHead only supports additive/affine, got {assay_mode}")

    def _nm_to_log1p(self, ic50_nM: torch.Tensor) -> torch.Tensor:
        return torch.log(1.0 + ic50_nM.clamp(min=0.0, max=self.max_nM))

    def _log1p_to_nM(self, y: torch.Tensor) -> torch.Tensor:
        return (torch.exp(y.clamp(max=self._y_max)) - 1.0).clamp(min=0.0)

    def forward(self, h: torch.Tensor, assay_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.mlp(h).squeeze(-1)  # (B,)
        # Softplus to keep in [0, ~y_max] with smooth gradient
        base = torch.nn.functional.softplus(raw)
        adjusted = self.integration(base, assay_emb)
        adjusted = adjusted.clamp(0.0, self._y_max)
        return {"pred_ic50_nM": self._log1p_to_nM(adjusted), "pred_log1p": adjusted}

    def compute_loss(
        self,
        h: torch.Tensor,
        assay_emb: torch.Tensor,
        ic50_nM: torch.Tensor,
        qual: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(h, assay_emb)
        pred_y = out["pred_log1p"]
        target_y = self._nm_to_log1p(ic50_nM)
        # In log1p space, larger values = weaker binding — same direction as nM quals
        loss = log_censored_mse(pred_y, target_y, qual, mask)
        return loss, {"loss_log_mse": float(loss.detach())}
