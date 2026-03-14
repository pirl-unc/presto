"""Quantile regression head — predict 5 quantiles with pinball loss.

Non-parametric uncertainty: predicts q10, q25, q50, q75, q90.
Point prediction is the median (q50). Uncertainty captured by
interquartile range (q75 - q25) or full 80% interval (q90 - q10).
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .base import AffinityHead
from ..assay_context import AdditiveIntegration, AffineIntegration
from ..losses import censored_pinball_loss, _masked_mean

QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


class QuantileHead(AffinityHead):
    """Quantile regression in log1p space.

    Predicts 5 quantiles (q10, q25, q50, q75, q90).
    Point estimate = exp(q50) - 1 nM.
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
        self.n_quantiles = len(QUANTILE_LEVELS)
        self.register_buffer(
            "tau",
            torch.tensor(QUANTILE_LEVELS, dtype=torch.float32),
        )

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, self.n_quantiles),
        )

        if assay_mode == "additive":
            self.integration = AdditiveIntegration(ctx_dim)
        elif assay_mode == "affine":
            self.integration = AffineIntegration(ctx_dim)
        else:
            raise ValueError(f"QuantileHead only supports additive/affine, got {assay_mode}")

    def _nm_to_log1p(self, ic50_nM: torch.Tensor) -> torch.Tensor:
        return torch.log(1.0 + ic50_nM.clamp(min=0.0, max=self.max_nM))

    def _log1p_to_nM(self, y: torch.Tensor) -> torch.Tensor:
        return (torch.exp(y.clamp(max=self._y_max)) - 1.0).clamp(min=0.0)

    def compute_binding_signal(self, h: torch.Tensor) -> torch.Tensor:
        """Median (q50) from raw quantile predictions (pre-integration)."""
        return self.mlp(h)[:, 2]  # q50 index

    def forward(self, h: torch.Tensor, assay_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.mlp(h)  # (B, 5)

        # Apply assay integration to the median (q50, index 2)
        # Then shift all quantiles by the same offset so ordering is preserved
        median_raw = raw[:, 2]
        median_adj = self.integration(median_raw, assay_emb)
        shift = (median_adj - median_raw).unsqueeze(-1)  # (B, 1)

        quantiles = (raw + shift).clamp(0.0, self._y_max)

        # Sort to ensure monotonicity (q10 <= q25 <= q50 <= q75 <= q90)
        quantiles = quantiles.sort(dim=-1).values

        median = quantiles[:, 2]
        return {
            "pred_ic50_nM": self._log1p_to_nM(median),
            "pred_log1p": median,
            "quantiles": quantiles,  # (B, 5) in log1p space
        }

    def predict_distribution(
        self, h: torch.Tensor, assay_emb: torch.Tensor
    ) -> Optional[Dict[str, torch.Tensor]]:
        out = self.forward(h, assay_emb)
        q = out["quantiles"]
        return {
            "quantiles": q,
            "quantile_levels": self.tau,
            "iqr": q[:, 3] - q[:, 1],  # q75 - q25
            "range_80": q[:, 4] - q[:, 0],  # q90 - q10
            "type": "quantile",
        }

    def compute_loss(
        self,
        h: torch.Tensor,
        assay_emb: torch.Tensor,
        ic50_nM: torch.Tensor,
        qual: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        out = self.forward(h, assay_emb)
        quantiles = out["quantiles"]  # (B, 5)

        y = self._nm_to_log1p(ic50_nM)  # (B,)

        # Censored pinball loss for each quantile level
        loss_per_q = censored_pinball_loss(quantiles, y, qual, self.tau)  # (B, 5)
        loss_mean = loss_per_q.mean(dim=-1)  # (B,) average across quantiles

        loss = _masked_mean(loss_mean, mask)

        # Per-quantile losses for monitoring
        metrics = {}
        for i, tau_val in enumerate(QUANTILE_LEVELS):
            qloss = _masked_mean(loss_per_q[:, i], mask)
            metrics[f"loss_q{int(tau_val*100)}"] = float(qloss.detach())

        # Crossing penalty (should be near-zero after sort)
        crossing = torch.relu(quantiles[:, :-1] - quantiles[:, 1:]).sum(dim=-1)
        metrics["crossing_penalty"] = float(_masked_mean(crossing, mask).detach())

        return loss, metrics
