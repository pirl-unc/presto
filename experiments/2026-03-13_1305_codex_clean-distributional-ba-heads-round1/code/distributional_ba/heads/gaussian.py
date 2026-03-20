"""Heteroscedastic Gaussian head — predict mean + log-variance.

Simplest distributional approach: model output as N(mu, sigma^2).
Point prediction is mu; uncertainty is sigma.
Uses Gaussian NLL for exact samples and censored normal (Tobit) NLL
for censored samples.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AffinityHead
from ..assay_context import AdditiveIntegration, AffineIntegration
from ..losses import gaussian_nll, censored_gaussian_nll, _masked_mean


class GaussianHead(AffinityHead):
    """Heteroscedastic Gaussian in log1p space.

    Predicts mu and log_sigma. Point estimate = exp(mu) - 1 nM.
    Loss: Gaussian NLL for exact, censored normal NLL for censored.
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

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
        )
        # Separate heads for mu and log_sigma
        self.mu_head = nn.Linear(in_dim // 2, 1)
        self.logsigma_head = nn.Linear(in_dim // 2, 1)

        if assay_mode == "additive":
            self.integration = AdditiveIntegration(ctx_dim)
        elif assay_mode == "affine":
            self.integration = AffineIntegration(ctx_dim)
        else:
            raise ValueError(f"GaussianHead only supports additive/affine, got {assay_mode}")

    def _nm_to_log1p(self, ic50_nM: torch.Tensor) -> torch.Tensor:
        return torch.log(1.0 + ic50_nM.clamp(min=0.0, max=self.max_nM))

    def _log1p_to_nM(self, y: torch.Tensor) -> torch.Tensor:
        return (torch.exp(y.clamp(max=self._y_max)) - 1.0).clamp(min=0.0)

    def forward(self, h: torch.Tensor, assay_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.trunk(h)
        mu_raw = self.mu_head(feat).squeeze(-1)
        log_sigma = self.logsigma_head(feat).squeeze(-1)

        # Assay integration on mu (not on sigma)
        mu = self.integration(mu_raw, assay_emb)
        mu = mu.clamp(0.0, self._y_max)

        sigma = torch.exp(log_sigma.clamp(-5.0, 5.0))

        return {
            "pred_ic50_nM": self._log1p_to_nM(mu),
            "pred_log1p": mu,
            "pred_sigma": sigma,
            "pred_log_sigma": log_sigma,
        }

    def predict_distribution(
        self, h: torch.Tensor, assay_emb: torch.Tensor
    ) -> Optional[Dict[str, torch.Tensor]]:
        out = self.forward(h, assay_emb)
        return {
            "mu": out["pred_log1p"],
            "sigma": out["pred_sigma"],
            "type": "gaussian",
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
        mu = out["pred_log1p"]
        sigma = out["pred_sigma"]

        y = self._nm_to_log1p(ic50_nM)

        is_exact = (qual == 0)
        is_censored = (qual != 0)
        exact_mask = mask * is_exact.float()
        censor_mask = mask * is_censored.float()

        # Exact: standard Gaussian NLL
        nll_exact = gaussian_nll(mu, sigma, y)
        loss_exact = _masked_mean(nll_exact, exact_mask) if exact_mask.sum() > 0 else torch.tensor(0.0, device=h.device)

        # Censored: Tobit NLL
        nll_censor = censored_gaussian_nll(mu, sigma, y, qual)
        loss_censor = _masked_mean(nll_censor, censor_mask) if censor_mask.sum() > 0 else torch.tensor(0.0, device=h.device)

        n_exact = exact_mask.sum().clamp(min=1.0)
        n_censor = censor_mask.sum().clamp(min=1.0)
        n_total = (n_exact + n_censor).clamp(min=1.0)

        loss = (n_exact / n_total) * loss_exact + (n_censor / n_total) * loss_censor

        return loss, {
            "loss_gaussian_nll": float(loss_exact.detach()),
            "loss_censor_nll": float(loss_censor.detach()),
            "mean_sigma": float(sigma[mask.bool()].mean().detach()) if mask.sum() > 0 else 0.0,
            "n_exact": float(n_exact),
            "n_censored": float(n_censor),
        }
