"""Distributional base class for bin-based affinity heads."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AffinityHead
from ..assay_context import D1AffineIntegration, D2LogitIntegration
from ..losses import distributional_cross_entropy, survival_nll, _masked_mean


class DistributionalBase(AffinityHead):
    """Base for distributional heads (Two-Hot, HL-Gauss).

    Bins are uniform in log1p space: edges = linspace(0, log(1+MAX), K+1).
    Subclasses implement ``_build_target_vector`` to produce soft labels.
    """

    def __init__(
        self,
        in_dim: int,
        ctx_dim: int,
        n_bins: int = 128,
        max_nM: float = 50_000.0,
        assay_mode: str = "d1_affine",
    ):
        super().__init__()
        self.n_bins = n_bins
        self.max_nM = float(max_nM)
        self._y_max = math.log(1.0 + self.max_nM)
        self.assay_mode = assay_mode

        # Register canonical bin edges/centers as buffers (not parameters)
        edges = torch.linspace(0.0, self._y_max, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        self.register_buffer("bin_edges", edges)
        self.register_buffer("bin_centers", centers)
        self.bin_width = float(edges[1] - edges[0])

        # MLP: encoder repr → K logits
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.GELU(),
            nn.Linear(in_dim // 2, n_bins),
        )

        # Assay integration
        if assay_mode == "d1_affine":
            self.integration = D1AffineIntegration(ctx_dim)
        elif assay_mode == "d2_logit":
            self.integration = D2LogitIntegration(ctx_dim, n_bins)
        else:
            raise ValueError(f"Distributional heads require d1_affine or d2_logit, got {assay_mode}")

    def _nm_to_log1p(self, ic50_nM: torch.Tensor) -> torch.Tensor:
        return torch.log(1.0 + ic50_nM.clamp(min=0.0, max=self.max_nM))

    def _log1p_to_nM(self, y: torch.Tensor) -> torch.Tensor:
        return (torch.exp(y.clamp(max=self._y_max)) - 1.0).clamp(min=0.0)

    def compute_binding_signal(self, h: torch.Tensor) -> torch.Tensor:
        """Expected value from raw logits with canonical bins (pre-integration)."""
        logits = self.mlp(h)  # (B, K)
        probs = F.softmax(logits, dim=-1)
        return (probs * self.bin_centers.unsqueeze(0)).sum(dim=-1)  # (B,)

    def _get_logits_centers_edges(
        self, h: torch.Tensor, assay_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (logits, centers, edges) with assay integration applied.

        logits: (B, K)
        centers: (B, K) or (K,)
        edges: (B, K+1) or (K+1,)
        """
        logits = self.mlp(h)  # (B, K)

        if self.assay_mode == "d1_affine":
            adj = self.integration(self.bin_centers, self.bin_edges, assay_emb)
            return logits, adj["centers"], adj["edges"]
        else:  # d2_logit
            logits = self.integration(logits, assay_emb)
            return logits, self.bin_centers, self.bin_edges

    def forward(self, h: torch.Tensor, assay_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits, centers, edges = self._get_logits_centers_edges(h, assay_emb)
        probs = F.softmax(logits, dim=-1)  # (B, K)

        # Expected value: sum(probs * centers)
        if centers.dim() == 1:
            ev_log1p = (probs * centers.unsqueeze(0)).sum(dim=-1)
        else:
            ev_log1p = (probs * centers).sum(dim=-1)

        pred_nM = self._log1p_to_nM(ev_log1p)
        return {"pred_ic50_nM": pred_nM, "pred_log1p": ev_log1p}

    def predict_distribution(
        self, h: torch.Tensor, assay_emb: torch.Tensor
    ) -> Optional[Dict[str, torch.Tensor]]:
        logits, centers, edges = self._get_logits_centers_edges(h, assay_emb)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-10))).sum(dim=-1)
        return {
            "probs": probs,
            "bin_centers": centers,
            "bin_edges": edges,
            "entropy": entropy,
        }

    def _build_target_vector(
        self, y: torch.Tensor, centers: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        """Build soft target vector. Implemented by subclasses."""
        raise NotImplementedError

    def compute_loss(
        self,
        h: torch.Tensor,
        assay_emb: torch.Tensor,
        ic50_nM: torch.Tensor,
        qual: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        logits, centers, edges = self._get_logits_centers_edges(h, assay_emb)
        probs = F.softmax(logits, dim=-1)

        y = self._nm_to_log1p(ic50_nM)

        is_exact = (qual == 0)
        is_censored = (qual != 0)

        # --- Exact samples: cross-entropy with soft target ---
        if centers.dim() == 1:
            centers_b = centers.unsqueeze(0).expand(h.shape[0], -1)
        else:
            centers_b = centers
        if edges.dim() == 1:
            edges_b = edges.unsqueeze(0).expand(h.shape[0], -1)
        else:
            edges_b = edges

        target_vec = self._build_target_vector(y, centers_b, edges_b)
        ce = distributional_cross_entropy(logits, target_vec)

        # --- Censored samples: survival NLL ---
        surv = survival_nll(probs, edges_b, y, qual)

        # Combine: exact → CE, censored → survival NLL
        exact_mask = mask * is_exact.float()
        censor_mask = mask * is_censored.float()

        loss_exact = _masked_mean(ce, exact_mask) if exact_mask.sum() > 0 else torch.tensor(0.0, device=h.device)
        loss_censor = _masked_mean(surv, censor_mask) if censor_mask.sum() > 0 else torch.tensor(0.0, device=h.device)

        # Weight by fraction of samples
        n_exact = exact_mask.sum().clamp(min=1.0)
        n_censor = censor_mask.sum().clamp(min=1.0)
        n_total = (n_exact + n_censor).clamp(min=1.0)

        loss = (n_exact / n_total) * loss_exact + (n_censor / n_total) * loss_censor

        metrics = {
            "loss_ce": float(loss_exact.detach()),
            "loss_surv": float(loss_censor.detach()),
            "n_exact": float(n_exact),
            "n_censored": float(n_censor),
        }
        return loss, metrics
