"""HL-Gauss distributional head.

Target vector: Gaussian CDF deltas at bin edges. Produces a soft, symmetric
target centered on the true value. From Imani & White (2018).
"""

from __future__ import annotations

import math

import torch

from .distributional import DistributionalBase


class HLGaussHead(DistributionalBase):
    """Histogram Loss with Gaussian targets for distributional affinity prediction.

    sigma_mult controls the width of the Gaussian relative to bin width:
      sigma = sigma_mult * bin_width
    """

    def __init__(
        self,
        in_dim: int,
        ctx_dim: int,
        n_bins: int = 128,
        max_nM: float = 50_000.0,
        sigma_mult: float = 0.75,
        assay_mode: str = "d1_affine",
    ):
        super().__init__(
            in_dim=in_dim,
            ctx_dim=ctx_dim,
            n_bins=n_bins,
            max_nM=max_nM,
            assay_mode=assay_mode,
        )
        self.sigma_mult = float(sigma_mult)

    def _build_target_vector(
        self, y: torch.Tensor, centers: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        """Gaussian CDF-delta target.

        P(bin k) = Phi((edge_{k+1} - y) / sigma) - Phi((edge_k - y) / sigma)

        Args:
            y: (B,) target values in log1p space.
            centers: (B, K) bin centers (unused, interface compat).
            edges: (B, K+1) bin edges.

        Returns:
            (B, K) target probability vector (sums to ~1).
        """
        # Compute per-example bin width (may vary with D1-affine)
        bin_widths = (edges[:, 1:] - edges[:, :-1]).mean(dim=-1, keepdim=True)  # (B, 1)
        sigma = self.sigma_mult * bin_widths.clamp(min=1e-8)  # (B, 1)

        y_exp = y.unsqueeze(-1)  # (B, 1)
        # Standardized edge positions
        z = (edges - y_exp) / sigma  # (B, K+1)
        cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))  # (B, K+1)

        # P(bin k) = CDF(right edge) - CDF(left edge)
        target = cdf[:, 1:] - cdf[:, :-1]  # (B, K)

        # Renormalize to handle truncation at boundaries
        target = target / target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return target
