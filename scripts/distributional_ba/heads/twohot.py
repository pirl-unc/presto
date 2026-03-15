"""Two-Hot distributional head.

Target vector: two-hot encoding at the two bins straddling the exact value,
weighted by proximity. Produces a sharp, unimodal target.
"""

from __future__ import annotations

import torch

from .distributional import DistributionalBase


class TwoHotHead(DistributionalBase):
    """Two-hot soft target for distributional affinity prediction."""

    def __init__(
        self,
        in_dim: int,
        ctx_dim: int,
        n_bins: int = 128,
        max_nM: float = 50_000.0,
        assay_mode: str = "d1_affine",
    ):
        super().__init__(
            in_dim=in_dim,
            ctx_dim=ctx_dim,
            n_bins=n_bins,
            max_nM=max_nM,
            assay_mode=assay_mode,
        )

    def _build_target_vector(
        self, y: torch.Tensor, centers: torch.Tensor, edges: torch.Tensor
    ) -> torch.Tensor:
        """Two-hot target: weight splits between the two nearest bin centers.

        Args:
            y: (B,) target values in log1p space.
            centers: (B, K) bin centers.
            edges: (B, K+1) bin edges (unused for two-hot, included for interface).

        Returns:
            (B, K) target probability vector (sums to 1).
        """
        B, K = centers.shape
        target = torch.zeros(B, K, device=y.device, dtype=y.dtype)

        # Find lower-bound bin index for each sample
        y_exp = y.unsqueeze(-1)  # (B, 1)
        # diffs[b, k] = centers[b, k] - y[b]
        diffs = centers - y_exp  # (B, K)

        # Index of first center >= y (upper neighbor)
        # If y < all centers, upper_idx = 0; if y > all centers, upper_idx = K
        ge_mask = (diffs >= 0).float()
        # Use argmax on the mask — gives first True index. If all False, gives 0.
        has_ge = ge_mask.sum(dim=-1) > 0  # (B,)
        upper_idx = ge_mask.argmax(dim=-1)  # (B,)
        # Fix: when y > all centers, upper_idx should be K-1
        upper_idx = torch.where(has_ge, upper_idx, torch.full_like(upper_idx, K - 1))
        lower_idx = (upper_idx - 1).clamp(min=0)

        # Gather center values at lower and upper indices
        c_lower = centers.gather(1, lower_idx.unsqueeze(-1)).squeeze(-1)
        c_upper = centers.gather(1, upper_idx.unsqueeze(-1)).squeeze(-1)

        span = (c_upper - c_lower).clamp(min=1e-8)
        # Weight for upper = how close y is to lower (higher fraction → more upper)
        w_upper = ((y - c_lower) / span).clamp(0.0, 1.0)
        w_lower = 1.0 - w_upper

        # Handle edge cases: if lower_idx == upper_idx, put all weight there
        same = (lower_idx == upper_idx).float()
        w_lower = w_lower * (1.0 - same) + same
        w_upper = w_upper * (1.0 - same)

        target.scatter_(1, lower_idx.unsqueeze(-1), w_lower.unsqueeze(-1))
        target.scatter_add_(1, upper_idx.unsqueeze(-1), w_upper.unsqueeze(-1))

        return target
