"""V2 experiment: bin sweep + simpler uncertainty heads.

28 conditions:
  - Regression baselines (4): MHCflurry/LogMSE × additive × 50k/250k
  - Gaussian head (4): affine/additive × 50k/250k
  - Quantile head (4): affine/additive × 50k/250k
  - Two-Hot D2-logit bin sweep (8): K=8/16/32/64 × 50k/250k
  - HL-Gauss D2-logit bin sweep (8): K=8/16/32/64 × 50k/250k (sigma=0.75)
"""

from __future__ import annotations

from typing import Dict, List

from .config import ConditionSpec, DistributionalModel, build_model  # noqa: F401


def _make_conditions_v2() -> List[ConditionSpec]:
    """Build the 28-condition v2 matrix."""
    conds: List[ConditionSpec] = []
    cid = 0

    def _add(head_type, assay_mode, max_nM, n_bins=128, sigma_mult=0.75):
        nonlocal cid
        cid += 1
        conds.append(ConditionSpec(
            cond_id=cid,
            head_type=head_type,
            assay_mode=assay_mode,
            max_nM=max_nM,
            n_bins=n_bins,
            sigma_mult=sigma_mult,
        ))

    # Block 1 (4): Regression baselines — additive (best from v1)
    for max_nM in (50_000, 250_000):
        _add("mhcflurry", "additive", max_nM)
    for max_nM in (50_000, 250_000):
        _add("log_mse", "additive", max_nM)

    # Block 2 (4): Gaussian head — affine and additive
    for max_nM in (50_000, 250_000):
        _add("gaussian", "affine", max_nM)
    for max_nM in (50_000, 250_000):
        _add("gaussian", "additive", max_nM)

    # Block 3 (4): Quantile head — affine and additive
    for max_nM in (50_000, 250_000):
        _add("quantile", "affine", max_nM)
    for max_nM in (50_000, 250_000):
        _add("quantile", "additive", max_nM)

    # Block 4 (8): Two-Hot D2-logit bin sweep
    for n_bins in (8, 16, 32, 64):
        for max_nM in (50_000, 250_000):
            _add("twohot", "d2_logit", max_nM, n_bins=n_bins)

    # Block 5 (8): HL-Gauss D2-logit bin sweep (sigma=0.75)
    for n_bins in (8, 16, 32, 64):
        for max_nM in (50_000, 250_000):
            _add("hlgauss", "d2_logit", max_nM, n_bins=n_bins, sigma_mult=0.75)

    assert len(conds) == 28, f"Expected 28 conditions, got {len(conds)}"
    return conds


CONDITIONS_V2: List[ConditionSpec] = _make_conditions_v2()
CONDITIONS_V2_BY_ID: Dict[int, ConditionSpec] = {c.cond_id: c for c in CONDITIONS_V2}
