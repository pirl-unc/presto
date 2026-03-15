"""V3 experiment: MHCflurry MAX sweep × encoder dimension sweep.

24 conditions:
  - MHCflurry additive only (clear winner from v1/v2)
  - MAX: 25k, 50k, 75k, 100k, 125k, 150k
  - embed_dim: 128, 256, 384, 512
"""

from __future__ import annotations

from typing import Dict, List

from .config import ConditionSpec, DistributionalModel, build_model  # noqa: F401


def _make_conditions_v3() -> List[ConditionSpec]:
    """Build the 24-condition v3 matrix."""
    conds: List[ConditionSpec] = []
    cid = 0

    for embed_dim in (128, 256, 384, 512):
        for max_nM in (25_000, 50_000, 75_000, 100_000, 125_000, 150_000):
            cid += 1
            conds.append(ConditionSpec(
                cond_id=cid,
                head_type="mhcflurry",
                assay_mode="additive",
                max_nM=max_nM,
                embed_dim=embed_dim,
            ))

    assert len(conds) == 24, f"Expected 24 conditions, got {len(conds)}"
    return conds


CONDITIONS_V3: List[ConditionSpec] = _make_conditions_v3()
CONDITIONS_V3_BY_ID: Dict[int, ConditionSpec] = {c.cond_id: c for c in CONDITIONS_V3}
