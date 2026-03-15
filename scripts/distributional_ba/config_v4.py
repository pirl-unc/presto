"""V4 experiment: fine-grained encoder dimension sweep.

6 conditions:
  - MHCflurry additive only (clear winner from v1/v2)
  - MAX: 50k (canonical, v3 showed MAX doesn't matter much)
  - embed_dim: 32, 64, 96, 128, 192, 256
"""

from __future__ import annotations

from typing import Dict, List

from .config import ConditionSpec, DistributionalModel, build_model  # noqa: F401


def _make_conditions_v4() -> List[ConditionSpec]:
    """Build the 6-condition v4 embed_dim sweep."""
    conds: List[ConditionSpec] = []
    cid = 0

    for embed_dim in (32, 64, 96, 128, 192, 256):
        cid += 1
        conds.append(ConditionSpec(
            cond_id=cid,
            head_type="mhcflurry",
            assay_mode="additive",
            max_nM=50_000,
            embed_dim=embed_dim,
        ))

    assert len(conds) == 6, f"Expected 6 conditions, got {len(conds)}"
    return conds


CONDITIONS_V4: List[ConditionSpec] = _make_conditions_v4()
CONDITIONS_V4_BY_ID: Dict[int, ConditionSpec] = {c.cond_id: c for c in CONDITIONS_V4}
