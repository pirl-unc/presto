"""V5 experiment: content-conditioned assay context.

Compare content-independent (v4 baseline) vs content-conditioned assay bias.
The content-conditioned variant feeds the binding logit and mean-pooled
pep/mhc_a/mhc_b representations (detached) into the assay context encoder.

6 conditions: embed_dim in (32, 64, 96, 128, 192, 256), all content-conditioned.
Compare against v4 (same dims, content-independent) to isolate the effect.
"""

from __future__ import annotations

from typing import Dict, List

from .config import ConditionSpec, DistributionalModel, build_model  # noqa: F401


def _make_conditions_v5() -> List[ConditionSpec]:
    """Build the 6-condition v5 content-conditioned sweep."""
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


CONDITIONS_V5: List[ConditionSpec] = _make_conditions_v5()
CONDITIONS_V5_BY_ID: Dict[int, ConditionSpec] = {c.cond_id: c for c in CONDITIONS_V5}
