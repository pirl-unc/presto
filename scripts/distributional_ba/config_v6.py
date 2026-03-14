"""V6 experiment: content-conditioned × embed_dim × head × max_nM factorial.

32-condition full factorial:
  content_conditioned: {yes, no} (2)  — handled at launch time via --content-conditioned flag
  embed_dim: {32, 64, 128, 256} (4)
  head_type: {mhcflurry, hlgauss} (2)
  max_nM: {50_000, 100_000} (2)

This config defines 16 condition specs; each is run twice (with/without flag)
to produce the full 32-condition matrix.
"""

from __future__ import annotations

from typing import Dict, List

from .config import ConditionSpec, DistributionalModel, build_model  # noqa: F401


def _make_conditions_v6() -> List[ConditionSpec]:
    """Build the 16-condition v6 factorial (before content-conditioned split)."""
    conds: List[ConditionSpec] = []
    cid = 0

    for embed_dim in (32, 64, 128, 256):
        for head_type in ("mhcflurry", "hlgauss"):
            for max_nM in (50_000, 100_000):
                cid += 1
                # hlgauss requires d2_logit; mhcflurry uses additive
                assay_mode = "d2_logit" if head_type == "hlgauss" else "additive"
                conds.append(ConditionSpec(
                    cond_id=cid,
                    head_type=head_type,
                    assay_mode=assay_mode,
                    max_nM=max_nM,
                    embed_dim=embed_dim,
                ))

    assert len(conds) == 16, f"Expected 16 conditions, got {len(conds)}"
    return conds


CONDITIONS_V6: List[ConditionSpec] = _make_conditions_v6()
CONDITIONS_V6_BY_ID: Dict[int, ConditionSpec] = {c.cond_id: c for c in CONDITIONS_V6}
