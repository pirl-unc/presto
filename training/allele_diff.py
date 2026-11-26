"""Within-peptide cross-allele comparative binding loss (`AlleleDiffPool`).

Mathematical dual of peptide-level MSE:
- Standard single-row MSE on `(peptide, allele) -> log10(nM)` lets shared peptide-level
  variance (`±1.5 log10 nM`) dominate gradients over 2-3 residue MHC groove polymorphisms
  such as `DRB1*04:01 / *04:04 / *04:05` (`β57`, `β71`, `β86`), allowing close alleles
  to collapse (`r_pred ≈ 0.985` vs `r_true ≈ 0.22–0.58`).
- Sampling exact quantitative pairs `(pep, mhc_A, y_A)` and `(pep, mhc_B, y_B)` for the
  *same* peptide measured on two distinct MHC alleles in the same endpoint & class and
  supervising `Huber((ŷ_A - ŷ_B) - (y_A - y_B))` cancels BOTH peptide-level baseline and
  endpoint-level intercept identically (`0`), sending 100% of the gradient straight into
  the allele-specific groove and 58-pocket IMGT interaction weights.

Uses strictly real, exact (`qualifier == 0`) training split rows — zero synthetic data.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from ..data.modular import MHCInput, Observation, observation_target

_QUANT_ENDPOINTS = frozenset(("ic50", "kd_via_ic50", "kd_via_ec50", "kd"))


def _groove_identity(m1: MHCInput, m2: MHCInput) -> float:
    """Fast normalized sequence identity across `(groove1, groove2)`."""
    s1 = f"{m1.groove1}|{m1.groove2}"
    s2 = f"{m2.groove1}|{m2.groove2}"
    n = max(len(s1), len(s2), 1)
    matches = sum(1 for a, b in zip(s1, s2, strict=False) if a == b)
    return matches / n


@dataclass(frozen=True)
class AlleleDiffPair:
    peptide: str
    mhc_class: str
    endpoint: str
    mhc_a: MHCInput
    mhc_b: MHCInput
    target_diff_log10: float
    groove_identity: float


class AlleleDiffPool:
    """Pre-indexed pool of real within-peptide cross-allele quantitative binding pairs."""

    def __init__(
        self,
        hard_close_by_class: dict[str, list[AlleleDiffPair]],
        general_by_class: dict[str, list[AlleleDiffPair]],
        *,
        total_groups: int = 0,
    ):
        self.hard_close_by_class = hard_close_by_class
        self.general_by_class = general_by_class
        self.total_groups = int(total_groups)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "total_groups": self.total_groups,
            "hard_close_I": len(self.hard_close_by_class.get("I", ())),
            "hard_close_II": len(self.hard_close_by_class.get("II", ())),
            "general_I": len(self.general_by_class.get("I", ())),
            "general_II": len(self.general_by_class.get("II", ())),
        }

    @classmethod
    def from_observations(
        cls,
        observations: Sequence[Observation],
        *,
        min_diff_log10: float = 0.15,
        close_identity_threshold: float = 0.78,
        max_pairs_per_group: int = 12,
    ) -> AlleleDiffPool:
        """Build comparative pairs from exact quantitative training observations."""
        grouped: dict[tuple[str, str, str], dict[tuple[str, str], tuple[MHCInput, list[float]]]] = (
            defaultdict(dict)
        )
        for row in observations:
            if row.split not in ("train", ""):
                continue
            if row.endpoint not in _QUANT_ENDPOINTS:
                continue
            if row.qualifier not in (0, "=", "exact", None):
                continue
            if row.value is None or not math.isfinite(float(row.value)):
                continue
            try:
                target_log10 = float(observation_target(row))
            except Exception:  # noqa: BLE001
                continue
            if not math.isfinite(target_log10):
                continue
            if len(row.mhc) != 1:
                continue
            mhc = row.mhc[0]
            mhc_class = mhc.class_override or mhc.mhc_class
            if mhc_class not in ("I", "II"):
                continue
            if not mhc.groove1 or not mhc.groove2:
                continue
            gkey = (mhc.groove1, mhc.groove2)
            slot = grouped[(row.peptide, row.endpoint, mhc_class)]
            if gkey not in slot:
                slot[gkey] = (mhc, [target_log10])
            else:
                slot[gkey][1].append(target_log10)

        hard_close: dict[str, list[AlleleDiffPair]] = {"I": [], "II": []}
        general: dict[str, list[AlleleDiffPair]] = {"I": [], "II": []}
        multi_groups = 0

        for (pep, ep, mhc_class), alleles_map in grouped.items():
            if len(alleles_map) < 2:
                continue
            multi_groups += 1
            entries = [(mhc, sum(vals) / len(vals)) for mhc, vals in alleles_map.values()]
            # Sort deterministically by groove strings so pool construction is reproducible
            entries.sort(key=lambda item: (item[0].groove1, item[0].groove2))
            group_hard: list[AlleleDiffPair] = []
            group_gen: list[AlleleDiffPair] = []
            n_ent = len(entries)
            for i in range(n_ent):
                m_a, y_a = entries[i]
                for j in range(i + 1, n_ent):
                    m_b, y_b = entries[j]
                    diff = y_a - y_b
                    if abs(diff) < min_diff_log10:
                        continue
                    gid = _groove_identity(m_a, m_b)
                    pair = AlleleDiffPair(
                        peptide=pep,
                        mhc_class=mhc_class,
                        endpoint=ep,
                        mhc_a=m_a,
                        mhc_b=m_b,
                        target_diff_log10=diff,
                        groove_identity=gid,
                    )
                    if gid >= close_identity_threshold:
                        group_hard.append(pair)
                    else:
                        group_gen.append(pair)
            # Cap per-group pairs to prevent any single huge peptide panel from dominating
            if len(group_hard) > max_pairs_per_group:
                group_hard.sort(key=lambda p: (-p.groove_identity, -abs(p.target_diff_log10)))
                group_hard = group_hard[:max_pairs_per_group]
            if len(group_gen) > max_pairs_per_group:
                group_gen.sort(key=lambda p: -abs(p.target_diff_log10))
                group_gen = group_gen[:max_pairs_per_group]
            hard_close[mhc_class].extend(group_hard)
            general[mhc_class].extend(group_gen)

        return cls(hard_close, general, total_groups=multi_groups)

    def sample_pairs(
        self,
        rng: random.Random,
        *,
        pairs_per_class: int = 12,
        hard_close_fraction: float = 0.65,
    ) -> list[AlleleDiffPair]:
        """Draw class-balanced comparative pairs prioritizing closely related alleles."""
        drawn: list[AlleleDiffPair] = []
        for mhc_class in ("I", "II"):
            hc = self.hard_close_by_class.get(mhc_class, [])
            gn = self.general_by_class.get(mhc_class, [])
            if not hc and not gn:
                continue
            n_hc_want = int(round(pairs_per_class * hard_close_fraction))
            n_hc = min(len(hc), n_hc_want)
            n_gn = min(len(gn), pairs_per_class - n_hc)
            # Top up from whichever bucket still has capacity
            if n_hc + n_gn < pairs_per_class:
                if len(hc) > n_hc:
                    n_hc = min(len(hc), pairs_per_class - n_gn)
                elif len(gn) > n_gn:
                    n_gn = min(len(gn), pairs_per_class - n_hc)
            if n_hc > 0:
                drawn.extend(rng.sample(hc, n_hc) if n_hc < len(hc) else list(hc))
            if n_gn > 0:
                drawn.extend(rng.sample(gn, n_gn) if n_gn < len(gn) else list(gn))
        return drawn


def allele_diff_loss(
    model,
    pairs: Sequence[AlleleDiffPair],
    scales: dict[str, Any],
    *,
    beta: float = 0.5,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Compute class-balanced Huber loss on within-peptide cross-allele binding differences."""
    if not pairs:
        return None, {}
    ic50_scale = scales.get("ic50")
    denom = float(ic50_scale.scale) if (ic50_scale is not None and ic50_scale.scale > 0) else 1.0

    flat_queries: list[tuple[str, MHCInput, str]] = []
    for p in pairs:
        flat_queries.append((p.peptide, p.mhc_a, p.mhc_class))
        flat_queries.append((p.peptide, p.mhc_b, p.mhc_class))

    preds = model.score_binding_pairs(flat_queries).float()
    pred_a = preds[0::2]
    pred_b = preds[1::2]
    pred_diff = pred_a - pred_b

    target_diff = torch.tensor(
        [p.target_diff_log10 / denom for p in pairs],
        dtype=torch.float32,
        device=preds.device,
    )

    by_cls_idx: dict[str, list[int]] = {"I": [], "II": []}
    for i, p in enumerate(pairs):
        by_cls_idx[p.mhc_class].append(i)

    cls_losses: list[torch.Tensor] = []
    stats: dict[str, float] = {"allele_diff_pairs": float(len(pairs))}
    for mhc_class in ("I", "II"):
        idxs = by_cls_idx[mhc_class]
        if not idxs:
            continue
        idx_t = torch.tensor(idxs, dtype=torch.long, device=preds.device)
        pd_c = pred_diff.index_select(0, idx_t)
        td_c = target_diff.index_select(0, idx_t)
        c_loss = F.smooth_l1_loss(pd_c, td_c, beta=float(beta))
        cls_losses.append(c_loss)
        with torch.no_grad():
            mae_log10 = float((pd_c - td_c).abs().mean().item() * denom)
            sign_acc = float(((pd_c * td_c) > 0).float().mean().item())
        stats[f"allele_diff_loss_{mhc_class}"] = float(c_loss.detach())
        stats[f"allele_diff_mae_log10_{mhc_class}"] = mae_log10
        stats[f"allele_diff_sign_acc_{mhc_class}"] = sign_acc
        stats[f"allele_diff_pairs_{mhc_class}"] = float(len(idxs))

    if not cls_losses:
        return None, {}
    total = torch.stack(cls_losses).mean()
    stats["allele_diff_loss"] = float(total.detach())
    return total, stats
