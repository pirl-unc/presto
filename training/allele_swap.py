"""Cross-allele swap binding contrast — the exact mathematical dual of PairPool.

- PairPool (same allele, affinity-matched pos vs same-protein neg, binding detached)
  isolates **pure processing** (Binding ⊥ Processing).
- AlleleSwapPool (different allele's real MS-eluted peptide scored on the anchor MHC
  through pure `pmhc_binding` / `ic50` with ZERO processing, ZERO flanks, ZERO MS bias)
  isolates **pure allele-specific binding** (Processing ⊥ Binding) for BOTH Class I
  and Class II.

Because every negative peptide is a real MS-eluted training ligand from a disjoint MHC
allele of the same class, neither antigen processing nor MS detectability can act as a
shortcut; the shift-invariant InfoNCE loss sharpens 9-mer pocket motifs directly inside
the calibrated affinity head without shifting absolute `log10(nM)` intercepts.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Sequence

import torch
from torch.nn import functional as F

from ..data.modular import MHCInput, Observation

_AA20 = frozenset("ACDEFGHIKLMNPQRSTVWY")


def _valid_peptide(peptide: str, mhc_class: str) -> bool:
    if not peptide:
        return False
    min_len = 9 if mhc_class == "II" else 8
    if len(peptide) < min_len:
        return False
    return all(ch in _AA20 for ch in peptide)


def _groove_key(mhc: MHCInput, mhc_class: str) -> tuple[str, str, str]:
    return (mhc_class, mhc.groove1, mhc.groove2)


@dataclass(frozen=True)
class AlleleSwapGroup:
    mhc_class: str
    mhc: MHCInput
    positive_peptide: str
    negative_peptides: tuple[str, ...]


class AlleleSwapPool:
    """Bounded per-class ring buffer of monoallelic MS elution training positives."""

    def __init__(self, max_buffer_per_class: int = 512):
        if max_buffer_per_class < 8:
            raise ValueError("max_buffer_per_class must be >= 8")
        self.max_buffer_per_class = int(max_buffer_per_class)
        self._buffers: dict[str, deque[tuple[str, MHCInput, tuple[str, str, str]]]] = {
            "I": deque(maxlen=self.max_buffer_per_class),
            "II": deque(maxlen=self.max_buffer_per_class),
        }

    def observe_batch(
        self, rows: Sequence[Observation]
    ) -> list[tuple[str, str, MHCInput, tuple[str, str, str]]]:
        """Record monoallelic MS elution training positives and return batch anchors."""
        batch_anchors: list[tuple[str, str, MHCInput, tuple[str, str, str]]] = []
        for row in rows:
            if row.endpoint != "elution_observation" or row.value != 1.0:
                continue
            if row.split not in ("train", ""):
                continue
            if len(row.mhc) != 1:
                continue
            mhc = row.mhc[0]
            mhc_class = mhc.class_override or mhc.mhc_class
            if mhc_class not in ("I", "II"):
                continue
            if not mhc.groove1 or not mhc.groove2:
                continue
            pep = row.peptide.upper()
            if not _valid_peptide(pep, mhc_class):
                continue
            gkey = _groove_key(mhc, mhc_class)
            entry = (pep, mhc, gkey)
            self._buffers[mhc_class].append(entry)
            batch_anchors.append((mhc_class, pep, mhc, gkey))
        return batch_anchors

    def sample_groups(
        self,
        rows: Sequence[Observation],
        rng: random.Random,
        *,
        anchors_per_class: int = 12,
        negatives_per_anchor: int = 16,
    ) -> list[AlleleSwapGroup]:
        """Update buffer from batch rows and draw cross-allele contrast groups per class."""
        batch_anchors = self.observe_batch(rows)
        by_cls_anchors: dict[str, list[tuple[str, MHCInput, tuple[str, str, str]]]] = {
            "I": [],
            "II": [],
        }
        seen_anchor: set[tuple[str, tuple[str, str, str]]] = set()
        for mhc_class, pep, mhc, gkey in batch_anchors:
            ak = (pep, gkey)
            if ak not in seen_anchor:
                seen_anchor.add(ak)
                by_cls_anchors[mhc_class].append((pep, mhc, gkey))

        groups: list[AlleleSwapGroup] = []
        for mhc_class in ("I", "II"):
            buf = list(self._buffers[mhc_class])
            if len(buf) < 2:
                continue
            known_on_allele: dict[tuple[str, str, str], set[str]] = {}
            for b_pep, _, b_gkey in buf:
                known_on_allele.setdefault(b_gkey, set()).add(b_pep)
            if len(known_on_allele) < 2:
                continue

            anchors = list(by_cls_anchors[mhc_class])
            if len(anchors) < anchors_per_class:
                # Top up from buffer so both Class I and Class II stay active every step
                extra = [e for e in buf if (e[0], e[2]) not in seen_anchor]
                rng.shuffle(extra)
                anchors.extend(extra[: anchors_per_class - len(anchors)])
            if len(anchors) > anchors_per_class:
                rng.shuffle(anchors)
                anchors = anchors[:anchors_per_class]

            for pep_i, mhc_i, gkey_i in anchors:
                forbidden = known_on_allele.get(gkey_i, set())
                disjoint_peps = list(
                    {
                        b_pep
                        for b_pep, _, b_gkey in buf
                        if b_gkey != gkey_i and b_pep != pep_i and b_pep not in forbidden
                    }
                )
                if not disjoint_peps:
                    continue
                if len(disjoint_peps) > negatives_per_anchor:
                    disjoint_peps = rng.sample(disjoint_peps, negatives_per_anchor)
                groups.append(
                    AlleleSwapGroup(
                        mhc_class=mhc_class,
                        mhc=mhc_i,
                        positive_peptide=pep_i,
                        negative_peptides=tuple(disjoint_peps),
                    )
                )
        return groups


def allele_swap_loss(
    model,
    groups: Sequence[AlleleSwapGroup],
    *,
    temperature: float = 0.5,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    """Compute class-balanced shift-invariant InfoNCE over pure pMHC binding scores."""
    if not groups:
        return None, {}
    if temperature <= 0:
        raise ValueError("allele_swap temperature must be positive")

    flat_pairs: list[tuple[str, MHCInput, str]] = []
    spans: list[tuple[str, int, int]] = []
    offset = 0
    for grp in groups:
        flat_pairs.append((grp.positive_peptide, grp.mhc, grp.mhc_class))
        for neg_pep in grp.negative_peptides:
            flat_pairs.append((neg_pep, grp.mhc, grp.mhc_class))
        length = 1 + len(grp.negative_peptides)
        spans.append((grp.mhc_class, offset, offset + length))
        offset += length

    # Pure calibrated binding scores (lower = tighter binding, exact 0.0 processing/MS)
    binding_scores = model.score_binding_pairs(flat_pairs)
    by_cls_losses: dict[str, list[torch.Tensor]] = {"I": [], "II": []}
    by_cls_acc: dict[str, list[float]] = {"I": [], "II": []}
    zero_target = torch.zeros(1, dtype=torch.long, device=binding_scores.device)

    for mhc_class, start, end in spans:
        logits = (-binding_scores[start:end].float()) / float(temperature)
        by_cls_losses[mhc_class].append(F.cross_entropy(logits.unsqueeze(0), zero_target))
        with torch.no_grad():
            by_cls_acc[mhc_class].append(1.0 if int(logits.argmax().item()) == 0 else 0.0)

    cls_means: list[torch.Tensor] = []
    stats: dict[str, float] = {
        "allele_swap_groups": float(len(groups)),
        "allele_swap_queries": float(len(flat_pairs)),
    }
    for mhc_class in ("I", "II"):
        if by_cls_losses[mhc_class]:
            m_loss = torch.stack(by_cls_losses[mhc_class]).mean()
            cls_means.append(m_loss)
            stats[f"allele_swap_loss_{mhc_class}"] = float(m_loss.detach())
            stats[f"allele_swap_top1_{mhc_class}"] = sum(by_cls_acc[mhc_class]) / len(
                by_cls_acc[mhc_class]
            )
            stats[f"allele_swap_groups_{mhc_class}"] = float(len(by_cls_losses[mhc_class]))
    if not cls_means:
        return None, {}
    total = torch.stack(cls_means).mean()
    stats["allele_swap_loss"] = float(total.detach())
    return total, stats
