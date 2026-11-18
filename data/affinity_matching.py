"""Allele-resolved affinity balance for observed-versus-sampled peptide pairs.

Only BA predictions belong here. Percentile values are in percentage points,
not fractions or EL ranks. Acceptance does not turn a sampled peptide into an
experimentally measured negative or identify its presenting allele.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping


def candidate_key(mhc):
    """Sequence/class identity, independent of allele spelling and provenance."""
    value = mhc if isinstance(mhc, dict) else vars(mhc)
    cls = value.get("class_override") or value["mhc_class"]
    if cls not in ("I", "II"):
        raise ValueError("affinity matching requires resolved class")
    sequence = value["groove1"] + "|" + value["groove2"]
    return cls + ":" + hashlib.sha256(sequence.encode()).hexdigest()


@dataclass(frozen=True)
class AffinityScore:
    nm: float
    rank: float

    def validate(self):
        if not math.isfinite(self.nm) or self.nm <= 0:
            raise ValueError("BA nM must be positive and finite")
        if not math.isfinite(self.rank) or not 0 <= self.rank <= 100:
            raise ValueError("BA percentile rank must be finite in [0, 100]")


@dataclass(frozen=True)
class MatchingPolicy:
    metric: str = "ba_rank"
    max_gap: float = 0.5
    profile: str = "all"
    max_log_rank_gap: float | None = 0.3
    rank_floor: float = 0.01

    def validate(self):
        if self.metric not in {"ba_rank", "log10_nm"}:
            raise ValueError("matching requires BA rank or BA log10(nM); EL is excluded")
        if self.profile not in {"all", "winner"}:
            raise ValueError("profile must be all candidates or winner only")
        if not math.isfinite(self.max_gap) or self.max_gap <= 0:
            raise ValueError("positive finite affinity caliper required")
        if not math.isfinite(self.rank_floor) or self.rank_floor <= 0:
            raise ValueError("positive finite rank floor required for printed zero ranks")
        if self.max_log_rank_gap is not None and (
            self.metric != "ba_rank"
            or not math.isfinite(self.max_log_rank_gap)
            or self.max_log_rank_gap <= 0
        ):
            raise ValueError("relative rank caliper requires BA rank and a positive finite gap")


@dataclass(frozen=True)
class MatchAssessment:
    accepted: bool
    reason: str
    positive_winner: str
    negative_winner: str
    distance: float
    max_rank_gap: float
    max_log_rank_gap: float
    max_log_nm_gap: float


def assess_match(
    positive: Mapping[str, AffinityScore],
    background: Mapping[str, AffinityScore],
    policy: MatchingPolicy,
) -> MatchAssessment:
    """Require identical typed candidates, the same winner and declared balance.

    `winner` is a proxy choice, never a measured restriction. `all` prevents a
    minimum-only match from hiding different binding to another typed molecule.
    Diagnostics cover every candidate even for the legacy winner-only rule.
    """
    policy.validate()
    if not positive or set(positive) != set(background):
        raise ValueError("identical nonempty typed candidate sets are required")
    if any(not isinstance(key, str) or not key for key in positive):
        raise ValueError("stable nonempty candidate identity keys required")
    for scores in (positive, background):
        for score in scores.values():
            score.validate()
    keys = sorted(positive)
    metric = (lambda s: s.rank) if policy.metric == "ba_rank" else (lambda s: math.log10(s.nm))
    pw = min(keys, key=lambda k: (metric(positive[k]), k))
    nw = min(keys, key=lambda k: (metric(background[k]), k))
    rank_gaps = {k: abs(positive[k].rank - background[k].rank) for k in keys}
    log_rank_gaps = {
        k: abs(
            math.log10(
                (positive[k].rank + policy.rank_floor) / (background[k].rank + policy.rank_floor)
            )
        )
        for k in keys
    }
    log_nm_gaps = {k: abs(math.log10(positive[k].nm / background[k].nm)) for k in keys}
    selected = keys if policy.profile == "all" else [pw]
    gaps = rank_gaps if policy.metric == "ba_rank" else log_nm_gaps
    distance = max(gaps[k] for k in selected)
    reason = "matched"
    if pw != nw:
        reason = "different_winning_candidate"
    elif distance > policy.max_gap + 1e-12:
        reason = "absolute_caliper"
    elif (
        policy.max_log_rank_gap is not None
        and max(log_rank_gaps[k] for k in selected) > policy.max_log_rank_gap + 1e-12
    ):
        reason = "relative_rank_caliper"
    return MatchAssessment(
        reason == "matched",
        reason,
        pw,
        nw,
        distance,
        max(rank_gaps.values()),
        max(log_rank_gaps.values()),
        max(log_nm_gaps.values()),
    )
