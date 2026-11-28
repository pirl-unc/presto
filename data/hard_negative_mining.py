"""Multi-negative affinity-matched hard-pair miner for detached processing supervision.

Mines up to ``neg_per_pos`` diverse, length-matched, same-protein unobserved
background negatives per TRAIN ``elution_observation`` positive, matched in
context-free binding affinity (``log10(IC50 nM)``) across monoallelic or
multiallelic MHC candidate sets.

Every emitted auxiliary background:
  - References the original positive via ``provenance["processing_pair_positive_id"]``
    (``PAIR_KEY``) so it trains ONLY the detached-binding ``processing_pair_training``
    pathway (``training/processing_pairs.py::PairPool``) and NEVER corrupts the
    binding affinity head via joint elution BCE.
  - Passes ``training.processing_pairs.validate_pair(positive, background)`` and
    ``data.affinity_matching.assess_match(..., policy).accepted``.
  - Passes ``ProtectedIdentityIndex`` (exact + 8-mer + Hamming-1 against VAL and
    sealed TEST; ``test_accessed=False``) and excludes all known MS-observed peptides.
  - Preserves exact knownness parity on ``(source_n, source_c, n_terminus, c_terminus)``
    and copies all non-flank APC/cell-type/component/cytokine/intervention context
    fields verbatim from the positive.
  - Enforces sequence-position separation and minimum Hamming distance between
    selected negatives for the same positive so each negative provides an
    independent flanking-cleavage, transport, and MS-detectability contrast.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from presto.data.affinity_matching import (
    AffinityScore,
    MatchingPolicy,
    assess_match,
    candidate_key,
)
from presto.data.hitlist_binding_expansion import AA_SET, ProtectedIdentityIndex
from presto.data.modular import Context, MHCInput, Observation, TargetScale
from presto.data.presentation import Protein, junction, stable_digest
from presto.training.processing_pairs import PAIR_KEY, validate_pair


@dataclass(frozen=True)
class MiningConfig:
    """Configuration for multi-negative affinity-matched hard-pair mining."""

    neg_per_pos: int = 3
    max_candidates_per_positive: int = 96
    max_log10_nm_gap: float = 0.25
    profile: str = "all"
    min_position_separation: int | None = None
    min_hamming_between_negs: int = 3
    flank_length: int = 15
    max_mhc_candidates: int = 6
    require_flanks: bool = False
    seed: int = 20261123
    batch_size: int = 512
    predictor_name: str = "Presto"
    predictor_version: str = "presto.omni.v2"

    def validate(self) -> None:
        if self.neg_per_pos < 1:
            raise ValueError("neg_per_pos must be >= 1")
        if self.max_candidates_per_positive < self.neg_per_pos:
            raise ValueError("max_candidates_per_positive must be >= neg_per_pos")
        if not math.isfinite(self.max_log10_nm_gap) or self.max_log10_nm_gap <= 0:
            raise ValueError("max_log10_nm_gap must be positive finite")
        if self.profile not in {"all", "winner"}:
            raise ValueError("profile must be 'all' or 'winner'")
        if self.min_position_separation is not None and self.min_position_separation < 1:
            raise ValueError("min_position_separation must be >= 1 when set")
        if self.min_hamming_between_negs < 1:
            raise ValueError("min_hamming_between_negs must be >= 1")
        if self.max_mhc_candidates < 1:
            raise ValueError("max_mhc_candidates must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.predictor_name not in {"Presto", "NetMHCpan", "NetMHCIIpan"}:
            raise ValueError("predictor_name must be Presto, NetMHCpan, or NetMHCIIpan")
        if not self.predictor_version.strip():
            raise ValueError("nonempty predictor_version required")

    def matching_policy(self) -> MatchingPolicy:
        policy = MatchingPolicy(
            metric="log10_nm",
            max_gap=self.max_log10_nm_gap,
            profile=self.profile,
            max_log_rank_gap=None,
        )
        policy.validate()
        return policy


class AffinityScorer(Protocol):
    """Context-free binding affinity scorer returning predicted log10(IC50 nM)."""

    def score_pairs(self, pairs: Sequence[tuple[str, MHCInput]]) -> list[float]:
        """Return predicted log10(nM) for each (peptide, MHCInput) pair."""
        ...


class PrestoAffinityScorer:
    """Batched, deduplicated context-free IC50 scorer backed by a Presto model."""

    def __init__(
        self,
        model: Any,
        scales: Mapping[str, TargetScale] | None = None,
        *,
        batch_size: int = 512,
        use_bf16: bool = True,
        endpoint: str = "ic50",
        cache_scores: bool = True,
    ) -> None:
        self.model = model
        self.model.eval()
        self.scale: TargetScale | None = (scales or {}).get(endpoint)
        self.batch_size = max(1, int(batch_size))
        self.use_bf16 = bool(use_bf16)
        self.endpoint = endpoint
        self._cache: dict[tuple[str, str], float] | None = {} if cache_scores else None

    def score_pairs(self, pairs: Sequence[tuple[str, MHCInput]]) -> list[float]:
        import torch

        if not pairs:
            return []

        keys = [(pep, candidate_key(mhc)) for pep, mhc in pairs]
        to_compute: dict[tuple[str, str], tuple[str, MHCInput]] = {}
        for (pep, mhc), key in zip(pairs, keys, strict=True):
            if self._cache is not None and key in self._cache:
                continue
            if key not in to_compute:
                to_compute[key] = (pep, mhc)

        computed: dict[tuple[str, str], float] = {}
        if to_compute:
            items = list(to_compute.items())
            param = next(self.model.parameters(), None)
            dev_type = param.device.type if param is not None else "cpu"
            amp_ok = self.use_bf16 and dev_type == "cuda"

            with torch.no_grad():
                for start in range(0, len(items), self.batch_size):
                    chunk = items[start : start + self.batch_size]
                    obs_batch = [
                        Observation(
                            observation_id=f"score:{start + j}",
                            source="hard_neg_miner",
                            endpoint=self.endpoint,
                            peptide=pep,
                            mhc=(mhc,),
                            value=None,
                            unit=None,
                            split="predict",
                            context=Context(),
                        )
                        for j, (_k, (pep, mhc)) in enumerate(chunk)
                    ]
                    if amp_ok:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            preds = self.model(obs_batch)
                    else:
                        preds = self.model(obs_batch)
                    for (k, _pm), pred in zip(chunk, preds, strict=True):
                        raw_val = float(pred.value.detach().float().cpu())
                        val = self.scale.decode(raw_val) if self.scale is not None else raw_val
                        val = max(-4.0, min(9.0, val))
                        computed[k] = val
                        if self._cache is not None:
                            self._cache[k] = val

        out: list[float] = []
        for k in keys:
            if self._cache is not None and k in self._cache:
                out.append(self._cache[k])
            else:
                out.append(computed[k])
        return out


@dataclass(frozen=True)
class _CandidateWindow:
    peptide: str
    start: int
    context_fields: dict[str, Any]


def _protein_sequence(proteins: Mapping[str, Protein | str], protein_id: str) -> str | None:
    entry = proteins.get(protein_id)
    if entry is None:
        return None
    return entry.sequence if isinstance(entry, Protein) else str(entry)


def _positive_starts(sequence: str, peptide: str) -> tuple[int, ...]:
    starts: list[int] = []
    pos = sequence.find(peptide, 0)
    while pos != -1:
        starts.append(pos)
        pos = sequence.find(peptide, pos + 1)
    return tuple(starts)


def _match_flank_knownness(
    pos_context: Context,
    raw_junc: dict[str, Any],
) -> dict[str, Any] | None:
    """Enforce exact knownness parity on (source_n, source_c, n_terminus, c_terminus)."""
    out: dict[str, Any] = {}
    for field in ("source_n", "source_c", "n_terminus", "c_terminus"):
        if getattr(pos_context, field) is None:
            out[field] = None
        else:
            val = raw_junc[field]
            if val is None:
                return None
            out[field] = val
    return out


def _enumerate_candidate_windows(
    positive: Observation,
    sequence: str,
    *,
    observed: set[str] | frozenset[str],
    protected: ProtectedIdentityIndex,
    config: MiningConfig,
    pos_starts: tuple[int, ...],
) -> list[_CandidateWindow]:
    """Generate eligible, non-embargoed, same-length windows from the positive's protein."""
    L = len(positive.peptide)
    n_pos = len(sequence) - L + 1
    if n_pos <= 0:
        return []

    min_sep = (
        config.min_position_separation
        if config.min_position_separation is not None
        else max(3, L // 2)
    )
    key = f"hard_neg_scan:{config.seed}:{positive.observation_id}"
    seed_int = int(stable_digest(key), 16)
    start_at = seed_int % n_pos
    stride = (seed_int // n_pos) % n_pos or 1
    while math.gcd(stride, n_pos) != 1:
        stride += 1

    candidates: list[_CandidateWindow] = []
    seen_peps: set[str] = {positive.peptide}
    for offset in range(n_pos):
        if len(candidates) >= config.max_candidates_per_positive:
            break
        st = (start_at + offset * stride) % n_pos
        if pos_starts and any(abs(st - ps) < min_sep for ps in pos_starts):
            continue
        sub = sequence[st : st + L]
        if sub in seen_peps or sub in observed or not (set(sub) <= AA_SET):
            continue
        raw_junc = junction(sequence, st, L, flank_length=config.flank_length)
        ctx_fields = _match_flank_knownness(positive.context, raw_junc)
        if ctx_fields is None:
            continue
        if protected.reason(sub) is not None:
            continue
        seen_peps.add(sub)
        candidates.append(_CandidateWindow(peptide=sub, start=st, context_fields=ctx_fields))
    return candidates


def _empirical_ranks(values_by_cand: Sequence[float]) -> list[float]:
    """Empirical within-protein percentile rank in [0.01, 100.0] (lower log10_nm = better)."""
    n = len(values_by_cand)
    if n <= 1:
        return [50.0] * n
    sorted_vals = sorted(values_by_cand)
    ranks: list[float] = []
    for v in values_by_cand:
        # Count strictly stronger (smaller log10_nm) peptides
        lo = 0
        hi = n
        while lo < hi:
            mid = (lo + hi) // 2
            if sorted_vals[mid] < v:
                lo = mid + 1
            else:
                hi = mid
        pct = 100.0 * (lo + 0.5) / n
        ranks.append(round(max(0.01, min(100.0, pct)), 4))
    return ranks


def _hamming_distance(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b, strict=True))


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def mine_affinity_matched_backgrounds(
    positives: Sequence[Observation],
    proteins: Mapping[str, Protein | str],
    scorer: AffinityScorer,
    *,
    observed: set[str] | frozenset[str],
    protected: ProtectedIdentityIndex,
    config: MiningConfig | None = None,
    source_sha256: str = "",
    out_path: Path | None = None,
) -> tuple[list[Observation], dict[str, Any]]:
    """Mine up to ``config.neg_per_pos`` diverse affinity-matched backgrounds per positive.

    Returns ``(backgrounds, stats)``. Every emitted background is validated with
    ``training.processing_pairs.validate_pair(positive, background)`` and is ready
    for direct ingestion by ``PairPool``.
    """
    cfg = config or MiningConfig()
    cfg.validate()
    policy = cfg.matching_policy()
    src_sha = source_sha256.strip() or stable_digest(
        f"mine_source:{cfg.predictor_name}:{cfg.predictor_version}:{len(positives)}"
    )

    counts: Counter[str] = Counter()
    jobs: list[tuple[Observation, str, tuple[str, ...], list[_CandidateWindow]]] = []
    seen_pos_ids: set[str] = set()

    for pos in positives:
        counts["positives_seen"] += 1
        if (
            pos.split != "train"
            or pos.endpoint != "elution_observation"
            or pos.unit != "binary"
            or pos.qualifier != "="
            or pos.value != 1
        ):
            counts["drop_not_train_elution_pos"] += 1
            continue
        if pos.observation_id in seen_pos_ids:
            counts["drop_duplicate_pos_id"] += 1
            continue
        seen_pos_ids.add(pos.observation_id)

        if not pos.mhc or len(pos.mhc) > cfg.max_mhc_candidates:
            counts["drop_mhc_candidate_count"] += 1
            continue
        classes = {m.class_override or m.mhc_class for m in pos.mhc}
        if len(classes) != 1 or next(iter(classes)) not in ("I", "II"):
            counts["drop_unresolved_or_mixed_class"] += 1
            continue
        if not pos.group_id or not pos.provenance.get("study_id"):
            counts["drop_missing_group_or_study"] += 1
            continue
        if cfg.require_flanks and (pos.context.source_n is None or pos.context.source_c is None):
            counts["drop_missing_flanks"] += 1
            continue

        prot_id = str(
            pos.provenance.get("source_protein") or pos.provenance.get("protein_id") or ""
        ).strip()
        seq = _protein_sequence(proteins, prot_id) if prot_id else None
        if seq is None or len(seq) < len(pos.peptide) + 2:
            counts["drop_missing_source_protein"] += 1
            continue

        pos_starts = _positive_starts(seq, pos.peptide)
        cand_windows = _enumerate_candidate_windows(
            pos,
            seq,
            observed=observed,
            protected=protected,
            config=cfg,
            pos_starts=pos_starts,
        )
        if not cand_windows:
            counts["drop_no_eligible_candidates"] += 1
            continue

        mhc_keys = tuple(candidate_key(m) for m in pos.mhc)
        if len(set(mhc_keys)) != len(mhc_keys):
            counts["drop_duplicate_mhc_candidates"] += 1
            continue
        jobs.append((pos, prot_id, mhc_keys, cand_windows))

    backgrounds: list[Observation] = []
    by_class_pairs: Counter[str] = Counter()
    by_class_positives_matched: Counter[str] = Counter()
    negs_per_pos_hist: Counter[int] = Counter()
    distances_by_class: dict[str, list[float]] = {"I": [], "II": []}

    # Score jobs in chunks so memory stays bounded while saturating GPU batches
    chunk_size = max(1, cfg.batch_size // max(1, min(cfg.max_candidates_per_positive, 32)))
    for c0 in range(0, len(jobs), chunk_size):
        chunk = jobs[c0 : c0 + chunk_size]
        score_requests: list[tuple[str, MHCInput]] = []
        slices: list[tuple[int, int, int]] = []
        for pos, _prot_id, _mhc_keys, cand_windows in chunk:
            st = len(score_requests)
            n_mhc = len(pos.mhc)
            for mhc in pos.mhc:
                score_requests.append((pos.peptide, mhc))
            for cw in cand_windows:
                for mhc in pos.mhc:
                    score_requests.append((cw.peptide, mhc))
            slices.append((st, len(cand_windows), n_mhc))

        scored_log10 = scorer.score_pairs(score_requests)
        if len(scored_log10) != len(score_requests):
            raise RuntimeError("AffinityScorer returned wrong number of predictions")

        for (pos, prot_id, mhc_keys, cand_windows), (st, n_cands, n_mhc) in zip(
            chunk, slices, strict=True
        ):
            cls = pos.mhc[0].class_override or pos.mhc[0].mhc_class
            assert cls in ("I", "II")
            L = len(pos.peptide)
            min_sep = (
                cfg.min_position_separation
                if cfg.min_position_separation is not None
                else max(3, L // 2)
            )

            # Reshape scores: row 0 = positive, rows 1..n_cands = candidates; col j = mhc j
            mat = [scored_log10[st + r * n_mhc : st + (r + 1) * n_mhc] for r in range(n_cands + 1)]
            # Per-MHC empirical percentile ranks across (positive + candidates)
            ranks_by_mhc = [
                _empirical_ranks([mat[r][j] for r in range(n_cands + 1)]) for j in range(n_mhc)
            ]

            pos_scores = {
                mhc_keys[j]: AffinityScore(
                    nm=float(10.0 ** mat[0][j]),
                    rank=ranks_by_mhc[j][0],
                )
                for j in range(n_mhc)
            }

            accepted: list[tuple[float, str, _CandidateWindow, dict[str, AffinityScore], Any]] = []
            for ci, cw in enumerate(cand_windows):
                bg_scores = {
                    mhc_keys[j]: AffinityScore(
                        nm=float(10.0 ** mat[ci + 1][j]),
                        rank=ranks_by_mhc[j][ci + 1],
                    )
                    for j in range(n_mhc)
                }
                assessment = assess_match(pos_scores, bg_scores, policy)
                if assessment.accepted:
                    tie = stable_digest(f"{cfg.seed}|{pos.observation_id}|{cw.peptide}")
                    accepted.append((assessment.distance, tie, cw, bg_scores, assessment))

            if not accepted:
                counts[f"drop_no_caliper_match_{cls}"] += 1
                continue

            accepted.sort(key=lambda item: (item[0], item[1]))

            selected: list[tuple[_CandidateWindow, dict[str, AffinityScore], Any]] = []
            for _dist, _tie, cw, bg_scores, assessment in accepted:
                if len(selected) >= cfg.neg_per_pos:
                    break
                if any(
                    abs(cw.start - prev_cw.start) < min_sep
                    or _hamming_distance(cw.peptide, prev_cw.peptide) < cfg.min_hamming_between_negs
                    for prev_cw, _, _ in selected
                ):
                    counts["skip_redundant_overlap_neg"] += 1
                    continue
                selected.append((cw, bg_scores, assessment))

            if not selected:
                counts[f"drop_no_caliper_match_{cls}"] += 1
                continue

            by_class_positives_matched[cls] += 1
            negs_per_pos_hist[len(selected)] += 1

            pos_scores_dict = {k: asdict(v) for k, v in pos_scores.items()}
            for rep, (cw, bg_scores, assessment) in enumerate(selected):
                bg_scores_dict = {k: asdict(v) for k, v in bg_scores.items()}
                score_payload = json.dumps(
                    {"pos": pos_scores_dict, "bg": bg_scores_dict},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                score_sha = hashlib.sha256(score_payload.encode("utf-8")).hexdigest()
                bg_id = (
                    "hardpair:"
                    + stable_digest(f"{cfg.seed}|{pos.observation_id}|{rep}|{cw.peptide}")[:20]
                )
                bg_context = replace(pos.context, **cw.context_fields)
                bg_obs = Observation(
                    observation_id=bg_id,
                    source="presto_affinity_matched_processing_background",
                    endpoint="elution_observation",
                    peptide=cw.peptide,
                    mhc=pos.mhc,
                    value=0.0,
                    unit="binary",
                    qualifier="=",
                    split="train",
                    group_id=pos.group_id,
                    context=bg_context,
                    provenance={
                        PAIR_KEY: pos.observation_id,
                        "synthetic": True,
                        "source_protein": prot_id,
                        "study_id": pos.provenance["study_id"],
                        "generated_position": cw.start,
                        "replicate": rep,
                        "measurement_lineage": {
                            "namespace": "presto_affinity_matched_processing_background",
                            "measurement_id": bg_id,
                        },
                        "affinity_match": {
                            "schema": "presto.affinity_match.v1",
                            "predictor": {
                                "name": cfg.predictor_name,
                                "version": cfg.predictor_version,
                            },
                            "source_sha256": src_sha,
                            "score_sha256": score_sha,
                            "policy": asdict(policy),
                            "positive_scores": pos_scores_dict,
                            "background_scores": bg_scores_dict,
                            "distance_log10_nm": round(float(assessment.distance), 6),
                            "winning_candidate": assessment.positive_winner,
                        },
                        "test_accessed": False,
                    },
                )
                bg_obs.validate()
                validate_pair(pos, bg_obs)
                backgrounds.append(bg_obs)
                by_class_pairs[cls] += 1
                distances_by_class[cls].append(float(assessment.distance))

    if out_path is not None:
        _write_jsonl(out_path, [b.to_dict() for b in backgrounds])

    mean_dist = {
        cls: round(sum(vals) / len(vals), 5) if vals else None
        for cls, vals in distances_by_class.items()
    }
    stats: dict[str, Any] = {
        "test_accessed": False,
        "config": asdict(cfg),
        "positives_evaluated": len(jobs),
        "positives_matched_by_class": dict(sorted(by_class_positives_matched.items())),
        "backgrounds_emitted": len(backgrounds),
        "backgrounds_by_class": dict(sorted(by_class_pairs.items())),
        "negatives_per_matched_positive_hist": {
            str(k): v for k, v in sorted(negs_per_pos_hist.items())
        },
        "mean_log10_nm_distance": mean_dist,
        "counts": dict(sorted(counts.items())),
        "out_path": str(out_path) if out_path is not None else None,
    }
    return backgrounds, stats
