"""Groove-centric MHC sequence parsing utilities.

This module extracts structurally relevant groove halves from full MHC chains:

- Class I alpha chain -> alpha1 groove half + alpha2 groove half
- Class II alpha chain -> alpha1 groove half
- Class II beta chain -> beta1 groove half

The implementation is deliberately heuristic and alignment-free. It relies on
conserved intrachain disulfide spacing and falls back conservatively when a
primary anchor cannot be identified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

from .allele_resolver import infer_gene, normalize_mhc_class


IG_SEP_MIN = 48
IG_SEP_MAX = 72

MIN_GROOVE_SOURCE_LEN = 70

CLASS_I_ALPHA2_CYS1_MATURE_POS = 101
CLASS_I_ALPHA3_CYS1_MATURE_POS = 203
CLASS_I_ALPHA2_CYS1_OFFSET = 10
CLASS_I_ALPHA2_END_AFTER_CYS2 = 20
CLASS_I_ALPHA2_CYS1_RAW_MIN = 60
CLASS_I_ALPHA2_CYS1_RAW_MAX = 180
CLASS_I_ALPHA3_CYS1_RAW_MIN = 180

CLASS_II_ALPHA_IG_CYS1_MATURE_POS = 107
CLASS_II_ALPHA_GROOVE_END_BEFORE_IG_CYS = 23
CLASS_II_ALPHA_CYS1_RAW_PRIMARY_MIN = 100
CLASS_II_ALPHA_CYS1_RAW_MIN = 80
CLASS_II_ALPHA_CYS1_RAW_MAX = 160

CLASS_II_BETA1_CYS1_MATURE_POS = 15
CLASS_II_BETA2_CYS1_MATURE_POS = 117
CLASS_II_BETA1_CYS1_RAW_MIN = 20
CLASS_II_BETA1_CYS1_RAW_MAX = 95
CLASS_II_BETA2_CYS1_RAW_MIN = 100
CLASS_II_BETA2_CYS1_RAW_MAX = 180
CLASS_II_BETA_GROOVE_END_BEFORE_BETA2_CYS = 23
CLASS_II_BETA1_ONLY_END_AFTER_CYS2 = 15

DEFAULT_CLASS_I_GROOVE_HALF_1_LEN = 91
DEFAULT_CLASS_I_GROOVE_HALF_2_LEN = 93
DEFAULT_CLASS_II_ALPHA_GROOVE_LEN = 84
DEFAULT_CLASS_II_BETA_GROOVE_LEN = 94
CLASS_II_ALPHA_FRAGMENT_MAX_LEN = 110
CLASS_II_BETA_FRAGMENT_MAX_LEN = 120

CLASS_II_ALPHA_GENE_PREFIXES = ("DRA", "DQA", "DPA", "DMA", "DOA")
CLASS_II_BETA_GENE_PREFIXES = ("DRB", "DQB", "DPB", "DMB", "DOB")


@dataclass(frozen=True)
class GrooveResult:
    """Result of parsing one MHC chain into groove halves."""

    allele: str = ""
    gene: str = ""
    mhc_class: str = ""
    chain: str = ""
    seq_len: int = 0
    mature_start: int = 0
    groove_seq: str = ""
    groove_half_1: str = ""
    groove_half_2: str = ""
    groove_h1_len: int = 0
    groove_h2_len: int = 0
    status: str = "ok"
    anchor_type: str = ""
    anchor_cys1: Optional[int] = None
    anchor_cys2: Optional[int] = None
    secondary_cys1: Optional[int] = None
    secondary_cys2: Optional[int] = None
    flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.status in {
            "ok",
            "alpha3_fallback",
            "beta1_only_fallback",
            "fragment_fallback",
        }


@dataclass(frozen=True)
class PreparedMHCInput:
    """Groove halves prepared for runtime use."""

    mhc_class: str
    groove_half_1: str
    groove_half_2: str
    groove_status_a: str
    groove_status_b: str = ""
    used_fallback: bool = False


def _clean_seq(sequence: Optional[str]) -> str:
    return "".join(ch for ch in str(sequence or "").strip().upper() if not ch.isspace())


def _infer_mature_start(cys1_raw: int, mature_pos: int) -> int:
    return max(0, int(cys1_raw) - int(mature_pos))


def _flags_to_tuple(flags: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(flag) for flag in flags if flag)


def find_cys_pairs(
    seq: str,
    min_sep: int = IG_SEP_MIN,
    max_sep: int = IG_SEP_MAX,
) -> list[tuple[int, int, int]]:
    """Find all Cys-Cys pairs with plausible Ig-fold separation."""

    cleaned = _clean_seq(seq)
    cys_positions = [idx for idx, aa in enumerate(cleaned) if aa == "C"]
    pairs: list[tuple[int, int, int]] = []
    for i, c1 in enumerate(cys_positions):
        for c2 in cys_positions[i + 1 :]:
            sep = c2 - c1
            if sep < min_sep:
                continue
            if sep > max_sep:
                break
            pairs.append((c1, c2, sep))
    return pairs


def classify_cys_pair(seq: str, c1: int, c2: int) -> str:
    """Coarsely classify a Cys pair using local motif context.

    The local `CW...` motif is a weak indicator of a class-I alpha3-like Ig
    domain. The signal is not strong enough to drive the parser by itself, so
    this function is intended for diagnostics and soft filtering only.
    """

    cleaned = _clean_seq(seq)
    after_c1 = cleaned[c1 : c1 + 8]
    after_c2 = cleaned[c2 : c2 + 8]
    if after_c1.startswith("CW") or after_c2.startswith("CW"):
        return "alpha3"
    return "ig_generic"


def _slice_or_empty(seq: str, start: int, end: int) -> str:
    lo = max(0, int(start))
    hi = max(lo, min(len(seq), int(end)))
    return seq[lo:hi]


def _class_ii_chain_from_name(
    *,
    gene: str,
    allele: str,
) -> Optional[str]:
    gene_token = str(gene or "").strip().upper()
    if not gene_token and allele:
        try:
            gene_token = infer_gene(allele)
        except Exception:
            gene_token = ""
        gene_token = str(gene_token or "").strip().upper()

    if not gene_token:
        return None
    if gene_token.startswith(CLASS_II_ALPHA_GENE_PREFIXES):
        return "alpha"
    if gene_token.startswith(CLASS_II_BETA_GENE_PREFIXES):
        return "beta"
    if gene_token.endswith("A"):
        return "alpha"
    if gene_token.endswith("B"):
        return "beta"
    return None


def _infer_class_ii_chain_from_sequence(
    *,
    seq: str,
    allele: str,
    gene: str,
) -> tuple[Optional[str], GrooveResult, GrooveResult]:
    alpha = parse_class_ii_alpha(seq, allele=allele, gene=gene)
    beta = parse_class_ii_beta(seq, allele=allele, gene=gene)

    if alpha.ok and not beta.ok:
        return "alpha", alpha, beta
    if beta.ok and not alpha.ok:
        return "beta", alpha, beta
    return None, alpha, beta


def _class_ii_chain_inference_failure(
    *,
    seq: str,
    allele: str,
    gene: str,
    alpha_result: GrooveResult,
    beta_result: GrooveResult,
) -> GrooveResult:
    if alpha_result.ok and beta_result.ok:
        status = "ambiguous_chain"
    else:
        status = "chain_inference_failed"
    return GrooveResult(
        allele=allele,
        gene=gene,
        mhc_class="II",
        chain="",
        seq_len=len(_clean_seq(seq)),
        status=status,
        anchor_type="chain_inference",
        flags=_flags_to_tuple(
            (
                f"alpha_status={alpha_result.status}",
                f"beta_status={beta_result.status}",
            )
        ),
    )


def _class_ii_fragment_result(
    *,
    seq: str,
    allele: str,
    gene: str,
    chain: str,
) -> GrooveResult:
    cleaned = _clean_seq(seq)
    if chain == "alpha":
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="II",
            chain="alpha",
            seq_len=len(cleaned),
            mature_start=0,
            groove_seq=cleaned,
            groove_half_1=cleaned,
            groove_half_2="",
            groove_h1_len=len(cleaned),
            groove_h2_len=0,
            status="fragment_fallback",
            anchor_type="raw_fragment",
            flags=("fragment_fallback",),
        )
    return GrooveResult(
        allele=allele,
        gene=gene,
        mhc_class="II",
        chain="beta",
        seq_len=len(cleaned),
        mature_start=0,
        groove_seq=cleaned,
        groove_half_1="",
        groove_half_2=cleaned,
        groove_h1_len=0,
        groove_h2_len=len(cleaned),
        status="fragment_fallback",
        anchor_type="raw_fragment",
        flags=("fragment_fallback",),
    )


def parse_class_i(
    seq: str,
    *,
    allele: str = "",
    gene: str = "",
) -> GrooveResult:
    """Parse a class-I alpha chain into alpha1/alpha2 groove halves."""

    cleaned = _clean_seq(seq)
    flags: list[str] = []
    if len(cleaned) < MIN_GROOVE_SOURCE_LEN:
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="I",
            chain="alpha",
            seq_len=len(cleaned),
            status="too_short",
        )

    pairs = find_cys_pairs(cleaned)
    if not pairs:
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="I",
            chain="alpha",
            seq_len=len(cleaned),
            status="no_cys_pairs",
        )

    alpha2_candidates = [
        pair
        for pair in pairs
        if CLASS_I_ALPHA2_CYS1_RAW_MIN <= pair[0] <= CLASS_I_ALPHA2_CYS1_RAW_MAX
    ]
    alpha2_pair = alpha2_candidates[0] if alpha2_candidates else None

    if alpha2_pair is None:
        alpha3_pair = next(
            (pair for pair in pairs if pair[0] >= CLASS_I_ALPHA3_CYS1_RAW_MIN),
            None,
        )
        if alpha3_pair is None:
            return GrooveResult(
                allele=allele,
                gene=gene,
                mhc_class="I",
                chain="alpha",
                seq_len=len(cleaned),
                status="no_alpha2_pair",
            )

        c1, c2, _ = alpha3_pair
        mature_start = _infer_mature_start(c1, CLASS_I_ALPHA3_CYS1_MATURE_POS)
        alpha1_end = mature_start + DEFAULT_CLASS_I_GROOVE_HALF_1_LEN
        alpha2_end = min(len(cleaned), c1 - 20)
        half_1 = _slice_or_empty(cleaned, mature_start, alpha1_end)
        half_2 = _slice_or_empty(cleaned, alpha1_end, alpha2_end)
        if not half_1 or not half_2:
            return GrooveResult(
                allele=allele,
                gene=gene,
                mhc_class="I",
                chain="alpha",
                seq_len=len(cleaned),
                mature_start=mature_start,
                status="alpha3_fallback_bad_boundaries",
                anchor_type="alpha3_cys",
                anchor_cys1=c1,
                anchor_cys2=c2,
            )
        flags.append("alpha3_fallback")
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="I",
            chain="alpha",
            seq_len=len(cleaned),
            mature_start=mature_start,
            groove_seq=half_1 + half_2,
            groove_half_1=half_1,
            groove_half_2=half_2,
            groove_h1_len=len(half_1),
            groove_h2_len=len(half_2),
            status="alpha3_fallback",
            anchor_type="alpha3_cys",
            anchor_cys1=c1,
            anchor_cys2=c2,
            flags=_flags_to_tuple(flags),
        )

    c1, c2, _ = alpha2_pair
    mature_start = _infer_mature_start(c1, CLASS_I_ALPHA2_CYS1_MATURE_POS)
    if mature_start > 40:
        flags.append(f"long_sp({mature_start})")

    alpha2_start = max(mature_start, c1 - CLASS_I_ALPHA2_CYS1_OFFSET)
    alpha2_end = min(len(cleaned), c2 + CLASS_I_ALPHA2_END_AFTER_CYS2)
    half_1 = _slice_or_empty(cleaned, mature_start, alpha2_start)
    half_2 = _slice_or_empty(cleaned, alpha2_start, alpha2_end)
    secondary = next(
        (
            pair
            for pair in pairs
            if pair[0] > c2 + 10
        ),
        None,
    )
    if len(half_1) < 50:
        flags.append(f"alpha1_short({len(half_1)})")
    if len(half_2) < 60:
        flags.append(f"alpha2_short({len(half_2)})")
    if not half_1 or not half_2:
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="I",
            chain="alpha",
            seq_len=len(cleaned),
            mature_start=mature_start,
            status="invalid_boundaries",
            anchor_type="alpha2_cys",
            anchor_cys1=c1,
            anchor_cys2=c2,
        )
    return GrooveResult(
        allele=allele,
        gene=gene,
        mhc_class="I",
        chain="alpha",
        seq_len=len(cleaned),
        mature_start=mature_start,
        groove_seq=half_1 + half_2,
        groove_half_1=half_1,
        groove_half_2=half_2,
        groove_h1_len=len(half_1),
        groove_h2_len=len(half_2),
        status="ok",
        anchor_type="alpha2_cys",
        anchor_cys1=c1,
        anchor_cys2=c2,
        secondary_cys1=(secondary[0] if secondary else None),
        secondary_cys2=(secondary[1] if secondary else None),
        flags=_flags_to_tuple(flags),
    )


def parse_class_ii_alpha(
    seq: str,
    *,
    allele: str = "",
    gene: str = "",
) -> GrooveResult:
    """Parse a class-II alpha chain into the alpha1 groove half."""

    cleaned = _clean_seq(seq)
    flags: list[str] = []
    if len(cleaned) < MIN_GROOVE_SOURCE_LEN:
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="II",
            chain="alpha",
            seq_len=len(cleaned),
            status="too_short",
        )

    pairs = find_cys_pairs(cleaned)
    if not pairs:
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="II",
            chain="alpha",
            seq_len=len(cleaned),
            status="no_cys_pairs",
        )

    primary = [
        pair
        for pair in pairs
        if CLASS_II_ALPHA_CYS1_RAW_PRIMARY_MIN <= pair[0] <= CLASS_II_ALPHA_CYS1_RAW_MAX
    ]
    candidates = primary or [
        pair
        for pair in pairs
        if CLASS_II_ALPHA_CYS1_RAW_MIN <= pair[0] <= CLASS_II_ALPHA_CYS1_RAW_MAX
    ]
    if not candidates:
        if len(cleaned) <= CLASS_II_ALPHA_FRAGMENT_MAX_LEN:
            return _class_ii_fragment_result(
                seq=cleaned,
                allele=allele,
                gene=gene,
                chain="alpha",
            )
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="II",
            chain="alpha",
            seq_len=len(cleaned),
            status="no_anchor_pair",
        )

    c1, c2, _ = min(candidates, key=lambda item: (abs(item[2] - 56), -item[0]))
    mature_start = _infer_mature_start(c1, CLASS_II_ALPHA_IG_CYS1_MATURE_POS)
    groove_end = max(mature_start, c1 - CLASS_II_ALPHA_GROOVE_END_BEFORE_IG_CYS)
    half_1 = _slice_or_empty(cleaned, mature_start, groove_end)
    if len(half_1) < 60:
        flags.append(f"alpha1_short({len(half_1)})")
    if not half_1:
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="II",
            chain="alpha",
            seq_len=len(cleaned),
            mature_start=mature_start,
            status="invalid_boundaries",
            anchor_type="alpha2_cys",
            anchor_cys1=c1,
            anchor_cys2=c2,
        )
    return GrooveResult(
        allele=allele,
        gene=gene,
        mhc_class="II",
        chain="alpha",
        seq_len=len(cleaned),
        mature_start=mature_start,
        groove_seq=half_1,
        groove_half_1=half_1,
        groove_half_2="",
        groove_h1_len=len(half_1),
        groove_h2_len=0,
        status="ok",
        anchor_type="alpha2_cys",
        anchor_cys1=c1,
        anchor_cys2=c2,
        flags=_flags_to_tuple(flags),
    )


def parse_class_ii_beta(
    seq: str,
    *,
    allele: str = "",
    gene: str = "",
) -> GrooveResult:
    """Parse a class-II beta chain into the beta1 groove half."""

    cleaned = _clean_seq(seq)
    flags: list[str] = []
    if len(cleaned) < MIN_GROOVE_SOURCE_LEN:
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="II",
            chain="beta",
            seq_len=len(cleaned),
            status="too_short",
        )

    pairs = find_cys_pairs(cleaned)
    if not pairs:
        if len(cleaned) <= CLASS_II_BETA_FRAGMENT_MAX_LEN:
            return _class_ii_fragment_result(
                seq=cleaned,
                allele=allele,
                gene=gene,
                chain="beta",
            )
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="II",
            chain="beta",
            seq_len=len(cleaned),
            status="no_cys_pairs",
        )

    beta1_candidates = [
        pair
        for pair in pairs
        if CLASS_II_BETA1_CYS1_RAW_MIN <= pair[0] <= CLASS_II_BETA1_CYS1_RAW_MAX
    ]
    beta2_candidates = [
        pair
        for pair in pairs
        if CLASS_II_BETA2_CYS1_RAW_MIN <= pair[0] <= CLASS_II_BETA2_CYS1_RAW_MAX
    ]

    beta1_pair = (
        min(beta1_candidates, key=lambda item: (abs(item[2] - 64), item[0]))
        if beta1_candidates
        else None
    )
    downstream_beta2 = [
        pair
        for pair in beta2_candidates
        if beta1_pair is None or pair[0] > beta1_pair[1] + 10
    ]
    beta2_pair = (
        min(downstream_beta2, key=lambda item: (abs(item[2] - 56), item[0]))
        if downstream_beta2
        else None
    )

    if beta1_pair is None and beta2_pair is None:
        if len(cleaned) <= CLASS_II_BETA_FRAGMENT_MAX_LEN:
            return _class_ii_fragment_result(
                seq=cleaned,
                allele=allele,
                gene=gene,
                chain="beta",
            )
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="II",
            chain="beta",
            seq_len=len(cleaned),
            status="no_anchor_pair",
        )

    if beta2_pair is not None:
        c1, c2, _ = beta2_pair
        mature_start = _infer_mature_start(c1, CLASS_II_BETA2_CYS1_MATURE_POS)
        groove_end = max(mature_start, c1 - CLASS_II_BETA_GROOVE_END_BEFORE_BETA2_CYS)
        status = "ok"
        anchor_type = "beta2_cys"
        anchor_pair = beta2_pair
    else:
        c1, c2, _ = beta1_pair  # type: ignore[misc]
        mature_start = _infer_mature_start(c1, CLASS_II_BETA1_CYS1_MATURE_POS)
        groove_end = min(len(cleaned), c2 + CLASS_II_BETA1_ONLY_END_AFTER_CYS2)
        status = "beta1_only_fallback"
        anchor_type = "beta1_cys"
        anchor_pair = beta1_pair
        flags.append("beta1_only_fallback")

    half_2 = _slice_or_empty(cleaned, mature_start, groove_end)
    if len(half_2) < 70:
        flags.append(f"beta1_short({len(half_2)})")
    if not half_2:
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class="II",
            chain="beta",
            seq_len=len(cleaned),
            mature_start=mature_start,
            status="invalid_boundaries",
            anchor_type=anchor_type,
            anchor_cys1=(anchor_pair[0] if anchor_pair else None),
            anchor_cys2=(anchor_pair[1] if anchor_pair else None),
        )
    return GrooveResult(
        allele=allele,
        gene=gene,
        mhc_class="II",
        chain="beta",
        seq_len=len(cleaned),
        mature_start=mature_start,
        groove_seq=half_2,
        groove_half_1="",
        groove_half_2=half_2,
        groove_h1_len=0,
        groove_h2_len=len(half_2),
        status=status,
        anchor_type=anchor_type,
        anchor_cys1=(anchor_pair[0] if anchor_pair else None),
        anchor_cys2=(anchor_pair[1] if anchor_pair else None),
        secondary_cys1=(beta1_pair[0] if beta1_pair and beta2_pair is not None else None),
        secondary_cys2=(beta1_pair[1] if beta1_pair and beta2_pair is not None else None),
        flags=_flags_to_tuple(flags),
    )


def extract_groove(
    seq: str,
    *,
    mhc_class: str,
    chain: Optional[str] = None,
    allele: str = "",
    gene: str = "",
) -> GrooveResult:
    """Dispatch groove parsing by class and chain."""

    normalized_class = normalize_mhc_class(mhc_class)
    if normalized_class == "I":
        return parse_class_i(seq, allele=allele, gene=gene)
    if normalized_class != "II":
        raise ValueError(f"Unsupported MHC class for groove extraction: {mhc_class!r}")

    chain_token = str(chain or "").strip().lower()
    if chain_token in {"a", "alpha", "mhc_a"}:
        return parse_class_ii_alpha(seq, allele=allele, gene=gene)
    if chain_token in {"b", "beta", "mhc_b"}:
        return parse_class_ii_beta(seq, allele=allele, gene=gene)

    if chain_token:
        raise ValueError(f"Unsupported class-II chain token: {chain!r}")

    name_chain = _class_ii_chain_from_name(gene=gene, allele=allele)
    if name_chain == "alpha":
        return parse_class_ii_alpha(seq, allele=allele, gene=gene)
    if name_chain == "beta":
        return parse_class_ii_beta(seq, allele=allele, gene=gene)

    inferred_chain, alpha_result, beta_result = _infer_class_ii_chain_from_sequence(
        seq=seq,
        allele=allele,
        gene=gene,
    )
    if inferred_chain == "alpha":
        return alpha_result
    if inferred_chain == "beta":
        return beta_result
    return _class_ii_chain_inference_failure(
        seq=seq,
        allele=allele,
        gene=gene,
        alpha_result=alpha_result,
        beta_result=beta_result,
    )


def _fallback_truncate(seq: str, target_len: int) -> str:
    return _clean_seq(seq)[: max(0, int(target_len))]


def prepare_mhc_input(
    *,
    mhc_a: str,
    mhc_b: Optional[str] = None,
    mhc_class: str,
    allow_fallback_truncation: bool = True,
) -> PreparedMHCInput:
    """Convert raw MHC sequences into groove halves for runtime use.

    This helper intentionally operates on sequences only. Allele resolution and
    class-II default pairing are added later at the loader/inference layer.
    """

    normalized_class = normalize_mhc_class(mhc_class)
    if normalized_class not in {"I", "II"}:
        raise ValueError(f"Unsupported MHC class: {mhc_class!r}")

    if normalized_class == "I":
        parsed = parse_class_i(mhc_a)
        if parsed.ok:
            return PreparedMHCInput(
                mhc_class="I",
                groove_half_1=parsed.groove_half_1,
                groove_half_2=parsed.groove_half_2,
                groove_status_a=parsed.status,
                used_fallback=False,
            )
        if not allow_fallback_truncation:
            raise ValueError(f"Class-I groove extraction failed: {parsed.status}")
        cleaned = _clean_seq(mhc_a)
        return PreparedMHCInput(
            mhc_class="I",
            groove_half_1=_fallback_truncate(cleaned, DEFAULT_CLASS_I_GROOVE_HALF_1_LEN),
            groove_half_2=_fallback_truncate(
                cleaned[DEFAULT_CLASS_I_GROOVE_HALF_1_LEN :],
                DEFAULT_CLASS_I_GROOVE_HALF_2_LEN,
            ),
            groove_status_a=parsed.status,
            used_fallback=True,
        )

    parsed_a = parse_class_ii_alpha(mhc_a)
    parsed_b = parse_class_ii_beta(mhc_b or "")
    if parsed_a.ok and parsed_b.ok:
        return PreparedMHCInput(
            mhc_class="II",
            groove_half_1=parsed_a.groove_half_1,
            groove_half_2=parsed_b.groove_half_2,
            groove_status_a=parsed_a.status,
            groove_status_b=parsed_b.status,
            used_fallback=False,
        )
    if not allow_fallback_truncation:
        raise ValueError(
            "Class-II groove extraction failed: "
            f"alpha={parsed_a.status}, beta={parsed_b.status}"
        )
    return PreparedMHCInput(
        mhc_class="II",
        groove_half_1=(
            parsed_a.groove_half_1
            if parsed_a.ok
            else _fallback_truncate(mhc_a, DEFAULT_CLASS_II_ALPHA_GROOVE_LEN)
        ),
        groove_half_2=(
            parsed_b.groove_half_2
            if parsed_b.ok
            else _fallback_truncate(mhc_b or "", DEFAULT_CLASS_II_BETA_GROOVE_LEN)
        ),
        groove_status_a=parsed_a.status,
        groove_status_b=parsed_b.status,
        used_fallback=not (parsed_a.ok and parsed_b.ok),
    )


__all__ = [
    "GrooveResult",
    "PreparedMHCInput",
    "find_cys_pairs",
    "classify_cys_pair",
    "parse_class_i",
    "parse_class_ii_alpha",
    "parse_class_ii_beta",
    "extract_groove",
    "prepare_mhc_input",
]
