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
from typing import Optional

from .allele_resolver import normalize_mhc_class


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


def groove_record(
    seq: str, *, mhc_class: str, chain: Optional[str] = None, allele: str = "", gene: str = ""
) -> GrooveResult:
    """Parse one MHC chain into a `GrooveResult`, backed by mhcseqs.

    Kept as presto's own return type so callers and the index schema do not
    have to change, but every field is mhcseqs' answer rather than a second
    opinion computed here.
    """
    record = _mhcseqs_record(seq, mhc_class=mhc_class, chain=chain)
    cleaned = _clean_seq(seq)
    if record is None:
        return GrooveResult(
            allele=allele,
            gene=gene,
            mhc_class=mhc_class,
            chain=chain or "",
            seq_len=len(cleaned),
            status="unparsed",
        )
    half_1 = record.groove1 if mhc_class == "I" or (chain or "") == "alpha" else ""
    half_2 = record.groove2 if mhc_class == "I" or (chain or "") == "beta" else ""
    return GrooveResult(
        allele=allele or str(record.allele or ""),
        gene=gene or str(record.gene or ""),
        mhc_class=str(record.mhc_class or mhc_class),
        chain=str(record.chain or chain or ""),
        seq_len=int(record.seq_len or len(cleaned)),
        mature_start=int(record.mature_start or 0),
        groove_seq=str(record.groove_seq or ""),
        groove_half_1=half_1,
        groove_half_2=half_2,
        groove_h1_len=len(half_1),
        groove_h2_len=len(half_2),
        status=str(record.status or "unparsed"),
        anchor_type=str(record.anchor_type or ""),
        anchor_cys1=record.anchor_cys1,
        anchor_cys2=record.anchor_cys2,
        secondary_cys1=record.secondary_cys1,
        secondary_cys2=record.secondary_cys2,
        flags=tuple(record.flags or ()),
    )


def _mhcseqs_record(seq: str, *, mhc_class: str, chain: Optional[str] = None):
    """Parse one chain with mhcseqs, the single source of truth for grooves.

    presto used to carry its own cysteine-anchored parser. Measured against
    mhcseqs on 1,500 alleles from `data/mhc_index.csv`, the two agreed on only
    19.5%: class I was consistently off by one residue -- presto kept the
    leader's terminal alanine, so `AGSHSMRY...` where the mature chain is
    `GSHSMRY...` -- and class II differed by nine or more. For a model that
    reads groove positions as pocket-forming residues, that means position `k`
    did not mean what it was supposed to mean.

    mhcseqs also *declines* the 0.9% presto used to accept, and it is right to:
    they are TAP2 transporters, HLA-DM chaperones and null alleles. A protein
    with no peptide groove should yield no groove, not a truncation of itself.
    """
    import mhcseqs

    cleaned = _clean_seq(seq)
    if not cleaned:
        return None
    try:
        return mhcseqs.extract_groove(cleaned, mhc_class=mhc_class, chain=chain)
    except Exception:  # noqa: BLE001 - an unparseable chain is a real answer
        return None


def prepare_mhc_input(
    *,
    mhc_a: str,
    mhc_b: Optional[str] = None,
    mhc_class: str,
    allow_fallback_truncation: bool = True,
) -> PreparedMHCInput:
    """Convert raw MHC sequences into groove halves for runtime use.

    Operates on sequences only; allele resolution and class-II default pairing
    live at the loader/inference layer.

    `allow_fallback_truncation` is retained for callers but no longer truncates
    a sequence that failed to parse. Fabricating a groove from the first N
    residues of a protein that has none is how TAP2 and HLA-DM acquired one.
    When mhcseqs declines a chain the result is empty and `used_fallback` is
    True, so the caller can drop or flag the row; with the flag False, the
    failure raises instead.
    """
    normalized_class = normalize_mhc_class(mhc_class)
    if normalized_class not in {"I", "II"}:
        raise ValueError(f"Unsupported MHC class: {mhc_class!r}")

    if normalized_class == "I":
        record = _mhcseqs_record(mhc_a, mhc_class="I", chain="alpha")
        if record is not None and record.ok:
            return PreparedMHCInput(
                mhc_class="I",
                groove_half_1=record.groove1,
                groove_half_2=record.groove2,
                groove_status_a=str(record.status),
                used_fallback=False,
            )
        status = str(getattr(record, "status", "unparsed"))
        if not allow_fallback_truncation:
            raise ValueError(f"Class-I groove extraction failed: {status}")
        return PreparedMHCInput(
            mhc_class="I",
            groove_half_1="",
            groove_half_2="",
            groove_status_a=status,
            used_fallback=True,
        )

    # Class II: the alpha chain supplies the first groove half and the beta
    # chain the second, which is how mhcseqs reports them -- `groove1` on an
    # alpha record, `groove2` on a beta record.
    record_a = _mhcseqs_record(mhc_a, mhc_class="II", chain="alpha")
    record_b = _mhcseqs_record(mhc_b or "", mhc_class="II", chain="beta")
    ok_a = record_a is not None and record_a.ok
    ok_b = record_b is not None and record_b.ok
    status_a = str(getattr(record_a, "status", "unparsed"))
    status_b = str(getattr(record_b, "status", "unparsed"))
    if not (ok_a and ok_b) and not allow_fallback_truncation:
        raise ValueError(f"Class-II groove extraction failed: alpha={status_a}, beta={status_b}")
    return PreparedMHCInput(
        mhc_class="II",
        groove_half_1=record_a.groove1 if ok_a else "",
        groove_half_2=record_b.groove2 if ok_b else "",
        groove_status_a=status_a,
        groove_status_b=status_b,
        used_fallback=not (ok_a and ok_b),
    )


__all__ = [
    "GrooveResult",
    "PreparedMHCInput",
    "groove_record",
    "prepare_mhc_input",
]
