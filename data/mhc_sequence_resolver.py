"""Exact MHC sequence resolution with mhcseqs-first catalog lookup.

This keeps Presto's local CSV index as a compatibility fallback, but prefers
`mhcseqs` as the canonical sequence inventory whenever it is available.
"""

from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .groove import prepare_mhc_input
from .mhc_index import resolve_alleles


@dataclass(frozen=True)
class ExactMHCInput:
    """Canonical exact-allele MHC input record.

    `mhcseqs` is the preferred source because it exports the full sequence plus
    the canonical groove decomposition (`groove1`, `groove2`) for exact alleles.
    When the old local index is used as a fallback, only `sequence` is expected
    to be populated and downstream code may still need local groove parsing.
    """

    allele: str
    sequence: str
    groove1: str = ""
    groove2: str = ""
    mhc_class: str = ""
    chain: str = ""
    groove_status: str = ""
    source: str = "mhcseqs"


def _require_mhcseqs() -> Any:
    try:
        return importlib.import_module("mhcseqs")
    except ImportError as exc:
        raise RuntimeError(
            "mhcseqs is required for mhcseqs-first MHC sequence resolution."
        ) from exc


@lru_cache(maxsize=4)
def _load_mhcseqs_input_lookup(search_dir: Optional[str] = None) -> Dict[str, ExactMHCInput]:
    mhcseqs = _require_mhcseqs()
    csv_path = None
    if search_dir:
        candidate = Path(search_dir) / "mhc-full-seqs.csv"
        if candidate.exists():
            csv_path = str(candidate)
    load_sequences_dict = getattr(mhcseqs, "load_sequences_dict", None)
    if load_sequences_dict is None:
        raise RuntimeError("mhcseqs does not expose load_sequences_dict()")
    rows = load_sequences_dict(path=csv_path)
    lookup: Dict[str, ExactMHCInput] = {}
    for row in rows:
        sequence = str(row.get("sequence") or "").strip().upper()
        if not sequence:
            continue
        record = ExactMHCInput(
            allele=(
                str(row.get("two_field_allele") or "").strip()
                or str(row.get("representative_allele") or "").strip()
            ),
            sequence=sequence,
            groove1=str(row.get("groove1") or "").strip().upper(),
            groove2=str(row.get("groove2") or "").strip().upper(),
            mhc_class=str(row.get("mhc_class") or row.get("class") or "").strip().upper(),
            chain=str(row.get("chain") or "").strip().lower(),
            groove_status=str(row.get("groove_status") or "").strip(),
            source="mhcseqs",
        )
        for token in (
            str(row.get("two_field_allele") or "").strip(),
            str(row.get("representative_allele") or "").strip(),
        ):
            if not token:
                continue
            lookup[token] = record
            lookup[token.upper()] = record
    return lookup


@lru_cache(maxsize=4)
def _load_mhcseqs_sequence_lookup(search_dir: Optional[str] = None) -> Dict[str, str]:
    lookup = _load_mhcseqs_input_lookup(search_dir=search_dir)
    return {
        key: value.sequence
        for key, value in lookup.items()
        if value.sequence
    }


def _resolve_input_via_mhcseqs(
    allele: str,
    *,
    search_dir: Optional[str] = None,
) -> Optional[ExactMHCInput]:
    query = str(allele or "").strip()
    if not query:
        return None
    lookup = _load_mhcseqs_input_lookup(search_dir=search_dir)
    mhcseqs = _require_mhcseqs()
    candidates = [query, query.upper()]
    try:
        normalized = mhcseqs.normalize_allele_name(query)
    except Exception:
        normalized = ""
    if normalized:
        candidates.extend([normalized, normalized.upper()])
    for token in candidates:
        record = lookup.get(token)
        if record:
            return record
    return None


def lookup_exact_mhc_input(
    allele: str,
    *,
    mhcseqs_search_dir: Optional[str] = None,
) -> Optional[ExactMHCInput]:
    """Return the canonical exact-allele record from `mhcseqs`, if available."""
    if mhcseqs_search_dir is None:
        env_search_dir = str(os.environ.get("PRESTO_MHCSEQS_SEARCH_DIR", "")).strip()
        if env_search_dir:
            mhcseqs_search_dir = env_search_dir
        elif Path("/opt/mhcseqs/mhc-full-seqs.csv").exists():
            mhcseqs_search_dir = "/opt/mhcseqs"
    try:
        return _resolve_input_via_mhcseqs(allele, search_dir=mhcseqs_search_dir)
    except (AttributeError, FileNotFoundError, RuntimeError):
        return None


def find_matching_allele_sequence(
    allele_sequences: Optional[Mapping[str, str]],
    allele: str,
) -> Optional[str]:
    """Best-effort lookup for a full MHC sequence by allele-ish key.

    This is a compatibility fallback for older probe/eval utilities that still
    carry an `allele -> sequence` mapping. Exact alleles should prefer
    `lookup_exact_mhc_input()` and consume `mhcseqs` groove exports directly.
    """
    if not allele_sequences:
        return None
    query = str(allele or "").strip()
    if not query:
        return None

    direct_candidates = [query, query.upper()]
    if query.upper().startswith("HLA-"):
        short = query[4:]
        direct_candidates.extend([short, short.upper()])

    for token in direct_candidates:
        value = allele_sequences.get(token)
        if value:
            return str(value).strip().upper()

    query_upper = query.upper()
    for key, value in allele_sequences.items():
        key_clean = str(key or "").strip()
        if not key_clean or not value:
            continue
        key_upper = key_clean.upper()
        if query_upper in key_upper or key_upper in query_upper:
            return str(value).strip().upper()
    return None


def resolve_class_i_groove_halves(
    *,
    allele: str,
    allele_sequences: Optional[Mapping[str, str]] = None,
    mhcseqs_search_dir: Optional[str] = None,
) -> Optional[Tuple[str, str]]:
    """Resolve class-I `(groove1, groove2)` with `mhcseqs`-first ownership.

    Exact alleles consume `mhcseqs` groove exports directly when available.
    Older full-sequence maps remain as a compatibility fallback.
    """
    exact = lookup_exact_mhc_input(
        allele,
        mhcseqs_search_dir=mhcseqs_search_dir,
    )
    if exact is not None and (
        str(exact.groove1 or "").strip() or str(exact.groove2 or "").strip()
    ):
        return (
            str(exact.groove1 or "").strip().upper(),
            str(exact.groove2 or "").strip().upper(),
        )

    mhc_seq = find_matching_allele_sequence(allele_sequences, allele)
    if not mhc_seq:
        return None
    prepared = prepare_mhc_input(mhc_a=mhc_seq, mhc_class="I")
    return prepared.groove_half_1, prepared.groove_half_2


def _sequence_map_from_exact_inputs(
    mapping: Mapping[str, ExactMHCInput],
) -> Dict[str, str]:
    return {
        allele: record.sequence
        for allele, record in mapping.items()
        if str(record.sequence or "").strip()
    }


def resolve_exact_mhc_inputs(
    alleles: Sequence[str],
    *,
    index_csv: Optional[str] = None,
    prefer_mhcseqs: bool = True,
    mhcseqs_search_dir: Optional[str] = None,
) -> Tuple[Dict[str, ExactMHCInput], Dict[str, int]]:
    """Resolve exact allele tokens to sequences.

    Input alleles are treated as exact candidate tokens. Ambiguous serotype or
    haplotype expansion should already have happened upstream.
    """
    unique_alleles = sorted({str(a or "").strip() for a in alleles if str(a or "").strip()})
    if not unique_alleles:
        return {}, {
            "total": 0,
            "resolved": 0,
            "resolved_mhcseqs": 0,
            "resolved_index": 0,
            "missing": 0,
        }

    mapping: Dict[str, ExactMHCInput] = {}
    resolved_mhcseqs = 0
    resolved_index = 0
    if mhcseqs_search_dir is None:
        env_search_dir = str(os.environ.get("PRESTO_MHCSEQS_SEARCH_DIR", "")).strip()
        if env_search_dir:
            mhcseqs_search_dir = env_search_dir
        elif Path("/opt/mhcseqs/mhc-full-seqs.csv").exists():
            mhcseqs_search_dir = "/opt/mhcseqs"

    unresolved = list(unique_alleles)
    if prefer_mhcseqs:
        still_unresolved = []
        for allele in unresolved:
            record = lookup_exact_mhc_input(
                allele,
                mhcseqs_search_dir=mhcseqs_search_dir,
            )
            if record:
                mapping[allele] = record
                resolved_mhcseqs += 1
            else:
                still_unresolved.append(allele)
        unresolved = still_unresolved

    if unresolved and index_csv:
        results = resolve_alleles(
            index_csv=index_csv,
            alleles=unresolved,
            include_sequence=True,
        )
        for row in results:
            if row.get("found") and row.get("sequence"):
                mapping[str(row["input"])] = ExactMHCInput(
                    allele=str(row["input"]),
                    sequence=str(row["sequence"]).strip().upper(),
                    source="index",
                )
                resolved_index += 1

    resolved = len(mapping)
    return mapping, {
        "total": len(unique_alleles),
        "resolved": resolved,
        "resolved_mhcseqs": resolved_mhcseqs,
        "resolved_index": resolved_index,
        "missing": len(unique_alleles) - resolved,
    }


def resolve_exact_mhc_sequences(
    alleles: Sequence[str],
    *,
    index_csv: Optional[str] = None,
    prefer_mhcseqs: bool = True,
    mhcseqs_search_dir: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Resolve exact allele tokens to sequences.

    This compatibility API preserves older callers that only need the full
    sequence mapping. New code that wants canonical `mhcseqs` groove exports
    should call `resolve_exact_mhc_inputs()` instead.
    """
    mapping, stats = resolve_exact_mhc_inputs(
        alleles=alleles,
        index_csv=index_csv,
        prefer_mhcseqs=prefer_mhcseqs,
        mhcseqs_search_dir=mhcseqs_search_dir,
    )
    return _sequence_map_from_exact_inputs(mapping), stats
