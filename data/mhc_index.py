"""MHC index build + allele resolution utilities.

Builds a CSV index from IMGT/IPD-MHC FASTA files using mhcgnomes for
normalization, then resolves allele names against that index.
"""

from __future__ import annotations

import csv
import gzip
import io
import zipfile
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .allele_resolver import (
    infer_gene,
    infer_mhc_class,
    infer_species,
    infer_species_identity,
    normalize_mhc_class,
    normalize_allele_name,
    normalize_species_label,
    parse_allele_name,
    require_mhcgnomes,
)
from .groove import (
    parse_class_i,
    parse_class_ii_alpha,
    parse_class_ii_beta,
)


INDEX_FIELDS = [
    "allele_raw",
    "normalized",
    "gene",
    "mhc_class",
    "species",
    "source",
    "seq_len",
    "sequence",
]

AUGMENTED_INDEX_FIELDS = INDEX_FIELDS + [
    "mature_start",
    "groove_half_1",
    "groove_half_2",
    "groove_status",
    "is_null",
    "is_questionable",
    "is_pseudogene",
    "is_functional",
]

FUNCTIONAL_GROOVE_STATUSES = {
    "ok",
    "alpha3_fallback",
    "beta1_only_fallback",
    "fragment_fallback",
}

_NUCLEOTIDE_LIKE_CHARS = set("ACGTUNWSMKRYBDHV")
_PROTEIN_FASTA_HINTS = ("prot", "protein", ".faa", "_aa", "-aa", "_pep", "-pep")
_NUCLEOTIDE_FASTA_HINTS = (
    "nuc",
    "nucleotide",
    ".fna",
    ".ffn",
    "_dna",
    "-dna",
    "_rna",
    "-rna",
    "cdna",
    "cds",
)
MIN_MHC_SEQUENCE_LEN = 70  # allow groove-bearing fragments, reject trivial truncations


@dataclass
class MHCIndexRecord:
    allele_raw: str
    normalized: str
    gene: str
    mhc_class: str
    species: str
    source: str
    seq_len: int
    sequence: str
    mature_start: Optional[int] = None
    groove_half_1: str = ""
    groove_half_2: str = ""
    groove_status: str = ""
    is_null: bool = False
    is_questionable: bool = False
    is_pseudogene: bool = False
    is_functional: Optional[bool] = None
    representative_allele: str = ""
    representative_policy: str = ""

    def to_row(self) -> Dict[str, str]:
        return {
            "allele_raw": self.allele_raw,
            "normalized": self.normalized,
            "gene": self.gene,
            "mhc_class": self.mhc_class,
            "species": self.species,
            "source": self.source,
            "seq_len": str(self.seq_len),
            "sequence": self.sequence,
        }

    def to_augmented_row(self) -> Dict[str, str]:
        row = self.to_row()
        row.update(
            {
                "mature_start": "" if self.mature_start is None else str(int(self.mature_start)),
                "groove_half_1": self.groove_half_1,
                "groove_half_2": self.groove_half_2,
                "groove_status": self.groove_status,
                "is_null": str(bool(self.is_null)),
                "is_questionable": str(bool(self.is_questionable)),
                "is_pseudogene": str(bool(self.is_pseudogene)),
                "is_functional": (
                    "" if self.is_functional is None else str(bool(self.is_functional))
                ),
            }
        )
        return row

    @staticmethod
    def from_row(row: Dict[str, str]) -> "MHCIndexRecord":
        def _parse_bool(value: object) -> bool:
            token = str(value or "").strip().lower()
            return token in {"1", "true", "t", "yes", "y"}

        seq = (row.get("sequence") or "").strip()
        seq_len = row.get("seq_len")
        if seq_len and seq_len.isdigit():
            length = int(seq_len)
        else:
            length = len(seq)
        mature_start_raw = str(row.get("mature_start") or "").strip()
        mature_start = int(mature_start_raw) if mature_start_raw.isdigit() else None
        is_functional_raw = str(row.get("is_functional") or "").strip()
        return MHCIndexRecord(
            allele_raw=(row.get("allele_raw") or row.get("allele") or "").strip(),
            normalized=(row.get("normalized") or "").strip(),
            gene=(row.get("gene") or "").strip(),
            mhc_class=(row.get("mhc_class") or "").strip(),
            species=(row.get("species") or "").strip(),
            source=(row.get("source") or "").strip(),
            seq_len=length,
            sequence=seq,
            mature_start=mature_start,
            groove_half_1=(row.get("groove_half_1") or "").strip(),
            groove_half_2=(row.get("groove_half_2") or "").strip(),
            groove_status=(row.get("groove_status") or "").strip(),
            is_null=_parse_bool(row.get("is_null")),
            is_questionable=_parse_bool(row.get("is_questionable")),
            is_pseudogene=_parse_bool(row.get("is_pseudogene")),
            is_functional=(
                _parse_bool(is_functional_raw) if is_functional_raw else None
            ),
        )


class MHCIndexError(RuntimeError):
    """Raised when MHC index operations fail."""


def _require_mhcgnomes():
    try:
        return require_mhcgnomes()
    except Exception as exc:
        raise MHCIndexError(
            "mhcgnomes is required for allele name normalization. "
            "Install it with `pip install mhcgnomes`."
        ) from exc


def _open_text(path: Path) -> Iterator[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            yield from f
        return
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            fasta_names = [
                name
                for name in zf.namelist()
                if name.endswith((".fa", ".fasta", ".faa", ".fa.gz", ".fasta.gz", ".faa.gz"))
            ]
            if not fasta_names:
                raise MHCIndexError(f"No FASTA files found in {path}")
            name = fasta_names[0]
            with zf.open(name) as fh:
                if name.endswith(".gz"):
                    with gzip.open(fh, "rt", encoding="utf-8") as gz:
                        yield from gz
                else:
                    yield from io.TextIOWrapper(fh, encoding="utf-8")
        return
    with open(path, "r", encoding="utf-8") as f:
        yield from f


def _iter_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    header = None
    seq_parts: List[str] = []
    for line in _open_text(path):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_parts)
            header = line[1:].strip()
            seq_parts = []
        else:
            seq_parts.append(line)
    if header is not None:
        yield header, "".join(seq_parts)


def _looks_like_nucleotide_sequence(sequence: str) -> bool:
    """Return True when a FASTA entry is overwhelmingly nucleotide-like.

    MHC protein chains should contain a broad amino-acid alphabet. Pure
    nucleotide FASTAs (`A/C/G/T/U/...`) can otherwise slip through because
    those letters are also valid amino-acid tokens.
    """
    seq = re.sub(r"\s+", "", str(sequence or "").strip().upper())
    if not seq:
        return False
    chars = {ch for ch in seq if ch.isalpha()}
    nucleotide_chars = chars & set("ACGTU")
    return bool(chars) and chars <= _NUCLEOTIDE_LIKE_CHARS and len(nucleotide_chars) >= 3


def _fasta_path_priority(path: Path) -> Tuple[int, str]:
    """Sort protein-like FASTA files ahead of nucleotide dumps."""
    name = path.name.lower()
    if any(token in name for token in _PROTEIN_FASTA_HINTS):
        return (0, name)
    if any(token in name for token in _NUCLEOTIDE_FASTA_HINTS):
        return (2, name)
    return (1, name)


def _candidate_tokens(header: str) -> List[str]:
    cleaned = header.replace("|", " ").replace(";", " ")
    tokens = [t.strip() for t in cleaned.split() if t.strip()]
    if not tokens:
        return []
    scored = []
    for tok in tokens:
        score = 0
        if "*" in tok:
            score += 3
        if tok.upper().startswith("HLA-"):
            score += 2
        if tok.upper().startswith("H-2"):
            score += 2
        if tok.upper().startswith("MAMU"):
            score += 2
        if ":" in tok:
            score += 1
        scored.append((score, tok))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [tok for _, tok in scored]


def _normalize_allele_token(token: str) -> str:
    """Normalize non-canonical allele tokens before indexing/aliasing."""
    value = token.strip().strip(",;")
    if not value:
        return value

    # Normalize common compact human forms (e.g. HLA-A2 -> HLA-A*02,
    # A0201 -> HLA-A*02:01) when token appears allele-like.
    compact_candidate = value.upper().replace("_", "-")
    if (
        " " not in compact_candidate
        and re.match(r"^(HLA[-_])?[A-Z][A-Z0-9]{0,6}[*]?[0-9A-Z:]+$", compact_candidate)
    ):
        try:
            value = normalize_allele_name(value)
        except Exception:
            pass

    upper = value.upper()

    # Canonicalize common murine prefixes.
    if upper.startswith("H-2-"):
        value = "H2-" + value[4:]
    elif upper.startswith("H-2"):
        value = "H2-" + value[3:]
    elif upper.startswith("H2") and not upper.startswith("H2-"):
        value = "H2-" + value[2:]

    # Canonicalize compact murine forms: H2-Kd -> H2-K*d.
    if value.upper().startswith("H2-") and "*" not in value:
        body = value[3:]
        if len(body) >= 2 and body[0].isalpha():
            value = f"H2-{body[0].upper()}*{body[1:]}"

    return value


def _normalize_with_mhcgnomes(token: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    parsed = parse_allele_name(token)
    if parsed is None:
        raise MHCIndexError(f"mhcgnomes failed to parse allele token: {token!r}")
    normalized = parsed.to_string()
    gene = getattr(parsed.gene, "name", None) if getattr(parsed, "gene", None) else None
    species = getattr(parsed.species, "name", None) if getattr(parsed, "species", None) else None
    mhc_class = normalize_mhc_class(getattr(parsed, "mhc_class", None))
    return normalized, gene, mhc_class, species


def _resolve_header_allele(
    header: str,
) -> Tuple[str, Optional[str], Optional[str], Optional[str], str]:
    last_error = None
    for token in _candidate_tokens(header):
        try:
            normalized, gene, mhc_class, species = _normalize_with_mhcgnomes(
                _normalize_allele_token(token)
            )
            return normalized, gene, mhc_class, species, token
        except Exception as exc:
            last_error = exc
            continue
    raise MHCIndexError(
        f"Failed to parse allele from FASTA header: '{header}'."
    ) from last_error


def _iter_fasta_paths(imgt_fasta: Optional[str], ipd_mhc_dir: Optional[str]) -> List[Tuple[Path, str]]:
    paths: List[Tuple[Path, str]] = []
    if imgt_fasta:
        path = Path(imgt_fasta)
        if not path.exists():
            raise MHCIndexError(f"IMGT FASTA not found: {path}")
        paths.append((path, "imgt"))
    if ipd_mhc_dir:
        root = Path(ipd_mhc_dir)
        if not root.exists():
            raise MHCIndexError(f"IPD-MHC path not found: {root}")
        if root.is_file():
            paths.append((root, "ipd_mhc"))
        else:
            fasta_paths = sorted(
                (
                    p
                    for p in root.rglob("*")
                    if p.suffix in {".fa", ".fasta", ".faa", ".gz", ".zip"}
                ),
                key=_fasta_path_priority,
            )
            if not fasta_paths:
                raise MHCIndexError(f"No FASTA files found under {root}")
            paths.extend((p, "ipd_mhc") for p in fasta_paths)
    if not paths:
        raise MHCIndexError("Provide at least one of --imgt-fasta or --ipd-mhc-dir")
    return paths


def build_mhc_index(
    imgt_fasta: Optional[str],
    ipd_mhc_dir: Optional[str],
    out_csv: str,
    out_fasta: Optional[str] = None,
) -> Dict[str, int]:
    """Build an MHC index CSV (and optional FASTA) from IMGT/IPD-MHC FASTA files.

    Returns stats dictionary.
    """
    _require_mhcgnomes()
    records: Dict[str, MHCIndexRecord] = {}
    stats = {
        "total": 0,
        "parsed": 0,
        "skipped": 0,
        "skipped_nucleotide": 0,
        "skipped_short": 0,
        "duplicates": 0,
        "replaced": 0,
    }

    for path, source in _iter_fasta_paths(imgt_fasta, ipd_mhc_dir):
        for header, seq in _iter_fasta(path):
            stats["total"] += 1
            if _looks_like_nucleotide_sequence(seq):
                stats["skipped_nucleotide"] += 1
                continue
            if len(seq) < MIN_MHC_SEQUENCE_LEN:
                stats["skipped_short"] += 1
                continue
            try:
                normalized, gene, mhc_class, species, allele_token = _resolve_header_allele(header)
            except MHCIndexError:
                stats["skipped"] += 1
                continue
            if not gene:
                gene = infer_gene(normalized)
            if not mhc_class:
                mhc_class = infer_mhc_class(normalized)
            mhc_class = normalize_mhc_class(mhc_class, default=infer_mhc_class(normalized))
            if not species:
                species = infer_species_identity(normalized) or infer_species(normalized) or ""

            record = MHCIndexRecord(
                allele_raw=allele_token,
                normalized=normalized,
                gene=gene,
                mhc_class=mhc_class,
                species=species,
                source=source,
                seq_len=len(seq),
                sequence=seq,
            )

            if normalized in records:
                existing = records[normalized]
                if _looks_like_nucleotide_sequence(existing.sequence):
                    records[normalized] = record
                    stats["replaced"] += 1
                elif existing.source != "imgt" and source == "imgt":
                    records[normalized] = record
                    stats["replaced"] += 1
                else:
                    stats["duplicates"] += 1
                continue

            records[normalized] = record
            stats["parsed"] += 1

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        for key in sorted(records.keys()):
            writer.writerow(records[key].to_row())

    if out_fasta:
        fasta_path = Path(out_fasta)
        fasta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fasta_path, "w", encoding="utf-8") as f:
            for key in sorted(records.keys()):
                rec = records[key]
                f.write(f">{rec.normalized} source={rec.source} gene={rec.gene} species={rec.species}\n")
                f.write(f"{rec.sequence}\n")

    return stats


# ---------------------------------------------------------------------------
# Non-classical Class I gene patterns (human and mouse)
# ---------------------------------------------------------------------------
_NONCLASSICAL_CLASS_I_GENES = {
    # Human non-classical
    "HLA-E", "HLA-F", "HLA-G",
    # Mouse non-classical (Qa, Tla family)
    "H2-Q", "H2-T", "H2-M",
}

_CLASSICAL_CLASS_I_GENES = {
    # Human classical
    "HLA-A", "HLA-B", "HLA-C",
    # Mouse classical
    "H2-K", "H2-D", "H2-L",
}

_CLASS_II_ALPHA_PATTERNS = ("DRA", "DQA", "DPA", "AA", "EA")
_CLASS_II_BETA_PATTERNS = ("DRB", "DQB", "DPB", "AB", "EB")

# B2M is typically short (~119 residues)
_B2M_MAX_SEQ_LEN = 150


def infer_fine_chain_type(
    gene: Optional[str],
    mhc_class: Optional[str],
    sequence_len: Optional[int] = None,
) -> str:
    """Infer fine-grained MHC chain type (5 classes) from gene and class info.

    Returns one of: MHC_I, MHC_IIa, MHC_IIb, B2M, unknown.
    """
    g = (gene or "").strip().upper()
    mc = normalize_mhc_class(mhc_class or "", default="")

    # Check for B2M by gene name or short invariant sequence length
    if "B2M" in g or "BETA-2" in g or "BETA2" in g:
        return "B2M"
    if sequence_len is not None and sequence_len <= _B2M_MAX_SEQ_LEN and mc in ("I", ""):
        # Short sequences in class I context are likely B2M
        return "B2M"

    # Class II chains
    if mc == "II" or any(tag in g for tag in _CLASS_II_ALPHA_PATTERNS + _CLASS_II_BETA_PATTERNS):
        if any(tag in g for tag in _CLASS_II_BETA_PATTERNS):
            return "MHC_IIb"
        if any(tag in g for tag in _CLASS_II_ALPHA_PATTERNS):
            return "MHC_IIa"
        # Ambiguous class II — default to alpha if in α slot
        return "MHC_IIa"

    # Class I: both classical and non-classical return MHC_I
    if mc == "I" or g:
        for nc_gene in _NONCLASSICAL_CLASS_I_GENES:
            if g.startswith(nc_gene.upper()):
                return "MHC_I"
        for cl_gene in _CLASSICAL_CLASS_I_GENES:
            if g.startswith(cl_gene.upper()):
                return "MHC_I"
        # Fallback: if class I but gene not recognized
        if mc == "I":
            return "MHC_I"

    return "unknown"


def load_mhc_index(index_csv: str) -> Dict[str, MHCIndexRecord]:
    path = Path(index_csv)
    if not path.exists():
        raise MHCIndexError(f"Index CSV not found: {path}")
    records: Dict[str, MHCIndexRecord] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = MHCIndexRecord.from_row(row)
            if not rec.normalized:
                continue
            records[rec.normalized] = rec
    if not records:
        raise MHCIndexError(f"No records loaded from index: {path}")
    return records


def _is_class_ii_alpha_gene(gene: str) -> bool:
    token = str(gene or "").strip().upper()
    return token.startswith(("DRA", "DQA", "DPA", "DMA", "DOA")) or token.endswith("A")


def _allele_suffix_flags(allele: str) -> Dict[str, bool]:
    token = str(allele or "").strip()
    suffix_match = re.search(r"([A-Za-z]+)$", token)
    suffix = suffix_match.group(1).upper() if suffix_match else ""
    return {
        "is_null": suffix == "N",
        "is_questionable": suffix == "Q",
        "is_pseudogene": suffix == "PS",
    }


def augment_mhc_index(index_csv: str, output_csv: str) -> Dict[str, object]:
    """Add groove-centric parsed fields to an existing MHC index CSV."""

    records = load_mhc_index(index_csv)
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)

    status_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    functional_true = 0
    functional_false = 0

    augmented_rows: List[Dict[str, str]] = []
    for rec in records.values():
        normalized_class = normalize_mhc_class(rec.mhc_class, default=rec.mhc_class)
        suffix_flags = _allele_suffix_flags(rec.normalized or rec.allele_raw)
        if normalized_class == "I":
            parsed = parse_class_i(rec.sequence, allele=rec.normalized, gene=rec.gene)
        elif normalized_class == "II":
            if _is_class_ii_alpha_gene(rec.gene):
                parsed = parse_class_ii_alpha(rec.sequence, allele=rec.normalized, gene=rec.gene)
            else:
                parsed = parse_class_ii_beta(rec.sequence, allele=rec.normalized, gene=rec.gene)
        else:
            raise MHCIndexError(
                f"Unsupported MHC class for groove augmentation: {rec.mhc_class!r}"
            )

        is_functional = (
            parsed.status in FUNCTIONAL_GROOVE_STATUSES
            and not suffix_flags["is_null"]
            and not suffix_flags["is_pseudogene"]
        )
        status_counts[parsed.status] += 1
        class_counts[normalized_class or ""] += 1
        if is_functional:
            functional_true += 1
        else:
            functional_false += 1

        augmented = MHCIndexRecord(
            allele_raw=rec.allele_raw,
            normalized=rec.normalized,
            gene=rec.gene,
            mhc_class=rec.mhc_class,
            species=rec.species,
            source=rec.source,
            seq_len=rec.seq_len,
            sequence=rec.sequence,
            mature_start=parsed.mature_start if parsed.status else None,
            groove_half_1=parsed.groove_half_1,
            groove_half_2=parsed.groove_half_2,
            groove_status=parsed.status,
            is_null=suffix_flags["is_null"],
            is_questionable=suffix_flags["is_questionable"],
            is_pseudogene=suffix_flags["is_pseudogene"],
            is_functional=is_functional,
        )
        augmented_rows.append(augmented.to_augmented_row())

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUGMENTED_INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(augmented_rows)

    return {
        "total_records": len(augmented_rows),
        "by_mhc_class": {key: class_counts[key] for key in sorted(class_counts)},
        "by_groove_status": {key: status_counts[key] for key in sorted(status_counts)},
        "functional_true": functional_true,
        "functional_false": functional_false,
        "output_csv": str(path),
    }


def _build_alias_map(records: Dict[str, MHCIndexRecord]) -> Dict[str, str]:
    aliases, _, _ = _build_alias_resolution(records)
    return aliases


def _emit_alias(alias_map: Dict[str, str], alias: str, normalized: str) -> None:
    token = alias.strip()
    if token:
        for variant in {token, token.upper(), token.lower()}:
            alias_map.setdefault(variant, normalized)


def _emit_alias_override(alias_map: Dict[str, str], alias: str, normalized: str) -> None:
    token = alias.strip()
    if token:
        for variant in {token, token.upper(), token.lower()}:
            alias_map[variant] = normalized


def _alias_variants(normalized: str) -> List[str]:
    variants = [normalized]
    token = _normalize_allele_token(normalized)
    variants.append(token)
    upper = token.upper()
    if upper.startswith("H2-"):
        suffix = token[3:]
        suffix_no_star = suffix.replace("*", "")
        variants.extend(
            [
                f"H-2-{suffix}",
                f"H-2{suffix}",
                f"H-2-{suffix_no_star}",
                f"H-2{suffix_no_star}",
                f"H2-{suffix_no_star}",
                f"H2{suffix}",
                f"H2{suffix_no_star}",
            ]
        )
    return variants


def _unique_sequence_records(records: List[MHCIndexRecord]) -> List[MHCIndexRecord]:
    chosen: Dict[str, MHCIndexRecord] = {}
    for rec in records:
        seq = str(rec.sequence or "")
        current = chosen.get(seq)
        if current is None:
            chosen[seq] = rec
            continue
        if (len(rec.normalized), rec.normalized) > (len(current.normalized), current.normalized):
            chosen[seq] = rec
    return list(chosen.values())


def _parse_record_sequence(
    record: MHCIndexRecord,
    *,
    sequence: Optional[str] = None,
    allele: Optional[str] = None,
):
    seq = str(sequence if sequence is not None else record.sequence or "").strip().upper()
    allele_name = str(allele or record.normalized or record.allele_raw or "").strip()
    normalized_class = normalize_mhc_class(record.mhc_class, default=record.mhc_class)
    if normalized_class == "I":
        return parse_class_i(seq, allele=allele_name, gene=record.gene)
    if normalized_class == "II":
        if _is_class_ii_alpha_gene(record.gene):
            return parse_class_ii_alpha(seq, allele=allele_name, gene=record.gene)
        return parse_class_ii_beta(seq, allele=allele_name, gene=record.gene)
    raise MHCIndexError(
        f"Unsupported MHC class for record parsing: {record.mhc_class!r} ({record.normalized!r})"
    )


def _full_length_parse_ok(record: MHCIndexRecord, parsed) -> bool:
    normalized_class = normalize_mhc_class(record.mhc_class, default=record.mhc_class)
    if normalized_class == "I":
        return parsed.status in {"ok", "alpha3_fallback"}
    if normalized_class == "II":
        if _is_class_ii_alpha_gene(record.gene):
            return parsed.status == "ok"
        return parsed.status == "ok"
    return False


def _record_groove_signature(record: MHCIndexRecord) -> Optional[Tuple[str, str]]:
    half_1 = str(record.groove_half_1 or "").strip().upper()
    half_2 = str(record.groove_half_2 or "").strip().upper()
    if half_1 or half_2:
        return (half_1, half_2)
    try:
        parsed = _parse_record_sequence(record)
    except Exception:
        return None
    if not parsed.ok:
        return None
    return (
        str(parsed.groove_half_1 or "").strip().upper(),
        str(parsed.groove_half_2 or "").strip().upper(),
    )


def _alias_record_from_representative(
    *,
    alias: str,
    representative: MHCIndexRecord,
    policy: str,
    sequence: Optional[str] = None,
) -> Optional[MHCIndexRecord]:
    seq = str(sequence if sequence is not None else representative.sequence or "").strip().upper()
    if not seq:
        return None

    groove_half_1 = str(representative.groove_half_1 or "").strip().upper()
    groove_half_2 = str(representative.groove_half_2 or "").strip().upper()
    groove_status = str(representative.groove_status or "").strip()
    mature_start = representative.mature_start
    is_functional = representative.is_functional

    if not groove_half_1 and not groove_half_2:
        parsed = _parse_record_sequence(representative, sequence=seq, allele=alias)
        groove_half_1 = parsed.groove_half_1
        groove_half_2 = parsed.groove_half_2
        groove_status = parsed.status
        mature_start = parsed.mature_start if parsed.status else None
        if is_functional is None:
            is_functional = parsed.status in FUNCTIONAL_GROOVE_STATUSES

    return MHCIndexRecord(
        allele_raw=alias,
        normalized=alias,
        gene=representative.gene,
        mhc_class=representative.mhc_class,
        species=representative.species,
        source=representative.source,
        seq_len=len(seq),
        sequence=seq,
        mature_start=mature_start,
        groove_half_1=groove_half_1,
        groove_half_2=groove_half_2,
        groove_status=groove_status,
        is_null=representative.is_null,
        is_questionable=representative.is_questionable,
        is_pseudogene=representative.is_pseudogene,
        is_functional=is_functional,
        representative_allele=representative.normalized,
        representative_policy=policy,
    )


def _merge_sequences_with_exact_overlap(
    left: str,
    right: str,
    *,
    min_overlap: int,
) -> Optional[str]:
    left_seq = str(left or "").strip().upper()
    right_seq = str(right or "").strip().upper()
    if not left_seq or not right_seq:
        return None
    if right_seq in left_seq:
        return left_seq
    if left_seq in right_seq:
        return right_seq

    candidates: List[Tuple[int, int, str]] = []
    min_offset = -(len(right_seq) - min_overlap)
    max_offset = len(left_seq) - min_overlap
    for offset in range(min_offset, max_offset + 1):
        left_start = max(0, offset)
        right_start = max(0, -offset)
        overlap_len = min(
            len(left_seq) - left_start,
            len(right_seq) - right_start,
        )
        if overlap_len < min_overlap:
            continue
        if left_seq[left_start : left_start + overlap_len] != right_seq[
            right_start : right_start + overlap_len
        ]:
            continue

        merged_start = min(0, offset)
        merged_end = max(len(left_seq), offset + len(right_seq))
        merged_chars: List[str] = []
        conflict = False
        for pos in range(merged_start, merged_end):
            left_char = left_seq[pos] if 0 <= pos < len(left_seq) else ""
            right_pos = pos - offset
            right_char = right_seq[right_pos] if 0 <= right_pos < len(right_seq) else ""
            if left_char and right_char and left_char != right_char:
                conflict = True
                break
            merged_chars.append(left_char or right_char)
        if conflict:
            continue
        candidates.append((overlap_len, len(merged_chars), "".join(merged_chars)))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    best_overlap, best_len, best_seq = candidates[0]
    tied = [
        seq
        for overlap, merged_len, seq in candidates
        if overlap == best_overlap and merged_len == best_len
    ]
    if len(set(tied)) != 1:
        return None
    return best_seq


def _assemble_overlap_representative(
    group_key: str,
    group_records: List[MHCIndexRecord],
    *,
    min_overlap: int = 100,
) -> Optional[MHCIndexRecord]:
    unique_records = sorted(
        _unique_sequence_records(group_records),
        key=lambda rec: (-len(rec.sequence), rec.normalized),
    )
    if len(unique_records) <= 1:
        return None

    anchor = unique_records[0]
    consensus = str(anchor.sequence or "").strip().upper()
    if not consensus:
        return None

    for rec in unique_records[1:]:
        seq = str(rec.sequence or "").strip().upper()
        if not seq:
            return None
        merged = _merge_sequences_with_exact_overlap(
            consensus,
            seq,
            min_overlap=min_overlap,
        )
        if merged is None:
            return None
        consensus = merged

    parsed = _parse_record_sequence(anchor, sequence=consensus, allele=group_key)
    if not _full_length_parse_ok(anchor, parsed):
        return None

    return MHCIndexRecord(
        allele_raw=group_key,
        normalized=group_key,
        gene=anchor.gene,
        mhc_class=anchor.mhc_class,
        species=anchor.species,
        source=anchor.source,
        seq_len=len(consensus),
        sequence=consensus,
        mature_start=parsed.mature_start if parsed.status else None,
        groove_half_1=parsed.groove_half_1,
        groove_half_2=parsed.groove_half_2,
        groove_status=parsed.status,
        is_null=False,
        is_questionable=False,
        is_pseudogene=False,
        is_functional=parsed.status in FUNCTIONAL_GROOVE_STATUSES,
        representative_allele=anchor.normalized,
        representative_policy="assembled_overlap",
    )


def _classify_two_field_alias_group(
    group_key: str,
    group_records: List[MHCIndexRecord],
) -> Dict[str, object]:
    unique_records = _unique_sequence_records(group_records)
    if len(unique_records) <= 1:
        rep = unique_records[0] if unique_records else group_records[0]
        return {
            "kind": "unique",
            "target": rep.normalized,
            "representative_record": _alias_record_from_representative(
                alias=group_key,
                representative=rep,
                policy="unique",
            ),
            "candidates": sorted({rec.normalized for rec in group_records}),
        }

    longest_len = max(len(rec.sequence) for rec in unique_records)
    longest_records = [rec for rec in unique_records if len(rec.sequence) == longest_len]
    if len(longest_records) == 1:
        longest = longest_records[0]
        if all(
            rec.sequence in longest.sequence
            for rec in unique_records
            if rec.normalized != longest.normalized
        ):
            return {
                "kind": "nested_longest_unique",
                "target": longest.normalized,
                "representative_record": _alias_record_from_representative(
                    alias=group_key,
                    representative=longest,
                    policy="nested_longest_unique",
                ),
                "candidates": sorted({rec.normalized for rec in group_records}),
            }

    assembled = _assemble_overlap_representative(group_key, unique_records)
    if assembled is not None:
        return {
            "kind": "assembled_overlap",
            "target": assembled.representative_allele or group_key,
            "representative_record": assembled,
            "candidates": sorted({rec.normalized for rec in group_records}),
        }

    groove_groups: Dict[Tuple[str, str], List[MHCIndexRecord]] = {}
    for rec in unique_records:
        signature = _record_groove_signature(rec)
        if signature is None:
            groove_groups = {}
            break
        groove_groups.setdefault(signature, []).append(rec)
    if len(groove_groups) == 1 and groove_groups:
        exemplar = sorted(
            unique_records,
            key=lambda rec: (-len(rec.sequence), rec.normalized),
        )[0]
        return {
            "kind": "groove_equivalent_exemplar",
            "target": exemplar.normalized,
            "representative_record": _alias_record_from_representative(
                alias=group_key,
                representative=exemplar,
                policy="groove_equivalent_exemplar",
            ),
            "candidates": sorted({rec.normalized for rec in group_records}),
        }

    lengths = sorted({len(rec.sequence) for rec in unique_records})
    if len(lengths) == 1:
        reason = "same_length_diff_content"
    else:
        reason = "non_nested_seq_conflict"
    return {
        "kind": reason,
        "target": None,
        "representative_record": None,
        "candidates": sorted({rec.normalized for rec in group_records}),
        "candidate_lengths": lengths,
        "group_key": group_key,
    }


def _build_alias_resolution(
    records: Dict[str, MHCIndexRecord],
) -> Tuple[Dict[str, str], Dict[str, Dict[str, object]], Dict[str, MHCIndexRecord]]:
    aliases: Dict[str, str] = {}
    ambiguous_aliases: Dict[str, Dict[str, object]] = {}
    alias_records: Dict[str, MHCIndexRecord] = {}

    for normalized in records:
        for variant in _alias_variants(normalized):
            _emit_alias(aliases, variant, normalized)
            if ":" not in variant:
                continue
            parts = variant.split(":")
            for i in range(1, len(parts)):
                _emit_alias(aliases, ":".join(parts[:i]), normalized)

    two_field_groups: Dict[str, List[MHCIndexRecord]] = {}
    for rec in records.values():
        try:
            group_key = normalize_allele_name(rec.normalized)
        except Exception:
            continue
        two_field_groups.setdefault(group_key, []).append(rec)

    for group_key, group_records in two_field_groups.items():
        decision = _classify_two_field_alias_group(group_key, group_records)
        for variant in _alias_variants(group_key):
            if decision["target"]:
                _emit_alias_override(aliases, variant, str(decision["target"]))
                for case_variant in {variant, variant.upper(), variant.lower()}:
                    ambiguous_aliases.pop(case_variant, None)
                    alias_records.pop(case_variant, None)
                    rep = decision.get("representative_record")
                    if isinstance(rep, MHCIndexRecord):
                        alias_records[case_variant] = rep
            else:
                for case_variant in {variant, variant.upper(), variant.lower()}:
                    aliases.pop(case_variant, None)
                    alias_records.pop(case_variant, None)
                    ambiguous_aliases[case_variant] = dict(decision)

    return aliases, ambiguous_aliases, alias_records


def build_mhc_sequence_lookup(
    records: Dict[str, MHCIndexRecord],
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    aliases, _, alias_records = _build_alias_resolution(records)

    for record in records.values():
        seq = str(record.sequence or "").strip().upper()
        if not seq:
            continue
        for token in {record.normalized, record.allele_raw}:
            name = str(token or "").strip()
            if not name:
                continue
            lookup[name] = seq
            lookup[name.upper()] = seq

    for alias, normalized in aliases.items():
        rec = alias_records.get(alias)
        if rec is None:
            rec = records.get(normalized)
        if rec is None:
            continue
        seq = str(rec.sequence or "").strip().upper()
        if not seq:
            continue
        lookup[alias] = seq

    return lookup


def resolve_alleles(
    index_csv: str,
    alleles: Iterable[str],
    include_sequence: bool = True,
) -> List[Dict[str, object]]:
    _require_mhcgnomes()
    records = load_mhc_index(index_csv)
    aliases, ambiguous_aliases, alias_records = _build_alias_resolution(records)

    def _resolve_record(
        key: str,
    ) -> Tuple[Optional[MHCIndexRecord], Optional[Dict[str, object]]]:
        for variant in (key, key.upper(), key.lower()):
            ambiguous = ambiguous_aliases.get(variant)
            if ambiguous is not None:
                return None, ambiguous
            alias_rec = alias_records.get(variant)
            if alias_rec is not None:
                return alias_rec, None
            alias_target = aliases.get(variant)
            if alias_target:
                rec = records.get(alias_target)
                if rec is not None:
                    return rec, None
            rec = records.get(variant)
            if rec is not None:
                return rec, None
        if ":" in key:
            candidate = key
            while ":" in candidate:
                candidate = candidate.rsplit(":", 1)[0]
                for variant in (candidate, candidate.upper(), candidate.lower()):
                    ambiguous = ambiguous_aliases.get(variant)
                    if ambiguous is not None:
                        return None, ambiguous
                    alias_target = aliases.get(variant, "")
                    if alias_target:
                        rec = records.get(alias_target)
                        if rec is not None:
                            return rec, None
                    rec = records.get(variant)
                    if rec is not None:
                        return rec, None
        return None, None

    results: List[Dict[str, object]] = []
    for allele in alleles:
        allele = allele.strip()
        if not allele:
            continue

        normalized = ""
        parse_error = None
        try:
            normalized, _, _, _ = _normalize_with_mhcgnomes(allele)
        except Exception as exc:
            parse_error = exc

        lookup_keys: List[str] = []
        if normalized:
            lookup_keys.append(normalized)
        fallback = _normalize_allele_token(allele)
        if fallback:
            lookup_keys.append(fallback)
        if " " not in allele:
            try:
                normalized_legacy = normalize_allele_name(allele)
            except Exception:
                normalized_legacy = ""
            if normalized_legacy:
                lookup_keys.append(normalized_legacy)
        if allele not in lookup_keys:
            lookup_keys.append(allele)

        rec = None
        ambiguity = None
        for key in lookup_keys:
            rec, ambiguity = _resolve_record(key)
            if rec is not None or ambiguity is not None:
                break

        if rec is None:
            row = {
                "input": allele,
                "normalized": normalized,
                "resolved": "",
                "found": False,
            }
            if ambiguity is not None:
                row["ambiguous"] = True
                row["ambiguity_reason"] = str(ambiguity.get("kind", "ambiguous"))
                row["candidate_count"] = len(ambiguity.get("candidates", ()))
                row["candidate_alleles"] = list(ambiguity.get("candidates", ()))
                if ambiguity.get("candidate_lengths"):
                    row["candidate_lengths"] = list(ambiguity["candidate_lengths"])
            if parse_error is not None:
                row["error"] = str(parse_error)
            results.append(row)
            continue

        row = {
            "input": allele,
            "normalized": normalized or fallback,
            "resolved": rec.representative_allele or rec.normalized,
            "found": True,
            "gene": rec.gene,
            "mhc_class": normalize_mhc_class(rec.mhc_class, default=rec.mhc_class),
            "species": rec.species,
            "source": rec.source,
            "seq_len": rec.seq_len,
        }
        if rec.representative_allele:
            row["representative_allele"] = rec.representative_allele
        if rec.representative_policy:
            row["representative_policy"] = rec.representative_policy
        if rec.mature_start is not None:
            row["mature_start"] = rec.mature_start
        if rec.groove_half_1:
            row["groove_half_1"] = rec.groove_half_1
        if rec.groove_half_2:
            row["groove_half_2"] = rec.groove_half_2
        if rec.groove_status:
            row["groove_status"] = rec.groove_status
        if rec.is_null:
            row["is_null"] = rec.is_null
        if rec.is_questionable:
            row["is_questionable"] = rec.is_questionable
        if rec.is_pseudogene:
            row["is_pseudogene"] = rec.is_pseudogene
        if rec.is_functional is not None:
            row["is_functional"] = rec.is_functional
        if include_sequence:
            row["sequence"] = rec.sequence
        results.append(row)

    return results


def _category_species_prefix(species: Optional[str]) -> str:
    normalized = normalize_species_label(species)
    return normalized or "other"


def _fallback_unresolved_category(token: str) -> str:
    upper = token.upper()
    if upper.startswith(("H2-", "H-2")):
        if "/" in upper:
            return "murine_pair_shorthand"
        if " CLASS " in f" {upper} ":
            return "murine_haplotype"
        if "*" in upper:
            return "murine_allele_missing_sequence"
        return "murine_coarse_label"
    if upper.startswith("HLA-"):
        if "BTN3" in upper:
            return "human_nonclassical_gene"
        if "*" in upper:
            return "human_allele_missing_sequence"
        if re.match(r"^HLA-[A-Z0-9]+[0-9]+$", upper):
            return "human_serotype"
        return "human_locus"
    return "unparseable_or_out_of_scope"


def classify_unresolved_allele(token: str) -> Dict[str, str]:
    """Classify unresolved allele tokens into deterministic triage buckets."""
    value = token.strip()
    details: Dict[str, str] = {
        "token": value,
        "category": _fallback_unresolved_category(value),
        "parsed_type": "",
        "normalized": "",
        "species": "",
        "mhc_class": "",
        "gene": "",
        "parse_error": "",
    }
    if not value:
        return details

    try:
        mhcgnomes = _require_mhcgnomes()
        parsed = mhcgnomes.parse(value)
    except Exception as exc:
        details["parse_error"] = str(exc)
        return details

    parsed_type = type(parsed).__name__
    species_name = getattr(getattr(parsed, "species", None), "name", None)
    gene_name = getattr(getattr(parsed, "gene", None), "name", None)
    mhc_class = normalize_mhc_class(getattr(parsed, "mhc_class", None))
    prefix = _category_species_prefix(species_name)
    category = f"{prefix}_{parsed_type.lower()}"
    if parsed_type == "Pair":
        category = f"{prefix}_pair_shorthand" if prefix == "murine" else f"{prefix}_pair"
    elif parsed_type == "Haplotype":
        category = f"{prefix}_haplotype"
    elif parsed_type == "Serotype":
        category = f"{prefix}_serotype"
    elif parsed_type in {"Class1Locus", "Class2Locus", "Locus"}:
        category = f"{prefix}_locus"
    elif parsed_type == "Gene":
        gene_upper = (gene_name or "").upper()
        if prefix == "human" and gene_upper.startswith("BTN3"):
            category = "human_nonclassical_gene"
        else:
            category = f"{prefix}_gene"
    elif parsed_type == "Allele":
        category = f"{prefix}_allele_missing_sequence"

    details.update(
        {
            "category": category,
            "parsed_type": parsed_type,
            "normalized": parsed.to_string(),
            "species": str(species_name or ""),
            "mhc_class": str(mhc_class or ""),
            "gene": str(gene_name or ""),
            "parse_error": "",
        }
    )
    return details


def _count_values(rows: List[Dict[str, str]], key: str) -> Dict[str, int]:
    counts = Counter((row.get(key) or "").strip() for row in rows if (row.get(key) or "").strip())
    return {name: counts[name] for name in sorted(counts)}


def summarize_mhc_index(index_csv: str) -> Dict[str, object]:
    """Summarize an index CSV with category counts."""
    path = Path(index_csv)
    if not path.exists():
        raise MHCIndexError(f"Index CSV not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise MHCIndexError(f"No records found in index: {path}")

    summary = {
        "total_records": len(rows),
        "by_source": _count_values(rows, "source"),
        "by_species": _count_values(rows, "species"),
        "by_mhc_class": _count_values(rows, "mhc_class"),
        "by_gene": _count_values(rows, "gene"),
    }
    if any((row.get("groove_status") or "").strip() for row in rows):
        summary["by_groove_status"] = _count_values(rows, "groove_status")
        functional_true = 0
        functional_false = 0
        for row in rows:
            token = str(row.get("is_functional") or "").strip().lower()
            if token in {"1", "true", "t", "yes", "y"}:
                functional_true += 1
            elif token in {"0", "false", "f", "no", "n"}:
                functional_false += 1
        summary["functional_true"] = functional_true
        summary["functional_false"] = functional_false
    return summary


def validate_mhc_index(index_csv: str) -> Dict[str, object]:
    """Validate index CSV content and canonical allele formatting."""
    _require_mhcgnomes()
    path = Path(index_csv)
    if not path.exists():
        raise MHCIndexError(f"Index CSV not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing_fields = [field for field in INDEX_FIELDS if field not in fields]
        if missing_fields:
            errors = [
                {
                    "row": 1,
                    "code": "missing_columns",
                    "message": f"Missing required columns: {', '.join(missing_fields)}",
                }
            ]
            return {
                "valid": False,
                "total_rows": 0,
                "error_count": len(errors),
                "warning_count": 0,
                "errors": errors,
                "warnings": [],
            }

        errors: List[Dict[str, object]] = []
        warnings: List[Dict[str, object]] = []
        seen_normalized: Dict[str, int] = {}
        total_rows = 0

        for row_idx, row in enumerate(reader, start=2):
            total_rows += 1
            normalized = (row.get("normalized") or "").strip()
            if not normalized:
                errors.append(
                    {
                        "row": row_idx,
                        "code": "missing_normalized",
                        "message": "Missing normalized allele value",
                    }
                )
            elif normalized in seen_normalized:
                errors.append(
                    {
                        "row": row_idx,
                        "code": "duplicate_normalized",
                        "message": f"Duplicate normalized allele first seen at row {seen_normalized[normalized]}",
                    }
                )
            else:
                seen_normalized[normalized] = row_idx

                try:
                    canonical, _, _, _ = _normalize_with_mhcgnomes(normalized)
                    if canonical != normalized:
                        warnings.append(
                            {
                                "row": row_idx,
                                "code": "non_canonical_normalized",
                                "message": f"Canonical form is '{canonical}'",
                            }
                        )
                except Exception as exc:
                    errors.append(
                        {
                            "row": row_idx,
                            "code": "mhcgnomes_parse_failed",
                            "message": str(exc),
                        }
                    )

            sequence = (row.get("sequence") or "").strip()
            if not sequence:
                errors.append(
                    {
                        "row": row_idx,
                        "code": "missing_sequence",
                        "message": "Missing sequence value",
                    }
                )
            elif _looks_like_nucleotide_sequence(sequence):
                errors.append(
                    {
                        "row": row_idx,
                        "code": "nucleotide_like_sequence",
                        "message": "Sequence looks nucleotide-like; expected protein FASTA content",
                    }
                )
            elif len(sequence) < MIN_MHC_SEQUENCE_LEN:
                errors.append(
                    {
                        "row": row_idx,
                        "code": "sequence_too_short",
                        "message": (
                            "Sequence shorter than minimum accepted groove-bearing "
                            f"MHC fragment ({MIN_MHC_SEQUENCE_LEN} aa)"
                        ),
                    }
                )

            seq_len_text = (row.get("seq_len") or "").strip()
            if seq_len_text:
                if seq_len_text.isdigit():
                    seq_len = int(seq_len_text)
                    if sequence and seq_len != len(sequence):
                        warnings.append(
                            {
                                "row": row_idx,
                                "code": "seq_len_mismatch",
                                "message": f"seq_len={seq_len} but sequence length is {len(sequence)}",
                            }
                        )
                else:
                    warnings.append(
                        {
                            "row": row_idx,
                            "code": "invalid_seq_len",
                            "message": f"Non-numeric seq_len: '{seq_len_text}'",
                        }
                    )

    return {
        "valid": len(errors) == 0,
        "total_rows": total_rows,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }
