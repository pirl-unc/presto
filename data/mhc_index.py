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
    normalize_mhc_class,
    normalize_allele_name,
    normalize_species_label,
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

    @staticmethod
    def from_row(row: Dict[str, str]) -> "MHCIndexRecord":
        seq = (row.get("sequence") or "").strip()
        seq_len = row.get("seq_len")
        if seq_len and seq_len.isdigit():
            length = int(seq_len)
        else:
            length = len(seq)
        return MHCIndexRecord(
            allele_raw=(row.get("allele_raw") or row.get("allele") or "").strip(),
            normalized=(row.get("normalized") or "").strip(),
            gene=(row.get("gene") or "").strip(),
            mhc_class=(row.get("mhc_class") or "").strip(),
            species=(row.get("species") or "").strip(),
            source=(row.get("source") or "").strip(),
            seq_len=length,
            sequence=seq,
        )


class MHCIndexError(RuntimeError):
    """Raised when MHC index operations fail."""


def _require_mhcgnomes():
    try:
        import mhcgnomes  # type: ignore
    except Exception as exc:
        raise MHCIndexError(
            "mhcgnomes is required for allele name normalization. "
            "Install it with `pip install mhcgnomes`."
        ) from exc
    return mhcgnomes


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


def _infer_species_from_normalized(normalized: str) -> str:
    """Best-effort species inference for fallback indexed alleles."""
    species = infer_species(normalized)
    if species != "human":
        return species
    if "-" in normalized:
        prefix = normalized.split("-", 1)[0].strip()
        if prefix and prefix.upper() != "HLA":
            return prefix
    return species


def _fallback_header_allele(
    header: str,
) -> Optional[Tuple[str, Optional[str], Optional[str], Optional[str], str]]:
    """Fallback parser for headers not handled by mhcgnomes."""
    for token in _candidate_tokens(header):
        normalized = _normalize_allele_token(token)
        if "*" not in normalized:
            continue
        gene = infer_gene(normalized)
        mhc_class = normalize_mhc_class(
            infer_mhc_class(normalized),
            default=infer_mhc_class(normalized),
        )
        species = _infer_species_from_normalized(normalized)
        return normalized, gene, mhc_class, species, token
    return None


def _normalize_with_mhcgnomes(token: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    mhcgnomes = _require_mhcgnomes()
    parsed = mhcgnomes.parse(token)
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
    fallback = _fallback_header_allele(header)
    if fallback is not None:
        return fallback
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
                species = infer_species(normalized)

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


def _build_alias_map(records: Dict[str, MHCIndexRecord]) -> Dict[str, str]:
    def _emit_alias(alias_map: Dict[str, str], alias: str, normalized: str) -> None:
        token = alias.strip()
        if token:
            for variant in {token, token.upper(), token.lower()}:
                alias_map.setdefault(variant, normalized)

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

    aliases: Dict[str, str] = {}
    for normalized in records:
        for variant in _alias_variants(normalized):
            _emit_alias(aliases, variant, normalized)
            if ":" not in variant:
                continue
            parts = variant.split(":")
            for i in range(1, len(parts)):
                _emit_alias(aliases, ":".join(parts[:i]), normalized)
    return aliases


def resolve_alleles(
    index_csv: str,
    alleles: Iterable[str],
    include_sequence: bool = True,
) -> List[Dict[str, object]]:
    _require_mhcgnomes()
    records = load_mhc_index(index_csv)
    aliases = _build_alias_map(records)

    def _resolve_record(key: str) -> Optional[MHCIndexRecord]:
        for variant in (key, key.upper(), key.lower()):
            rec = records.get(variant)
            if rec is not None:
                return rec
            alias_target = aliases.get(variant)
            if alias_target:
                rec = records.get(alias_target)
                if rec is not None:
                    return rec
        if ":" in key:
            candidate = key
            while ":" in candidate:
                candidate = candidate.rsplit(":", 1)[0]
                for variant in (candidate, candidate.upper(), candidate.lower()):
                    rec = records.get(variant)
                    if rec is not None:
                        return rec
                    alias_target = aliases.get(variant, "")
                    if alias_target:
                        rec = records.get(alias_target)
                        if rec is not None:
                            return rec
        return None

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
        for key in lookup_keys:
            rec = _resolve_record(key)
            if rec is not None:
                break

        if rec is None:
            row = {
                "input": allele,
                "normalized": normalized,
                "resolved": "",
                "found": False,
            }
            if parse_error is not None:
                row["error"] = str(parse_error)
            results.append(row)
            continue

        row = {
            "input": allele,
            "normalized": normalized or fallback,
            "resolved": rec.normalized,
            "found": True,
            "gene": rec.gene,
            "mhc_class": normalize_mhc_class(rec.mhc_class, default=rec.mhc_class),
            "species": rec.species,
            "source": rec.source,
            "seq_len": rec.seq_len,
        }
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

    return {
        "total_records": len(rows),
        "by_source": _count_values(rows, "source"),
        "by_species": _count_values(rows, "species"),
        "by_mhc_class": _count_values(rows, "mhc_class"),
        "by_gene": _count_values(rows, "gene"),
    }


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
