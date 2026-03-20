"""Cross-source deduplication for immunology datasets.

Handles merging and deduplicating data from multiple sources (IEDB, CEDAR,
VDJdb, McPAS, etc.) based on publication references (PubMed IDs).

Each source has different file formats but often cites the same papers.
This module normalizes the data and removes duplicate entries that appear
in multiple databases from the same original publication.
"""

import csv
import io
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Any, Iterable
import zipfile
from tqdm.auto import tqdm

from .allele_resolver import (
    expand_mhc_restriction,
    infer_mhc_class_optional,
    infer_species,
    parse_allele_name,
)
from .vocab import normalize_organism


_CELL_CONTEXT_WS_RE = re.compile(r"\s+")
_ALLELE_SPLIT_RE = re.compile(r"[,;/]")
_MURINE_ALLELE_SHORTHAND_RE = re.compile(r"^H2-[A-Z0-9]+$", re.IGNORECASE)
_STAR_ALLELE_RE = re.compile(r"^[A-Z0-9-]+\*[A-Z0-9:]+$", re.IGNORECASE)


@dataclass(slots=True)
class UnifiedRecord:
    """Unified record format for cross-source deduplication.

    Attributes:
        peptide: Epitope/peptide sequence
        mhc_allele: MHC allele (normalized format)
        mhc_class: MHC class (I or II)
        pmid: PubMed ID (normalized, no prefix)
        source: Original data source (iedb, vdjdb, mcpas, etc.)
        record_type: Type of record (binding, tcell, bcell, tcr)

        # For binding assays
        value: Measurement value (IC50, KD, etc.)
        value_type: Type of measurement
        qualifier: -1 for '<', 0 for '=', 1 for '>'

        # For T-cell assays
        response: Response category (positive, negative)

        # For TCR data
        cdr3_alpha: CDR3 alpha chain sequence
        cdr3_beta: CDR3 beta chain sequence
        trav: TRAV gene
        trbv: TRBV gene

        # Metadata
        species: Species (human, mouse, etc.)
        raw_data: Original row data for reference
    """

    peptide: str
    mhc_allele: str
    mhc_allele_set: Optional[str] = None
    mhc_allele_provenance: Optional[str] = None
    mhc_allele_bag_size: Optional[int] = None
    mhc_class: str = "I"
    pmid: Optional[str] = None
    doi: Optional[str] = None
    reference_text: Optional[str] = None
    source: str = ""
    record_type: str = ""  # binding, tcell, bcell, tcr

    # Binding assay fields
    value: Optional[float] = None
    value_type: Optional[str] = None
    qualifier: int = 0

    # T-cell assay fields
    response: Optional[str] = None
    assay_type: Optional[str] = None
    assay_method: Optional[str] = None
    apc_name: Optional[str] = None
    effector_culture_condition: Optional[str] = None
    apc_culture_condition: Optional[str] = None
    in_vitro_process_type: Optional[str] = None
    in_vitro_responder_cell: Optional[str] = None
    in_vitro_stimulator_cell: Optional[str] = None

    # TCR fields
    cdr3_alpha: Optional[str] = None
    cdr3_beta: Optional[str] = None
    trav: Optional[str] = None
    trbv: Optional[str] = None
    evidence_method_identification: Optional[str] = None
    evidence_method_verification: Optional[str] = None
    evidence_singlecell: Optional[str] = None
    evidence_sequencing: Optional[str] = None
    evidence_score: Optional[int] = None

    # Metadata
    species: Optional[str] = None
    antigen_species: Optional[str] = None
    cell_hla_allele_set: Optional[str] = None
    cell_hla_n_alleles: Optional[int] = None
    raw_data: Optional[Dict[str, str]] = None
    _assay_bucket_cache: Optional[str] = field(default=None, repr=False, compare=False)

    def dedup_key(self) -> str:
        """Generate key for deduplication.

        Includes normalized species to prevent merging records from
        experiments in different host organisms (e.g. human vs murine).
        """
        sp = self.species or ""
        if self.record_type == "tcr":
            # For TCR data, include CDR3 sequences
            return (
                f"{self.peptide}|{self.mhc_allele}|{self.cdr3_alpha or ''}|"
                f"{self.cdr3_beta or ''}|{sp}|{self.record_type}"
            )
        elif self.record_type == "binding":
            return f"{self.peptide}|{self.mhc_allele}|{self.value_type or 'binding'}|{sp}|{self.record_type}"
        elif self.record_type == "tcell":
            response = normalize_binary_response(self.response) or "unknown"
            return f"{self.peptide}|{self.mhc_allele}|{response}|{sp}|{self.record_type}"
        elif self.record_type == "bcell":
            response = normalize_binary_response(self.response) or "unknown"
            return f"{self.peptide}|{self.mhc_allele}|{response}|{sp}|{self.record_type}"
        else:
            return f"{self.peptide}|{self.mhc_allele}|{sp}|{self.record_type}"

    def reference_key(self) -> str:
        """Generate key based on reference."""
        if self.pmid:
            return f"pmid:{self.pmid}"
        return f"source:{self.source}"


def normalize_pmid(raw_pmid: str) -> Optional[str]:
    """Normalize PubMed ID to just the numeric part.

    Handles formats like:
    - "12345678"
    - "PMID:12345678"
    - "PMID: 12345678"
    - "https://pubmed.ncbi.nlm.nih.gov/12345678/"
    """
    if not raw_pmid:
        return None

    raw_pmid = str(raw_pmid).strip()

    # Already just numbers
    if raw_pmid.isdigit():
        return raw_pmid

    # Extract numeric part
    match = re.search(r"(\d{6,10})", raw_pmid)
    if match:
        return match.group(1)

    return None


def normalize_doi(raw_doi: str) -> Optional[str]:
    """Normalize DOI strings to lowercase canonical form."""
    if not raw_doi:
        return None
    text = str(raw_doi).strip().lower()
    if not text:
        return None
    text = text.replace("https://doi.org/", "").replace("http://doi.org/", "")
    text = text.replace("doi:", "").strip()
    match = re.search(r"(10\.\d{4,9}/\S+)", text)
    if not match:
        return None
    doi = match.group(1).rstrip(".;,)")
    return doi or None


def normalize_binary_response(raw_response: Optional[str]) -> Optional[str]:
    """Normalize response text to positive/negative/unknown buckets."""
    if raw_response is None:
        return None
    text = str(raw_response).strip().lower()
    if not text:
        return None
    if text in {"positive", "pos", "1", "true", "yes", "+"}:
        return "positive"
    if text in {"negative", "neg", "0", "false", "no", "-"}:
        return "negative"
    if "positive" in text:
        return "positive"
    if "negative" in text:
        return "negative"
    return "unknown"


def _normalize_reference_text(raw_text: Optional[str]) -> Optional[str]:
    """Normalize free-text references for fuzzy matching."""
    if raw_text is None:
        return None
    text = str(raw_text).strip().lower()
    if not text:
        return None
    # Remove URL wrappers and punctuation noise.
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s:/.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _extract_reference_text(
    row: List[str],
    reference_text_cols: List[int],
    *,
    keep: bool,
) -> Optional[str]:
    """Collect and normalize reference text only when it is needed."""
    if not keep or not reference_text_cols:
        return None

    parts: List[str] = []
    seen: set[str] = set()
    for idx in reference_text_cols:
        if idx >= len(row):
            continue
        value = row[idx].strip()
        if not value or value in seen:
            continue
        seen.add(value)
        parts.append(value)

    if not parts:
        return None
    return _normalize_reference_text(" | ".join(parts))


@lru_cache(maxsize=50000)
def normalize_allele(raw_allele: str) -> str:
    """Normalize MHC allele to standard format using mhcgnomes.

    Uses mhcgnomes as the canonical parser across species and naming
    conventions. If mhcgnomes cannot parse a token, preserve the raw allele
    string instead of applying a handwritten normalization heuristic.

    Examples:
    - "HLA-A*02:01" -> "HLA-A*02:01"
    - "HLA-A*0201" -> "HLA-A*02:01"
    - "A*0201" -> "HLA-A*02:01"
    - "HLA-A2" -> "HLA-A2"
    - "H-2Kb" -> "H2-K*b"
    """
    if not raw_allele:
        return ""

    allele = raw_allele.strip()
    if not allele:
        return ""

    try:
        result = parse_allele_name(allele)
    except Exception:
        return allele
    return result.to_string() if result else allele


def parse_allele_set_field(raw_alleles: Optional[str]) -> List[str]:
    """Parse a serialized allele-set field into normalized exact alleles."""
    if raw_alleles is None:
        return []
    text = str(raw_alleles).strip()
    if not text:
        return []
    return [token for token in text.split(";") if token]


def _restriction_record_fields(
    raw_allele: str,
    raw_mhc_class: Optional[str] = None,
) -> Dict[str, Any]:
    expansion = expand_mhc_restriction(raw_allele)
    allele_set = _join_allele_set(expansion.exact_alleles) if expansion.exact_alleles else None
    return {
        "mhc_allele": expansion.normalized_token,
        "mhc_allele_set": allele_set,
        "mhc_allele_provenance": expansion.provenance,
        "mhc_allele_bag_size": expansion.bag_size if expansion.exact_alleles else None,
        "mhc_class": _normalize_mhc_class(
            raw_mhc_class or "",
            expansion.normalized_token or raw_allele,
        ),
    }


def _infer_mhc_class_from_allele(allele: str, default: str = "I") -> str:
    """Infer MHC class from allele text using mhcgnomes."""
    return infer_mhc_class_optional(allele) or default


def _normalize_mhc_class(raw_class: str, allele: str) -> str:
    """Normalize class labels to I/II with allele-based fallback."""
    text = (raw_class or "").strip().upper()
    if text in {"I", "II"}:
        return text
    if text in {"MHCII"} or "CLASS II" in text or text.endswith(" II") or text.startswith("II"):
        return "II"
    if text in {"MHCI"} or "CLASS I" in text or text.endswith(" I") or text.startswith("I"):
        return "I"
    return _infer_mhc_class_from_allele(allele)


# =============================================================================
# Source-Specific Parsers
# =============================================================================


def parse_iedb_binding(file_path: Path) -> Iterator[UnifiedRecord]:
    """Parse IEDB/CEDAR MHC ligand export with streaming CSV parsing."""
    source = "cedar" if "cedar" in str(file_path).lower() else "iedb"

    if file_path.suffix == ".zip":
        with zipfile.ZipFile(file_path) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_files:
                return
            # Prefer the largest CSV member when multiple are present.
            csv_files.sort(key=lambda name: zf.getinfo(name).file_size, reverse=True)
            with zf.open(csv_files[0]) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="") as text:
                    yield from _parse_iedb_binding_stream(text, source)
        return

    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as text:
        yield from _parse_iedb_binding_stream(text, source)


def _parse_iedb_binding_stream(f, source: str) -> Iterator[UnifiedRecord]:
    """Parse IEDB/CEDAR ligand rows from a text stream."""
    reader = csv.reader(f)
    category_row = next(reader, None)
    header = next(reader, None)
    if category_row is None or header is None:
        return

    category_lower = [(c or "").strip().lower() for c in category_row]
    header_lower = [(h or "").strip().lower() for h in header]

    col_indices: Dict[str, int] = {}
    reference_text_cols: List[int] = []
    for i, (cat, h) in enumerate(zip(category_lower, header_lower)):
        if "reference" in cat:
            if "pmid" in h and "pmid" not in col_indices:
                col_indices["pmid"] = i
            elif "doi" in h and "doi" not in col_indices:
                col_indices["doi"] = i
            elif h not in {"cedar iri", "submission id", "pmid", "doi", "iri"}:
                reference_text_cols.append(i)
            continue

        if "epitope" in cat:
            if h in {"name", "description", "linear sequence"} and "peptide" not in col_indices:
                col_indices["peptide"] = i
            elif h == "source organism":
                col_indices["species_epitope"] = i
            continue

        if "host" in cat and h in {"name", "species", "organism"}:
            if "species_host" not in col_indices:
                col_indices["species_host"] = i
            continue

        if "mhc restriction" in cat:
            if h in {"name", "allele", "allele name"} and "allele" not in col_indices:
                col_indices["allele"] = i
            elif h == "class" and "mhc_class" not in col_indices:
                col_indices["mhc_class"] = i
            continue

        if "antigen presenting cell" in cat:
            if h in {"name", "cell type", "cell line", "cell"} and "apc_name" not in col_indices:
                col_indices["apc_name"] = i
            elif "tissue" in h and "apc_tissue" not in col_indices:
                col_indices["apc_tissue"] = i
            continue

        if "assay" in cat:
            if h == "method":
                col_indices["assay_method"] = i
            elif h == "response measured":
                col_indices["response_measured"] = i
            elif h == "quantitative measurement":
                col_indices["value"] = i
            elif h == "measurement inequality":
                col_indices["qualifier"] = i
            elif h == "qualitative measurement":
                col_indices["qualitative"] = i

    if "peptide" not in col_indices:
        return

    for row in reader:
        if not row:
            continue
        if len(row) < len(header):
            row = list(row) + [""] * (len(header) - len(row))

        try:
            peptide = row[col_indices["peptide"]].strip()
            if not peptide or len(peptide) < 5:
                continue

            allele = row[col_indices["allele"]].strip() if "allele" in col_indices else ""
            mhc_class = row[col_indices["mhc_class"]].strip() if "mhc_class" in col_indices else ""
            method = row[col_indices["assay_method"]].strip() if "assay_method" in col_indices else ""
            response_measured = (
                row[col_indices["response_measured"]].strip()
                if "response_measured" in col_indices
                else ""
            )
            qualitative = row[col_indices["qualitative"]].strip() if "qualitative" in col_indices else ""
            apc_name_raw = row[col_indices["apc_name"]].strip() if "apc_name" in col_indices else ""
            apc_tissue_raw = row[col_indices["apc_tissue"]].strip() if "apc_tissue" in col_indices else ""
            apc_context = apc_name_raw or apc_tissue_raw

            pmid_raw = row[col_indices["pmid"]].strip() if "pmid" in col_indices else ""
            doi_raw = row[col_indices["doi"]].strip() if "doi" in col_indices else ""
            normalized_pmid = normalize_pmid(pmid_raw)
            normalized_doi = normalize_doi(doi_raw)
            reference_text = _extract_reference_text(
                row,
                reference_text_cols,
                keep=not (normalized_pmid or normalized_doi),
            )

            value = None
            if "value" in col_indices:
                value_str = row[col_indices["value"]].strip()
                if value_str:
                    try:
                        value = float(value_str)
                    except ValueError:
                        value = None

            qualifier = 0
            if "qualifier" in col_indices:
                qual_str = row[col_indices["qualifier"]].strip()
                if "<" in qual_str:
                    qualifier = -1
                elif ">" in qual_str:
                    qualifier = 1

            species_raw = None
            antigen_species_raw = ""
            if "species_epitope" in col_indices:
                antigen_species_raw = row[col_indices["species_epitope"]].strip()
            if "species_host" in col_indices:
                species_host = row[col_indices["species_host"]].strip()
                if species_host:
                    species_raw = species_host
            if not species_raw and allele:
                species_raw = infer_species(allele)

            yield UnifiedRecord(
                peptide=peptide,
                **_restriction_record_fields(allele, mhc_class),
                pmid=normalized_pmid,
                doi=normalized_doi,
                reference_text=reference_text,
                source=source,
                record_type="binding",
                value=value,
                value_type=response_measured or method or "IC50",
                qualifier=qualifier,
                response=qualitative or None,
                assay_type=response_measured or None,
                assay_method=method or None,
                apc_name=apc_context or None,
                species=normalize_organism(species_raw),
                antigen_species=normalize_organism(antigen_species_raw) if antigen_species_raw else None,
            )
        except (IndexError, KeyError):
            continue


def parse_iedb_tcell(file_path: Path) -> Iterator[UnifiedRecord]:
    """Parse IEDB T-cell assay data."""
    source = "cedar" if "cedar" in str(file_path).lower() else "iedb"

    if file_path.suffix == ".zip":
        with zipfile.ZipFile(file_path) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_files:
                return
            csv_files.sort(key=lambda name: zf.getinfo(name).file_size, reverse=True)
            with zf.open(csv_files[0]) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="") as text:
                    yield from _parse_iedb_tcell_stream(text, source)
        return

    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as text:
        yield from _parse_iedb_tcell_stream(text, source)


def _parse_iedb_bcell_stream(f, source: str) -> Iterator[UnifiedRecord]:
    """Parse IEDB/CEDAR B-cell rows from a text stream.

    Uses category-aware column matching (like binding/tcell parsers) to correctly
    distinguish Host species (col 44) from Epitope species (col 25).
    """
    reader = csv.reader(f)
    category_row = next(reader, None)
    header = next(reader, None)
    if category_row is None or header is None:
        return

    category_lower = [(c or "").strip().lower() for c in category_row]
    header_lower = [(h or "").strip().lower() for h in header]

    col_indices: Dict[str, int] = {}
    reference_text_cols: List[int] = []
    for i, (cat, h) in enumerate(zip(category_lower, header_lower)):
        if "reference" in cat:
            if "pmid" in h and "pmid" not in col_indices:
                col_indices["pmid"] = i
            elif "doi" in h and "doi" not in col_indices:
                col_indices["doi"] = i
            elif h not in {"cedar iri", "submission id", "pmid", "doi", "iri"}:
                reference_text_cols.append(i)
            continue

        if "epitope" in cat:
            if h in {"name", "description", "linear sequence"} and "peptide" not in col_indices:
                col_indices["peptide"] = i
            elif h == "source organism" and "species_epitope" not in col_indices:
                col_indices["species_epitope"] = i
            continue

        if "host" in cat and h in {"name", "species", "organism"} and "species_host" not in col_indices:
            col_indices["species_host"] = i
            continue

        if "assay" == cat:
            if h == "qualitative measure":
                col_indices["response"] = i
            elif h == "response measured":
                col_indices["assay_type"] = i
            elif h == "method":
                col_indices["assay_method"] = i
            continue

        if "assay antibody" in cat:
            if "heavy chain isotype" in h and "heavy_isotype" not in col_indices:
                col_indices["heavy_isotype"] = i
            elif "light chain isotype" in h and "light_isotype" not in col_indices:
                col_indices["light_isotype"] = i

    if "peptide" not in col_indices:
        return

    for row in reader:
        if not row:
            continue
        if len(row) < len(header):
            row = list(row) + [""] * (len(header) - len(row))

        try:
            peptide = row[col_indices["peptide"]].strip()
            if not peptide or len(peptide) < 5:
                continue

            pmid_raw = row[col_indices["pmid"]].strip() if "pmid" in col_indices else ""
            doi_raw = row[col_indices["doi"]].strip() if "doi" in col_indices else ""
            normalized_pmid = normalize_pmid(pmid_raw)
            normalized_doi = normalize_doi(doi_raw)
            reference_text = _extract_reference_text(
                row,
                reference_text_cols,
                keep=not (normalized_pmid or normalized_doi),
            )

            response = row[col_indices["response"]].strip() if "response" in col_indices else ""
            # Normalize to positive/negative
            response_lower = response.lower()
            if "positive" in response_lower or "reactive" in response_lower:
                response_label = "positive"
            elif "negative" in response_lower or "non-reactive" in response_lower:
                response_label = "negative"
            else:
                response_label = response or None

            # Host species from Host category (NOT Epitope category)
            species_raw = None
            if "species_host" in col_indices:
                species_host = row[col_indices["species_host"]].strip()
                if species_host:
                    species_raw = species_host

            # Antigen species from Epitope Source Organism
            antigen_species_raw = (
                row[col_indices["species_epitope"]].strip()
                if "species_epitope" in col_indices
                else ""
            )

            yield UnifiedRecord(
                peptide=peptide,
                mhc_allele="",  # B-cell entries generally do not carry MHC alleles
                mhc_class="",
                pmid=normalized_pmid,
                doi=normalized_doi,
                reference_text=reference_text,
                source=source,
                record_type="bcell",
                response=response_label,
                assay_type=row[col_indices["assay_type"]].strip() if "assay_type" in col_indices else None,
                assay_method=row[col_indices["assay_method"]].strip() if "assay_method" in col_indices else None,
                species=normalize_organism(species_raw),
                antigen_species=normalize_organism(antigen_species_raw) if antigen_species_raw else None,
            )
        except (IndexError, KeyError):
            continue


def parse_iedb_bcell(file_path: Path) -> Iterator[UnifiedRecord]:
    """Parse IEDB/CEDAR B-cell data with category-aware column matching."""
    source = "cedar" if "cedar" in str(file_path).lower() else "iedb"

    if file_path.suffix == ".zip":
        with zipfile.ZipFile(file_path) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_files:
                return
            csv_files.sort(key=lambda name: zf.getinfo(name).file_size, reverse=True)
            with zf.open(csv_files[0]) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="") as text:
                    yield from _parse_iedb_bcell_stream(text, source)
        return

    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as text:
        yield from _parse_iedb_bcell_stream(text, source)


def _parse_iedb_tcell_stream(f, source: str) -> Iterator[UnifiedRecord]:
    """Parse IEDB/CEDAR T-cell rows from a text stream."""
    reader = csv.reader(f)
    category_row = next(reader, None)
    header = next(reader, None)
    if category_row is None or header is None:
        return

    category_lower = [(c or "").strip().lower() for c in category_row]
    header_lower = [(h or "").strip().lower() for h in header]

    col_indices: Dict[str, int] = {}
    reference_text_cols: List[int] = []
    for i, (cat, h) in enumerate(zip(category_lower, header_lower)):
        if "reference" in cat:
            if "pmid" in h and "pmid" not in col_indices:
                col_indices["pmid"] = i
            elif "doi" in h and "doi" not in col_indices:
                col_indices["doi"] = i
            elif h not in {"cedar iri", "submission id", "pmid", "doi", "iri"}:
                reference_text_cols.append(i)
            continue

        if "epitope" in cat:
            if h in {"name", "description", "linear sequence"} and "peptide" not in col_indices:
                col_indices["peptide"] = i
            elif h == "source organism" and "species_epitope" not in col_indices:
                col_indices["species_epitope"] = i
            continue

        if "mhc restriction" in cat:
            if h == "name" and "allele" not in col_indices:
                col_indices["allele"] = i
            elif h == "class" and "mhc_class" not in col_indices:
                col_indices["mhc_class"] = i
            continue

        if cat == "assay":
            if h == "qualitative measurement":
                col_indices["response"] = i
            elif h == "response measured":
                col_indices["assay_type"] = i
            elif h == "method":
                col_indices["assay_method"] = i
            continue

        if cat == "effector cell" and h == "culture condition":
            col_indices["effector_culture_condition"] = i
            continue

        if cat == "antigen presenting cell":
            if h == "name":
                col_indices["apc_name"] = i
            elif h == "culture condition":
                col_indices["apc_culture_condition"] = i
            continue

        if cat == "in vitro process" and h == "process type":
            col_indices["in_vitro_process_type"] = i
            continue
        if cat == "in vitro responder cell" and h == "name":
            col_indices["in_vitro_responder_cell"] = i
            continue
        if cat == "in vitro stimulator cell" and h == "name":
            col_indices["in_vitro_stimulator_cell"] = i
            continue

        if "host" in cat and h in {"name", "species", "organism"} and "species_host" not in col_indices:
            col_indices["species_host"] = i

    if "peptide" not in col_indices:
        return

    for row in reader:
        if not row:
            continue
        if len(row) < len(header):
            row = list(row) + [""] * (len(header) - len(row))

        try:
            peptide = row[col_indices["peptide"]].strip()
            if not peptide or len(peptide) < 5:
                continue

            allele = row[col_indices["allele"]].strip() if "allele" in col_indices else ""
            mhc_class_raw = row[col_indices["mhc_class"]].strip() if "mhc_class" in col_indices else ""
            pmid_raw = row[col_indices["pmid"]].strip() if "pmid" in col_indices else ""
            doi_raw = row[col_indices["doi"]].strip() if "doi" in col_indices else ""
            normalized_pmid = normalize_pmid(pmid_raw)
            normalized_doi = normalize_doi(doi_raw)
            reference_text = _extract_reference_text(
                row,
                reference_text_cols,
                keep=not (normalized_pmid or normalized_doi),
            )

            response = row[col_indices["response"]].strip() if "response" in col_indices else ""
            assay_type = row[col_indices["assay_type"]].strip() if "assay_type" in col_indices else ""
            assay_method = row[col_indices["assay_method"]].strip() if "assay_method" in col_indices else ""
            apc_name = row[col_indices["apc_name"]].strip() if "apc_name" in col_indices else ""
            effector_culture = (
                row[col_indices["effector_culture_condition"]].strip()
                if "effector_culture_condition" in col_indices
                else ""
            )
            apc_culture = (
                row[col_indices["apc_culture_condition"]].strip()
                if "apc_culture_condition" in col_indices
                else ""
            )
            in_vitro_process = (
                row[col_indices["in_vitro_process_type"]].strip()
                if "in_vitro_process_type" in col_indices
                else ""
            )
            in_vitro_responder = (
                row[col_indices["in_vitro_responder_cell"]].strip()
                if "in_vitro_responder_cell" in col_indices
                else ""
            )
            in_vitro_stimulator = (
                row[col_indices["in_vitro_stimulator_cell"]].strip()
                if "in_vitro_stimulator_cell" in col_indices
                else ""
            )
            species_raw = (
                row[col_indices["species_host"]].strip()
                if "species_host" in col_indices
                else None
            )
            if not species_raw and allele:
                species_raw = infer_species(allele)
            antigen_species_raw = (
                row[col_indices["species_epitope"]].strip()
                if "species_epitope" in col_indices
                else ""
            )

            yield UnifiedRecord(
                peptide=peptide,
                **_restriction_record_fields(allele, mhc_class_raw),
                pmid=normalized_pmid,
                doi=normalized_doi,
                reference_text=reference_text,
                source=source,
                record_type="tcell",
                response=response,
                assay_type=assay_type or None,
                assay_method=assay_method or None,
                apc_name=apc_name or None,
                effector_culture_condition=effector_culture or None,
                apc_culture_condition=apc_culture or None,
                in_vitro_process_type=in_vitro_process or None,
                in_vitro_responder_cell=in_vitro_responder or None,
                in_vitro_stimulator_cell=in_vitro_stimulator or None,
                species=normalize_organism(species_raw),
                antigen_species=normalize_organism(antigen_species_raw) if antigen_species_raw else None,
            )
        except (IndexError, KeyError):
            continue


def parse_vdjdb(file_path: Path) -> Iterator[UnifiedRecord]:
    """Parse VDJdb TCR-pMHC data."""
    # Handle zip files
    if file_path.suffix == ".zip":
        with zipfile.ZipFile(file_path) as zf:
            txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
            # Prefer vdjdb.txt (exact match) or vdjdb_full.txt
            preferred = ["vdjdb.txt", "vdjdb_full.txt", "vdjdb_full_filtered.txt"]
            txt_file = None
            for pref in preferred:
                if pref in txt_files:
                    txt_file = pref
                    break
            if not txt_file and txt_files:
                # Fallback: any txt file with 'vdjdb' in name, not slim/broken
                for f in txt_files:
                    if (
                        "vdjdb" in f.lower()
                        and "slim" not in f.lower()
                        and "broken" not in f.lower()
                        and "meta" not in f.lower()
                    ):
                        txt_file = f
                        break
            if txt_file:
                with zf.open(txt_file) as f:
                    content = f.read().decode("utf-8", errors="replace")
                    yield from _parse_vdjdb_content(content)
        return

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        yield from _parse_vdjdb_content(f.read())


def _parse_vdjdb_content(content: str) -> Iterator[UnifiedRecord]:
    """Parse VDJdb content."""
    lines = content.strip().split("\n")
    if not lines:
        return

    header = lines[0].split("\t")
    header_map = {h: i for i, h in enumerate(header)}

    for line in lines[1:]:
        row = line.split("\t")
        if len(row) < len(header):
            continue

        try:
            epitope = row[header_map.get("antigen.epitope", 0)]
            if not epitope or len(epitope) < 5:
                continue

            cdr3 = row[header_map.get("cdr3", 0)]
            gene = row[header_map.get("gene", 0)]  # TRA or TRB
            mhc_a = row[header_map.get("mhc.a", 0)]
            mhc_class = row[header_map.get("mhc.class", 0)]
            ref_id = row[header_map.get("reference.id", 0)]
            species = row[header_map.get("species", 0)]
            v_gene = row[header_map.get("v.segm", 0)]
            method_raw = row[header_map["method"]] if "method" in header_map else ""
            score_raw = row[header_map["vdjdb.score"]] if "vdjdb.score" in header_map else ""
            antigen_species_raw = (
                row[header_map["antigen.species"]]
                if "antigen.species" in header_map and header_map["antigen.species"] < len(row)
                else ""
            ).strip()
            try:
                method_meta = json.loads(method_raw) if method_raw else {}
            except (TypeError, ValueError, json.JSONDecodeError):
                method_meta = {}
            normalized_pmid = normalize_pmid(ref_id)
            normalized_doi = normalize_doi(ref_id)

            species_raw = species if species else (infer_species(mhc_a) if mhc_a else None)

            yield UnifiedRecord(
                peptide=epitope,
                **_restriction_record_fields(mhc_a, mhc_class),
                pmid=normalized_pmid,
                doi=normalized_doi,
                reference_text=_normalize_reference_text(ref_id),
                source="vdjdb",
                record_type="tcr",
                cdr3_alpha=cdr3 if gene == "TRA" else None,
                cdr3_beta=cdr3 if gene == "TRB" else None,
                trav=v_gene if gene == "TRA" else None,
                trbv=v_gene if gene == "TRB" else None,
                evidence_method_identification=str(method_meta.get("identification") or "").strip() or None,
                evidence_method_verification=str(method_meta.get("verification") or "").strip() or None,
                evidence_singlecell=str(method_meta.get("singlecell") or "").strip() or None,
                evidence_sequencing=str(method_meta.get("sequencing") or "").strip() or None,
                evidence_score=int(score_raw) if str(score_raw).strip().isdigit() else None,
                species=normalize_organism(species_raw),
                antigen_species=normalize_organism(antigen_species_raw) if antigen_species_raw else None,
            )
        except (IndexError, KeyError):
            continue


def parse_mcpas(file_path: Path) -> Iterator[UnifiedRecord]:
    """Parse McPAS-TCR data."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epitope = row.get("Epitope.peptide", "").strip()
                if not epitope or len(epitope) < 5:
                    continue

                mcpas_species = row.get("Species", "").strip() or None
                if not mcpas_species and row.get("MHC", "").strip():
                    mcpas_species = infer_species(row.get("MHC", ""))
                mcpas_pathology = row.get("Pathology", "").strip() or None

                yield UnifiedRecord(
                    peptide=epitope,
                    **_restriction_record_fields(row.get("MHC", ""), None),
                    pmid=normalize_pmid(row.get("PubMed.ID", "")),
                    doi=normalize_doi(row.get("DOI", "")),
                    reference_text=_normalize_reference_text(
                        row.get("Reference", "") or row.get("PubMed.ID", "")
                    ),
                    source="mcpas",
                    record_type="tcr",
                    cdr3_alpha=row.get("CDR3.alpha.aa", ""),
                    cdr3_beta=row.get("CDR3.beta.aa", ""),
                    trav=row.get("TRAV", ""),
                    trbv=row.get("TRBV", ""),
                    evidence_method_identification=row.get("Antigen.identification.method", "") or None,
                    evidence_sequencing=row.get("NGS", "") or None,
                    species=normalize_organism(mcpas_species),
                    antigen_species=normalize_organism(mcpas_pathology),
                )
            except (KeyError, ValueError):
                continue


# =============================================================================
# Cross-Source Deduplication
# =============================================================================


class CrossSourceDeduplicator:
    """Deduplicate records across multiple data sources.

    Strategy:
    1. Load records from multiple sources
    2. Normalize PMIDs and allele names
    3. Group by (peptide, allele, record_type)
    4. Within each group, deduplicate by PMID
    5. When same PMID appears in multiple sources, prefer:
       - Source with more complete data
       - More specific measurements
    """

    def __init__(
        self,
        prefer_sources: Optional[List[str]] = None,
        reference_similarity_threshold: float = 0.92,
    ):
        """Initialize deduplicator.

        Args:
            prefer_sources: Ordered list of sources to prefer when deduping.
                           Default: ['iedb', 'vdjdb', 'mcpas', 'cedar']
        """
        self.prefer_sources = prefer_sources or ["iedb", "vdjdb", "mcpas", "cedar"]
        self._source_order = {source: idx for idx, source in enumerate(self.prefer_sources)}
        self.reference_similarity_threshold = float(reference_similarity_threshold)
        self.stats = {
            "total_input": 0,
            "total_output": 0,
            "input_by_source": defaultdict(int),
            "by_source": defaultdict(int),
            "output_by_assay": defaultdict(int),
            "cross_source_duplicates": 0,
            "same_source_duplicates": 0,
            "fuzzy_reference_duplicates": 0,
            "unique_pmids": 0,
            "records_without_pmid": 0,
            "dedup_groups": 0,
            "dedup_groups_multi_record": 0,
            "dedup_largest_group": 0,
            "dedup_sample_buckets": 0,
        }

    def deduplicate(
        self,
        records: List[UnifiedRecord],
        show_progress: bool = False,
    ) -> List[UnifiedRecord]:
        """Deduplicate a list of unified records.

        Args:
            records: List of UnifiedRecord from multiple sources

        Returns:
            Deduplicated list
        """
        self.stats["total_input"] = len(records)

        # Count input by source
        for rec in records:
            self.stats["input_by_source"][rec.source] += 1

        # Group by dedup key
        groups: Dict[str, List[UnifiedRecord]] = defaultdict(list)
        record_iter: Iterable[UnifiedRecord] = records
        if show_progress:
            record_iter = tqdm(
                records,
                total=len(records),
                desc="Grouping records",
                unit="rec",
                leave=False,
                mininterval=1.0,
                miniters=5000,
            )
        for rec in record_iter:
            groups[rec.dedup_key()].append(rec)

        self.stats["dedup_groups"] = len(groups)
        if groups:
            group_sizes = [len(group_records) for group_records in groups.values()]
            self.stats["dedup_largest_group"] = max(group_sizes)
            self.stats["dedup_groups_multi_record"] = sum(1 for size in group_sizes if size > 1)
        else:
            self.stats["dedup_largest_group"] = 0
            self.stats["dedup_groups_multi_record"] = 0

        # Deduplicate each group
        deduped: List[UnifiedRecord] = []
        group_items = groups.items()
        if show_progress:
            group_items = tqdm(
                group_items,
                total=len(groups),
                desc="Deduplicating groups",
                unit="group",
                leave=False,
                mininterval=1.0,
            )

        sample_bucket_count = 0
        for _, group_records in group_items:
            if len(group_records) == 1:
                deduped.append(group_records[0])
                continue

            sample_buckets: Dict[Tuple[str, ...], List[UnifiedRecord]] = defaultdict(list)
            for rec in group_records:
                sample_buckets[self._sample_signature(rec)].append(rec)

            sample_bucket_count += len(sample_buckets)
            for bucket_records in sample_buckets.values():
                if len(bucket_records) == 1:
                    rec = bucket_records[0]
                    if not rec.pmid:
                        self.stats["records_without_pmid"] += 1
                    deduped.append(rec)
                    continue

                deduped.extend(self._deduplicate_sample_bucket(bucket_records))

        self.stats["dedup_sample_buckets"] = sample_bucket_count

        self.stats["total_output"] = len(deduped)
        self.stats["unique_pmids"] = len(set(r.pmid for r in deduped if r.pmid))
        output_by_source: Dict[str, int] = defaultdict(int)
        output_by_assay: Dict[str, int] = defaultdict(int)
        for rec in deduped:
            output_by_source[rec.source] += 1
            output_by_assay[classify_assay_type(rec)] += 1
        self.stats["by_source"] = output_by_source
        self.stats["output_by_assay"] = output_by_assay

        return deduped

    @staticmethod
    def _sample_signature(rec: UnifiedRecord) -> Tuple[str, ...]:
        """Sample-level identity within a dedup key."""
        if rec.record_type == "binding":
            value_bucket = ""
            if rec.value is not None:
                # Keep distinct quantitative measurements from collapsing together.
                value_bucket = f"{float(rec.value):.6g}"
            # For elution-like ligand rows (value absent), preserve experiment context
            # so different APC/cell settings do not collapse into one sample.
            elution_context = ""
            if rec.value is None:
                elution_context = "|".join(
                    (
                        (rec.assay_type or "").strip().lower(),
                        (rec.assay_method or "").strip().lower(),
                        (rec.apc_name or "").strip().lower(),
                    )
                )
            return (
                rec.record_type,
                (rec.value_type or "").strip().lower(),
                str(rec.qualifier),
                value_bucket,
                elution_context,
            )
        if rec.record_type == "tcell":
            return (
                rec.record_type,
                normalize_binary_response(rec.response) or "unknown",
                (rec.assay_type or "").strip().lower(),
                (rec.assay_method or "").strip().lower(),
                (rec.apc_name or "").strip().lower(),
                (rec.effector_culture_condition or "").strip().lower(),
                (rec.apc_culture_condition or "").strip().lower(),
                (rec.in_vitro_process_type or "").strip().lower(),
                (rec.in_vitro_responder_cell or "").strip().lower(),
                (rec.in_vitro_stimulator_cell or "").strip().lower(),
            )
        if rec.record_type == "bcell":
            return (
                rec.record_type,
                normalize_binary_response(rec.response) or "unknown",
            )
        if rec.record_type == "tcr":
            return (
                rec.record_type,
                (rec.cdr3_alpha or "").strip().upper(),
                (rec.cdr3_beta or "").strip().upper(),
                (rec.trav or "").strip().upper(),
                (rec.trbv or "").strip().upper(),
            )
        return (rec.record_type,)

    @staticmethod
    def _record_payload_signature(rec: UnifiedRecord) -> Tuple[str, ...]:
        """Exact payload signature used for no-reference same-source duplicate collapse."""
        value_bucket = ""
        if rec.value is not None:
            value_bucket = f"{float(rec.value):.10g}"
        return (
            rec.record_type or "",
            (rec.peptide or "").strip().upper(),
            (rec.mhc_allele or "").strip().upper(),
            (rec.mhc_class or "").strip().upper(),
            value_bucket,
            (rec.value_type or "").strip().lower(),
            str(rec.qualifier),
            normalize_binary_response(rec.response) or "",
            (rec.assay_type or "").strip().lower(),
            (rec.assay_method or "").strip().lower(),
            (rec.apc_name or "").strip().lower(),
            (rec.effector_culture_condition or "").strip().lower(),
            (rec.apc_culture_condition or "").strip().lower(),
            (rec.in_vitro_process_type or "").strip().lower(),
            (rec.in_vitro_responder_cell or "").strip().lower(),
            (rec.in_vitro_stimulator_cell or "").strip().lower(),
            (rec.cdr3_alpha or "").strip().upper(),
            (rec.cdr3_beta or "").strip().upper(),
            (rec.trav or "").strip().upper(),
            (rec.trbv or "").strip().upper(),
            (rec.species or "").strip().lower(),
        )

    def _deduplicate_sample_bucket(self, records: List[UnifiedRecord]) -> List[UnifiedRecord]:
        """Fast-path dedup for records that already share dedup+sample identity."""
        selected: List[UnifiedRecord] = []
        pmid_to_indices: Dict[str, List[int]] = defaultdict(list)
        doi_to_indices: Dict[str, List[int]] = defaultdict(list)
        text_ref_indices: List[int] = []
        payload_no_ref_index: Dict[Tuple[str, Tuple[str, ...]], int] = {}

        for rec in records:
            if not rec.pmid:
                self.stats["records_without_pmid"] += 1

            has_ref = bool(rec.pmid or rec.doi or rec.reference_text)
            candidate_indices: List[int] = []

            if rec.pmid:
                candidate_indices.extend(pmid_to_indices.get(rec.pmid, []))
            if rec.doi:
                candidate_indices.extend(doi_to_indices.get(rec.doi, []))

            if not candidate_indices:
                if has_ref:
                    candidate_indices.extend(text_ref_indices)
                else:
                    payload_key = (rec.source, self._record_payload_signature(rec))
                    payload_idx = payload_no_ref_index.get(payload_key)
                    if payload_idx is not None:
                        candidate_indices.append(payload_idx)

            matched_idx: Optional[int] = None
            seen_indices = set()
            for idx in candidate_indices:
                if idx in seen_indices or idx >= len(selected):
                    continue
                seen_indices.add(idx)
                existing = selected[idx]
                if self._records_equivalent_same_sample(rec, existing):
                    matched_idx = idx
                    break

            # Fallback: preserve fuzzy matching behavior when index hints miss.
            if matched_idx is None and has_ref:
                for idx, existing in enumerate(selected):
                    if idx in seen_indices:
                        continue
                    if self._references_match(rec, existing):
                        matched_idx = idx
                        break

            if matched_idx is None:
                selected.append(rec)
                new_idx = len(selected) - 1
                if rec.pmid:
                    pmid_to_indices[rec.pmid].append(new_idx)
                if rec.doi:
                    doi_to_indices[rec.doi].append(new_idx)
                if rec.reference_text:
                    text_ref_indices.append(new_idx)
                if not has_ref:
                    payload_no_ref_index[(rec.source, self._record_payload_signature(rec))] = new_idx
                continue

            existing = selected[matched_idx]
            if rec.source == existing.source:
                self.stats["same_source_duplicates"] += 1
            else:
                self.stats["cross_source_duplicates"] += 1
            if (
                not (rec.pmid and existing.pmid and rec.pmid == existing.pmid)
                and self._references_match(rec, existing)
            ):
                self.stats["fuzzy_reference_duplicates"] += 1

            best = self._select_best([existing, rec])
            selected[matched_idx] = best

            # Refresh lookup maps for the retained row.
            if best.pmid:
                pmid_to_indices[best.pmid].append(matched_idx)
            if best.doi:
                doi_to_indices[best.doi].append(matched_idx)
            if best.reference_text:
                text_ref_indices.append(matched_idx)
            if not (best.pmid or best.doi or best.reference_text):
                payload_no_ref_index[(best.source, self._record_payload_signature(best))] = matched_idx

        return selected

    def _references_match(self, left: UnifiedRecord, right: UnifiedRecord) -> bool:
        """Reference-level match using PMID/DOI first, then fuzzy free text."""
        if left.pmid and right.pmid:
            return left.pmid == right.pmid
        if left.doi and right.doi:
            return left.doi == right.doi
        if left.pmid and right.reference_text and left.pmid in right.reference_text:
            return True
        if right.pmid and left.reference_text and right.pmid in left.reference_text:
            return True

        ref_left = _normalize_reference_text(left.reference_text)
        ref_right = _normalize_reference_text(right.reference_text)
        if not ref_left or not ref_right:
            return False

        # Use both sequence ratio and token-overlap as robust fuzzy checks.
        ratio = SequenceMatcher(None, ref_left, ref_right).ratio()
        if ratio >= self.reference_similarity_threshold:
            return True

        tokens_left = set(ref_left.split())
        tokens_right = set(ref_right.split())
        if not tokens_left or not tokens_right:
            return False
        overlap = len(tokens_left & tokens_right) / float(len(tokens_left | tokens_right))
        return overlap >= 0.85

    def _records_equivalent_same_sample(self, left: UnifiedRecord, right: UnifiedRecord) -> bool:
        """Equivalent check for records already grouped by dedup/sample identity."""
        if self._references_match(left, right):
            return True

        left_has_ref = bool(left.pmid or left.doi or left.reference_text)
        right_has_ref = bool(right.pmid or right.doi or right.reference_text)
        if not left_has_ref and not right_has_ref:
            return (
                left.source == right.source
                and self._record_payload_signature(left) == self._record_payload_signature(right)
            )
        return False

    def _records_equivalent(self, left: UnifiedRecord, right: UnifiedRecord) -> bool:
        """Whether two records should collapse to one canonical row."""
        if left.dedup_key() != right.dedup_key():
            return False
        if self._sample_signature(left) != self._sample_signature(right):
            return False
        return self._records_equivalent_same_sample(left, right)

    def _select_best(self, records: List[UnifiedRecord]) -> UnifiedRecord:
        """Select best record from duplicates."""
        # Prefer records with more data
        def completeness(r: UnifiedRecord) -> int:
            score = 0
            if r.value is not None:
                score += 2
            if r.response:
                score += 1
            if r.cdr3_beta:
                score += 2
            if r.cdr3_alpha:
                score += 1
            if r.trbv:
                score += 1
            return score

        return max(
            records,
            key=lambda r: (
                completeness(r),
                -self._source_order.get(r.source, 999),
            ),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        normalized = dict(self.stats)
        for key in ("input_by_source", "by_source", "output_by_assay"):
            if key in normalized:
                value = normalized[key]
                if isinstance(value, defaultdict):
                    normalized[key] = dict(sorted(value.items(), key=lambda item: (-item[1], item[0])))
                elif isinstance(value, dict):
                    normalized[key] = dict(sorted(value.items(), key=lambda item: (-item[1], item[0])))
        return normalized


def load_all_sources(
    data_dir: Path,
    record_types: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    verbose: bool = False,
) -> Tuple[List[UnifiedRecord], Dict[str, Any]]:
    """Load and normalize records from all available sources.

    Args:
        data_dir: Directory containing downloaded datasets
        record_types: Filter by record types (binding, tcell, tcr)
        sources: Filter by sources (iedb, vdjdb, mcpas, cedar)
        verbose: Print progress

    Returns:
        Tuple of:
        - list of UnifiedRecord from all sources
        - detailed load statistics for verbose reporting
    """
    records: List[UnifiedRecord] = []
    load_stats: Dict[str, Any] = {
        "files": [],
        "by_source": defaultdict(int),
        "by_assay": defaultdict(int),
        "by_record_type": defaultdict(int),
        "by_source_assay": defaultdict(lambda: defaultdict(int)),
    }

    def _sorted_count_dict(counts: Dict[str, int]) -> Dict[str, int]:
        return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))

    def _pick_preferred_file(paths: List[Path]) -> Optional[Path]:
        """Pick one canonical file among multiple export variants for a dataset."""
        if not paths:
            return None

        def _score(path: Path) -> Tuple[int, int, int, int, str]:
            name = path.name.lower()
            return (
                1 if "single_file" in name else 0,
                1 if "full" in name else 0,
                1 if path.suffix.lower() == ".zip" else 0,
                int(path.stat().st_size),
                name,
            )

        return sorted(paths, key=_score, reverse=True)[0]

    def _ingest(path: Path, iterator: Iterator[UnifiedRecord]) -> None:
        file_record_count = 0
        file_source_counts: Dict[str, int] = defaultdict(int)
        file_assay_counts: Dict[str, int] = defaultdict(int)
        file_record_type_counts: Dict[str, int] = defaultdict(int)

        if verbose:
            print(f"  Loading {path.relative_to(data_dir)}...")

        pbar = None
        if verbose:
            pbar = tqdm(
                desc=f"    {path.name}",
                unit="rec",
                leave=False,
                mininterval=1.0,
                miniters=5000,
            )

        pbar_pending = 0
        for rec in iterator:
            records.append(rec)
            assay_type = classify_assay_type(rec)
            source = rec.source or "unknown"
            record_type = rec.record_type or "unknown"

            file_record_count += 1
            file_source_counts[source] += 1
            file_assay_counts[assay_type] += 1
            file_record_type_counts[record_type] += 1

            load_stats["by_source"][source] += 1
            load_stats["by_assay"][assay_type] += 1
            load_stats["by_record_type"][record_type] += 1
            load_stats["by_source_assay"][source][assay_type] += 1

            if pbar is not None:
                pbar_pending += 1
                if pbar_pending >= 5000:
                    pbar.update(pbar_pending)
                    pbar_pending = 0

        if pbar is not None:
            if pbar_pending:
                pbar.update(pbar_pending)
            pbar.close()

        file_entry = {
            "file": str(path),
            "records": file_record_count,
            "by_source": _sorted_count_dict(file_source_counts),
            "by_assay": _sorted_count_dict(file_assay_counts),
            "by_record_type": _sorted_count_dict(file_record_type_counts),
        }
        load_stats["files"].append(file_entry)

        if verbose:
            if file_record_count == 0:
                print("    Loaded 0 records")
            else:
                source_summary = ", ".join(
                    f"{name}:{count}" for name, count in file_entry["by_source"].items()
                )
                assay_summary = ", ".join(
                    f"{name}:{count}" for name, count in file_entry["by_assay"].items()
                )
                print(f"    Loaded {file_record_count} records ({source_summary})")
                print(f"    Assays: {assay_summary}")

    # Determine which sources to load based on record types
    load_binding = record_types is None or "binding" in record_types
    load_tcell = record_types is None or "tcell" in record_types
    load_bcell = record_types is None or "bcell" in record_types
    load_tcr = record_types is None or "tcr" in record_types

    # Source filtering
    load_iedb = sources is None or "iedb" in sources
    load_cedar = sources is None or "cedar" in sources
    load_vdjdb = sources is None or "vdjdb" in sources
    load_mcpas = sources is None or "mcpas" in sources

    iedb_dir = data_dir / "iedb"

    # IEDB binding (use one canonical export variant to avoid same-source duplicates)
    if load_binding and load_iedb and iedb_dir.exists():
        binding_candidates = [
            path
            for pattern in ("mhc_ligand*.csv", "mhc_ligand*.zip")
            for path in iedb_dir.glob(pattern)
            if "cedar" not in path.name.lower()
        ]
        preferred_binding = _pick_preferred_file(binding_candidates)
        if preferred_binding is not None:
            _ingest(preferred_binding, parse_iedb_binding(preferred_binding))

    # IEDB T-cell
    if load_tcell and load_iedb and iedb_dir.exists():
        tcell_candidates = [
            path
            for pattern in ("tcell*.csv", "tcell*.zip")
            for path in iedb_dir.glob(pattern)
            if "cedar" not in path.name.lower()
        ]
        preferred_tcell = _pick_preferred_file(tcell_candidates)
        if preferred_tcell is not None:
            _ingest(preferred_tcell, parse_iedb_tcell(preferred_tcell))

    # CEDAR (same format as IEDB, one canonical file per assay family)
    if load_cedar and iedb_dir.exists():
        if load_tcell:
            cedar_tcell_candidates = [
                path
                for pattern in ("cedar*tcell*.csv", "cedar*tcell*.zip")
                for path in iedb_dir.glob(pattern)
            ]
            preferred_cedar_tcell = _pick_preferred_file(cedar_tcell_candidates)
            if preferred_cedar_tcell is not None:
                _ingest(preferred_cedar_tcell, parse_iedb_tcell(preferred_cedar_tcell))

        if load_bcell:
            cedar_bcell_candidates = [
                path
                for pattern in ("cedar*bcell*.csv", "cedar*bcell*.zip")
                for path in iedb_dir.glob(pattern)
            ]
            preferred_cedar_bcell = _pick_preferred_file(cedar_bcell_candidates)
            if preferred_cedar_bcell is not None:
                _ingest(preferred_cedar_bcell, parse_iedb_bcell(preferred_cedar_bcell))

        if load_binding:
            cedar_binding_candidates = [
                path
                for pattern in ("cedar*mhc_ligand*.csv", "cedar*mhc_ligand*.zip", "cedar*binding*.csv", "cedar*binding*.zip")
                for path in iedb_dir.glob(pattern)
            ]
            preferred_cedar_binding = _pick_preferred_file(cedar_binding_candidates)
            if preferred_cedar_binding is not None:
                _ingest(preferred_cedar_binding, parse_iedb_binding(preferred_cedar_binding))

    # IEDB B-cell
    if load_bcell and load_iedb and iedb_dir.exists():
        bcell_candidates = [
            path
            for pattern in ("bcell*.csv", "bcell*.zip")
            for path in iedb_dir.glob(pattern)
            if "cedar" not in path.name.lower()
        ]
        preferred_bcell = _pick_preferred_file(bcell_candidates)
        if preferred_bcell is not None:
            _ingest(preferred_bcell, parse_iedb_bcell(preferred_bcell))

    # VDJdb (TCR data)
    if load_tcr and load_vdjdb:
        vdjdb_dir = data_dir / "vdjdb"
        if vdjdb_dir.exists():
            for path in vdjdb_dir.glob("vdjdb*.zip"):
                _ingest(path, parse_vdjdb(path))
                break  # Only load one file

    # McPAS (TCR data)
    if load_tcr and load_mcpas:
        mcpas_dir = data_dir / "mcpas"
        if mcpas_dir.exists():
            for path in mcpas_dir.glob("*.csv"):
                _ingest(path, parse_mcpas(path))
                break  # Only load one file

    normalized_stats = {
        "files": load_stats["files"],
        "by_source": _sorted_count_dict(load_stats["by_source"]),
        "by_assay": _sorted_count_dict(load_stats["by_assay"]),
        "by_record_type": _sorted_count_dict(load_stats["by_record_type"]),
        "by_source_assay": {
            source: _sorted_count_dict(assay_counts)
            for source, assay_counts in sorted(load_stats["by_source_assay"].items())
        },
        "total_loaded": len(records),
    }
    return records, normalized_stats


def record_to_row(rec: UnifiedRecord) -> Dict[str, Any]:
    """Serialize a unified record to a flat row."""
    return {
        "peptide": rec.peptide,
        "mhc_allele": rec.mhc_allele,
        "mhc_allele_set": rec.mhc_allele_set or "",
        "mhc_allele_provenance": rec.mhc_allele_provenance or "",
        "mhc_allele_bag_size": rec.mhc_allele_bag_size if rec.mhc_allele_bag_size is not None else "",
        "mhc_class": rec.mhc_class,
        "pmid": rec.pmid or "",
        "doi": rec.doi or "",
        "reference_text": rec.reference_text or "",
        "source": rec.source,
        "record_type": rec.record_type,
        "value": rec.value if rec.value is not None else "",
        "value_type": rec.value_type or "",
        "qualifier": rec.qualifier,
        "response": rec.response or "",
        "assay_type": rec.assay_type or "",
        "assay_method": rec.assay_method or "",
        "apc_name": rec.apc_name or "",
        "cell_hla_allele_set": rec.cell_hla_allele_set or "",
        "cell_hla_n_alleles": rec.cell_hla_n_alleles if rec.cell_hla_n_alleles is not None else "",
        "effector_culture_condition": rec.effector_culture_condition or "",
        "apc_culture_condition": rec.apc_culture_condition or "",
        "in_vitro_process_type": rec.in_vitro_process_type or "",
        "in_vitro_responder_cell": rec.in_vitro_responder_cell or "",
        "in_vitro_stimulator_cell": rec.in_vitro_stimulator_cell or "",
        "cdr3_alpha": rec.cdr3_alpha or "",
        "cdr3_beta": rec.cdr3_beta or "",
        "trav": rec.trav or "",
        "trbv": rec.trbv or "",
        "evidence_method_identification": rec.evidence_method_identification or "",
        "evidence_method_verification": rec.evidence_method_verification or "",
        "evidence_singlecell": rec.evidence_singlecell or "",
        "evidence_sequencing": rec.evidence_sequencing or "",
        "evidence_score": rec.evidence_score if rec.evidence_score is not None else "",
        "species": rec.species,
        "antigen_species": rec.antigen_species or "",
    }


def classify_assay_type(rec: UnifiedRecord) -> str:
    """Map a unified record into a simplified assay bucket."""
    if rec._assay_bucket_cache:
        return rec._assay_bucket_cache

    assay = "unknown"
    if rec.record_type == "elution":
        assay = "elution_ms"
    elif rec.record_type == "processing":
        assay = "processing"
    elif rec.record_type == "binding":
        method_text = " ".join(
            token
            for token in (
                (rec.value_type or "").strip().lower(),
                (rec.assay_method or "").strip().lower(),
            )
            if token
        )
        if any(token in method_text for token in ("processing", "cleavage", "tap transport", "tap assay", "erap")):
            assay = "processing"
        elif any(token in method_text for token in ("kon", "on rate", "association rate", "ka")):
            assay = "binding_kon"
        elif any(token in method_text for token in ("koff", "off rate", "dissociation rate")):
            assay = "binding_koff"
        elif any(token in method_text for token in ("t_half", "half life", "half-life", "t1/2")):
            assay = "binding_t_half"
        elif any(token in method_text for token in ("tm", "melt", "dissociation temperature")):
            assay = "binding_tm"
        elif rec.value is None:
            if any(token in method_text for token in ("dia", "data-independent")):
                assay = "elution_ms_dia"
            elif any(token in method_text for token in ("dda", "data-dependent")):
                assay = "elution_ms_dda"
            elif any(token in method_text for token in ("targeted", "prm", "srm", "srm/ms", "mrm")):
                assay = "elution_ms_targeted"
            else:
                assay = "elution_ms"
        else:
            assay = "binding_affinity"
    elif rec.record_type == "tcell":
        assay = "tcell_response"
    elif rec.record_type == "tcr":
        assay = "tcr_evidence"
    elif rec.record_type == "bcell":
        assay = "bcell_response"
    else:
        assay = rec.record_type or "unknown"

    rec._assay_bucket_cache = assay
    return assay


def _normalize_cell_context(text: Optional[str]) -> str:
    """Normalize free-text cell context for lookup joins."""
    if text is None:
        return ""
    return _normalize_cell_context_cached(str(text))


@lru_cache(maxsize=200000)
def _normalize_cell_context_cached(text: str) -> str:
    return _CELL_CONTEXT_WS_RE.sub(" ", text.strip().lower())


def _split_allele_tokens(raw_allele: Optional[str]) -> List[str]:
    """Split potentially multi-allele strings and normalize each token."""
    if raw_allele is None:
        return []
    text = str(raw_allele).strip()
    if not text:
        return []
    return list(_split_allele_tokens_cached(text))


@lru_cache(maxsize=200000)
def _split_allele_tokens_cached(text: str) -> Tuple[str, ...]:
    tokens = [tok.strip() for tok in _ALLELE_SPLIT_RE.split(text) if tok.strip()]
    normalized: List[str] = []
    for token in tokens:
        allele = normalize_allele(token)
        normalized.append(allele or token)
    return tuple(normalized)


def _is_informative_allele_token(allele: str) -> bool:
    """Strict heuristic for allele-like identifiers (not class/generic labels)."""
    token = (allele or "").strip()
    if not token:
        return False
    token_u = _CELL_CONTEXT_WS_RE.sub(" ", token).upper()

    if token_u in {"HLA", "MHC", "UNKNOWN", "N/A", "NA", "OTHER"}:
        return False
    if any(
        banned in token_u
        for banned in ("CLASS", "MHC", "MOLECULE", "MUTANT", "SEROTYPE", "HAPLOTYPE")
    ):
        return False
    if " " in token_u:
        return False

    # Canonical star-style alleles (e.g. HLA-A*02:01, DPB1*04:01, DLA-88*501:01).
    if _STAR_ALLELE_RE.match(token_u):
        return True

    # Accept murine shorthand loci (e.g. H2-Db, H2-Kb, H2-IAg7).
    if _MURINE_ALLELE_SHORTHAND_RE.match(token_u):
        return True

    return False


def _is_elution_assay(assay: str) -> bool:
    return str(assay).startswith("elution_ms")


def _join_allele_set(tokens: Iterable[str]) -> str:
    uniq = sorted({tok for tok in tokens if tok})
    return ";".join(uniq)


def _cell_lookup_key(rec: UnifiedRecord) -> Optional[Tuple[str, str]]:
    pmid = (rec.pmid or "").strip()
    if not pmid:
        return None
    cell_norm = _normalize_cell_context(rec.apc_name)
    if not cell_norm:
        return None
    return pmid, cell_norm


def _resolve_record_cell_hla_set(
    rec: UnifiedRecord,
    lookup: Dict[Tuple[str, str], str],
) -> Tuple[Optional[str], str]:
    """Resolve a record's allele-set from direct row alleles or strict PMID+cell lookup."""
    direct_tokens = tuple(parse_allele_set_field(rec.mhc_allele_set))
    if not direct_tokens:
        direct_tokens = _informative_allele_tokens(rec.mhc_allele)
    if direct_tokens:
        return _join_allele_set(direct_tokens), "direct"

    key = _cell_lookup_key(rec)
    if key is not None:
        allele_set = lookup.get(key)
        if allele_set:
            return allele_set, "lookup"
        return None, "lookup_miss"
    return None, "missing_lookup_key"


def _count_records_by_assay(records: List[UnifiedRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for rec in records:
        counts[classify_assay_type(rec)] += 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


@lru_cache(maxsize=200000)
def _informative_allele_tokens(raw_allele: Optional[str]) -> Tuple[str, ...]:
    if raw_allele is None:
        return tuple()
    allele_text = str(raw_allele).strip()
    if not allele_text:
        return tuple()
    tokens = [
        token
        for token in _split_allele_tokens_cached(allele_text)
        if _is_informative_allele_token(token)
    ]
    if not tokens:
        return tuple()
    # Keep deterministic output while preserving first-seen order.
    return tuple(dict.fromkeys(tokens))


def _build_elution_cell_hla_lookup(
    records: List[UnifiedRecord],
    show_progress: bool = False,
) -> Tuple[Dict[Tuple[str, str], str], Dict[str, Any]]:
    """Build strict (pmid, normalized-cell-context) -> allele-set map from elution rows."""
    allele_sets: Dict[Tuple[str, str], set[str]] = defaultdict(set)
    stats: Counter[str] = Counter()
    record_iter: Iterable[UnifiedRecord] = records
    if show_progress:
        record_iter = tqdm(
            records,
            total=len(records),
            desc="Cell-HLA lookup",
            unit="rec",
            leave=False,
            mininterval=1.0,
            miniters=5000,
        )

    for rec in record_iter:
        assay = classify_assay_type(rec)
        if not _is_elution_assay(assay):
            continue
        stats["elution_rows_total"] += 1

        # IEDB ligand rows are predominantly positives; skip explicit negatives only.
        response = normalize_binary_response(rec.response)
        if response == "negative":
            stats["elution_rows_negative"] += 1
            continue

        lookup_key = _cell_lookup_key(rec)
        if lookup_key is None:
            if not rec.pmid:
                stats["elution_rows_missing_pmid"] += 1
            if not _normalize_cell_context(rec.apc_name):
                stats["elution_rows_missing_cell_context"] += 1
            continue
        stats["elution_rows_with_cell_context"] += 1
        stats["elution_rows_with_pmid"] += 1

        allele_tokens = _informative_allele_tokens(rec.mhc_allele)
        if not allele_tokens:
            stats["elution_rows_missing_informative_allele"] += 1
            continue
        stats["elution_rows_with_informative_allele"] += 1

        for token in allele_tokens:
            allele_sets[lookup_key].add(token)

    lookup = {
        key: _join_allele_set(tokens)
        for key, tokens in allele_sets.items()
        if tokens
    }
    stats["pmid_cell_context_pairs_with_allele_set"] = len(lookup)
    return lookup, dict(stats)


def _annotate_cell_hla_sets(
    records: List[UnifiedRecord],
    lookup: Dict[Tuple[str, str], str],
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Annotate cellular-assay records with inferred presenting-cell allele sets."""
    stats: Counter[str] = Counter()
    cellular_assays = {"tcell_response", "bcell_response"}

    record_iter: Iterable[UnifiedRecord] = records
    if show_progress:
        record_iter = tqdm(
            records,
            total=len(records),
            desc="Cell-HLA annotate",
            unit="rec",
            leave=False,
            mininterval=1.0,
            miniters=5000,
        )

    for rec in record_iter:
        assay = classify_assay_type(rec)
        is_cellular = _is_elution_assay(assay) or assay in cellular_assays
        if not is_cellular:
            continue
        stats["cellular_records_total"] += 1

        lookup_key = _cell_lookup_key(rec)
        if lookup_key is None:
            if not rec.pmid:
                stats["cellular_records_missing_pmid"] += 1
            if not _normalize_cell_context(rec.apc_name):
                stats["cellular_records_missing_cell_context"] += 1
        else:
            stats["cellular_records_with_lookup_key"] += 1

        allele_set, source = _resolve_record_cell_hla_set(rec, lookup)
        if not allele_set:
            stats["cellular_records_without_allele_set"] += 1
            continue
        rec.cell_hla_allele_set = allele_set
        rec.cell_hla_n_alleles = len([tok for tok in allele_set.split(";") if tok])
        stats["cellular_records_with_allele_set"] += 1
        if source == "direct":
            stats["cellular_records_with_direct_allele_set"] += 1
        elif source == "lookup":
            stats["cellular_records_with_lookup_allele_set"] += 1
        if _is_elution_assay(assay):
            stats["elution_records_with_allele_set"] += 1

    return dict(stats)


def _filter_elution_without_cell_hla(
    records: List[UnifiedRecord],
    show_progress: bool = False,
) -> Tuple[List[UnifiedRecord], Dict[str, Any]]:
    """Drop elution rows that cannot be tied to a resolvable cell allele set."""
    kept: List[UnifiedRecord] = []
    stats: Counter[str] = Counter()
    record_iter: Iterable[UnifiedRecord] = records
    if show_progress:
        record_iter = tqdm(
            records,
            total=len(records),
            desc="Elution HLA filter",
            unit="rec",
            leave=False,
            mininterval=1.0,
            miniters=5000,
        )

    for rec in record_iter:
        assay = classify_assay_type(rec)
        if not _is_elution_assay(assay):
            kept.append(rec)
            continue
        stats["elution_rows_total"] += 1
        if rec.cell_hla_allele_set:
            stats["elution_rows_kept_with_allele_set"] += 1
            kept.append(rec)
        else:
            stats["elution_rows_dropped_missing_allele_set"] += 1
    if stats["elution_rows_total"] > 0:
        stats["elution_rows_with_allele_set_fraction"] = float(
            stats["elution_rows_kept_with_allele_set"] / float(stats["elution_rows_total"])
        )
    else:
        stats["elution_rows_with_allele_set_fraction"] = 0.0
    stats["output_total_after_elution_filter"] = len(kept)
    return kept, dict(stats)


def _annotate_and_filter_cell_hla(
    records: List[UnifiedRecord],
    lookup: Dict[Tuple[str, str], str],
    show_progress: bool = False,
) -> Tuple[List[UnifiedRecord], Dict[str, Any], Dict[str, Any]]:
    """Single-pass cellular annotation + elution filtering for faster merge runs."""
    kept: List[UnifiedRecord] = []
    annotate_stats: Counter[str] = Counter()
    filter_stats: Counter[str] = Counter()
    cellular_assays = {"tcell_response", "bcell_response"}

    record_iter: Iterable[UnifiedRecord] = records
    if show_progress:
        record_iter = tqdm(
            records,
            total=len(records),
            desc="Cell-HLA annotate+filter",
            unit="rec",
            leave=False,
            mininterval=1.0,
            miniters=5000,
        )

    for rec in record_iter:
        assay = classify_assay_type(rec)
        is_elution = _is_elution_assay(assay)
        is_cellular = is_elution or assay in cellular_assays

        if is_cellular:
            annotate_stats["cellular_records_total"] += 1
            lookup_key = _cell_lookup_key(rec)
            if lookup_key is None:
                if not rec.pmid:
                    annotate_stats["cellular_records_missing_pmid"] += 1
                if not _normalize_cell_context(rec.apc_name):
                    annotate_stats["cellular_records_missing_cell_context"] += 1
            else:
                annotate_stats["cellular_records_with_lookup_key"] += 1

            allele_set, source = _resolve_record_cell_hla_set(rec, lookup)
            if not allele_set:
                annotate_stats["cellular_records_without_allele_set"] += 1
            else:
                rec.cell_hla_allele_set = allele_set
                rec.cell_hla_n_alleles = len([tok for tok in allele_set.split(";") if tok])
                annotate_stats["cellular_records_with_allele_set"] += 1
                if source == "direct":
                    annotate_stats["cellular_records_with_direct_allele_set"] += 1
                elif source == "lookup":
                    annotate_stats["cellular_records_with_lookup_allele_set"] += 1
                if is_elution:
                    annotate_stats["elution_records_with_allele_set"] += 1

            if is_elution:
                filter_stats["elution_rows_total"] += 1
                if rec.cell_hla_allele_set:
                    filter_stats["elution_rows_kept_with_allele_set"] += 1
                    kept.append(rec)
                else:
                    filter_stats["elution_rows_dropped_missing_allele_set"] += 1
            else:
                filter_stats["cellular_rows_total_non_elution"] += 1
                if rec.cell_hla_allele_set:
                    filter_stats["cellular_rows_kept_with_allele_set"] += 1
                    kept.append(rec)
                else:
                    filter_stats["cellular_rows_dropped_missing_allele_set"] += 1
            continue

        # Non-cellular assays are kept unchanged.
        kept.append(rec)
        continue

    if filter_stats["elution_rows_total"] > 0:
        filter_stats["elution_rows_with_allele_set_fraction"] = float(
            filter_stats["elution_rows_kept_with_allele_set"] / float(filter_stats["elution_rows_total"])
        )
    else:
        filter_stats["elution_rows_with_allele_set_fraction"] = 0.0

    if filter_stats["cellular_rows_total_non_elution"] > 0:
        filter_stats["cellular_rows_with_allele_set_fraction"] = float(
            filter_stats["cellular_rows_kept_with_allele_set"]
            / float(filter_stats["cellular_rows_total_non_elution"])
        )
    else:
        filter_stats["cellular_rows_with_allele_set_fraction"] = 0.0

    filter_stats["output_total_after_elution_filter"] = len(kept)

    return kept, dict(annotate_stats), dict(filter_stats)


def _write_merge_funnel_artifacts(
    output_path: Path,
    stage_counts: List[Tuple[str, Dict[str, int]]],
) -> Dict[str, str]:
    """Write per-assay funnel counts and, when possible, a PNG visualization."""
    out_dir = output_path.parent
    stem = output_path.stem
    tsv_path = out_dir / f"{stem}_funnel.tsv"
    png_path = out_dir / f"{stem}_funnel.png"

    assay_names = sorted(
        {
            assay
            for _, counts in stage_counts
            for assay in counts.keys()
        }
    )

    # Also include aggregate "all" rows to show total funnel behavior.
    with open(tsv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
                "assay",
                "count",
                "fraction_of_previous_stage",
                "fraction_of_loaded_stage",
            ],
            delimiter="\t",
        )
        writer.writeheader()

        loaded_lookup = dict(stage_counts[0][1]) if stage_counts else {}
        prev_lookup: Optional[Dict[str, int]] = None
        for stage, counts in stage_counts:
            all_count = int(sum(counts.values()))
            all_prev = int(sum(prev_lookup.values())) if prev_lookup is not None else all_count
            all_loaded = int(sum(loaded_lookup.values())) if loaded_lookup else all_count
            writer.writerow(
                {
                    "stage": stage,
                    "assay": "all",
                    "count": all_count,
                    "fraction_of_previous_stage": (
                        (all_count / float(all_prev)) if all_prev > 0 else 0.0
                    ),
                    "fraction_of_loaded_stage": (
                        (all_count / float(all_loaded)) if all_loaded > 0 else 0.0
                    ),
                }
            )

            for assay in assay_names:
                count = int(counts.get(assay, 0))
                prev_count = int(prev_lookup.get(assay, 0)) if prev_lookup is not None else count
                loaded_count = int(loaded_lookup.get(assay, 0))
                writer.writerow(
                    {
                        "stage": stage,
                        "assay": assay,
                        "count": count,
                        "fraction_of_previous_stage": (
                            (count / float(prev_count)) if prev_count > 0 else 0.0
                        ),
                        "fraction_of_loaded_stage": (
                            (count / float(loaded_count)) if loaded_count > 0 else 0.0
                        ),
                    }
                )
            prev_lookup = dict(counts)

    files = {"tsv": str(tsv_path)}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return files

    stages = [name for name, _ in stage_counts]
    totals = [int(sum(counts.values())) for _, counts in stage_counts]
    loaded_counts = stage_counts[0][1] if stage_counts else {}
    top_assays = [
        assay
        for assay, _ in sorted(
            loaded_counts.items(),
            key=lambda item: (-int(item[1]), item[0]),
        )[:8]
    ]

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

    ax_top.plot(stages, totals, marker="o", linewidth=2.2, color="#1f77b4")
    ax_top.set_title("Unified Merge Funnel (All Assays)")
    ax_top.set_ylabel("Records")
    ax_top.grid(alpha=0.25)

    for assay in top_assays:
        series = [int(counts.get(assay, 0)) for _, counts in stage_counts]
        ax_bottom.plot(stages, series, marker="o", linewidth=1.6, label=assay)
    ax_bottom.set_title("Top Assays Through Merge Funnel")
    ax_bottom.set_ylabel("Records")
    ax_bottom.grid(alpha=0.25)
    if top_assays:
        ax_bottom.legend(fontsize=8, ncol=2)

    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    files["png"] = str(png_path)
    return files


def _write_tsv(records: List[UnifiedRecord], output_path: Path) -> None:
    fieldnames = [
        "peptide",
        "mhc_allele",
        "mhc_allele_set",
        "mhc_allele_provenance",
        "mhc_allele_bag_size",
        "mhc_class",
        "pmid",
        "doi",
        "reference_text",
        "source",
        "record_type",
        "value",
        "value_type",
        "qualifier",
        "response",
        "assay_type",
        "assay_method",
        "apc_name",
        "cell_hla_allele_set",
        "cell_hla_n_alleles",
        "effector_culture_condition",
        "apc_culture_condition",
        "in_vitro_process_type",
        "in_vitro_responder_cell",
        "in_vitro_stimulator_cell",
        "cdr3_alpha",
        "cdr3_beta",
        "trav",
        "trbv",
        "species",
        "antigen_species",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(fieldnames)
        for rec in records:
            writer.writerow(
                [
                    rec.peptide,
                    rec.mhc_allele,
                    rec.mhc_allele_set or "",
                    rec.mhc_allele_provenance or "",
                    rec.mhc_allele_bag_size if rec.mhc_allele_bag_size is not None else "",
                    rec.mhc_class,
                    rec.pmid or "",
                    rec.doi or "",
                    rec.reference_text or "",
                    rec.source,
                    rec.record_type,
                    rec.value if rec.value is not None else "",
                    rec.value_type or "",
                    rec.qualifier,
                    rec.response or "",
                    rec.assay_type or "",
                    rec.assay_method or "",
                    rec.apc_name or "",
                    rec.cell_hla_allele_set or "",
                    rec.cell_hla_n_alleles if rec.cell_hla_n_alleles is not None else "",
                    rec.effector_culture_condition or "",
                    rec.apc_culture_condition or "",
                    rec.in_vitro_process_type or "",
                    rec.in_vitro_responder_cell or "",
                    rec.in_vitro_stimulator_cell or "",
                    rec.cdr3_alpha or "",
                    rec.cdr3_beta or "",
                    rec.trav or "",
                    rec.trbv or "",
                    rec.species or "",
                    rec.antigen_species or "",
                ]
            )


def write_assay_csvs(
    records: List[UnifiedRecord],
    output_dir: Path,
) -> Dict[str, str]:
    """Write one simplified CSV per assay bucket."""
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: Dict[str, List[UnifiedRecord]] = defaultdict(list)
    for rec in records:
        grouped[classify_assay_type(rec)].append(rec)

    fieldnames = [
        "peptide",
        "mhc_allele",
        "mhc_allele_set",
        "mhc_allele_provenance",
        "mhc_allele_bag_size",
        "mhc_class",
        "species",
        "antigen_species",
        "source",
        "pmid",
        "doi",
        "reference_text",
        "record_type",
        "value",
        "value_type",
        "qualifier",
        "response",
        "assay_type",
        "assay_method",
        "apc_name",
        "cell_hla_allele_set",
        "cell_hla_n_alleles",
        "effector_culture_condition",
        "apc_culture_condition",
        "in_vitro_process_type",
        "in_vitro_responder_cell",
        "in_vitro_stimulator_cell",
        "cdr3_alpha",
        "cdr3_beta",
        "trav",
        "trbv",
        "evidence_method_identification",
        "evidence_method_verification",
        "evidence_singlecell",
        "evidence_sequencing",
        "evidence_score",
    ]
    file_map: Dict[str, str] = {}
    for assay_type, assay_records in sorted(grouped.items()):
        out_path = output_dir / f"{assay_type}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in assay_records:
                writer.writerow(record_to_row(rec))
        file_map[assay_type] = str(out_path)
    return file_map


def deduplicate_all(
    data_dir: Path,
    output_path: Optional[Path] = None,
    assay_output_dir: Optional[Path] = None,
    record_types: Optional[List[str]] = None,
    sources: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[List[UnifiedRecord], Dict[str, Any]]:
    """Load all sources and deduplicate.

    Args:
        data_dir: Directory with downloaded data
        output_path: Optional path to save deduplicated data
        record_types: Filter by record types (binding, tcell, tcr)
        sources: Filter by sources (iedb, vdjdb, mcpas, cedar)
        verbose: Print progress

    Returns:
        Tuple of (deduplicated records, statistics)
    """
    run_start = time.perf_counter()
    timing_sec: Dict[str, float] = {}

    def _elapsed(start: float) -> float:
        return float(time.perf_counter() - start)

    def _fmt_rate(count: int, elapsed_sec: float) -> str:
        if elapsed_sec <= 0.0:
            return "n/a"
        return f"{count / elapsed_sec:,.1f}/s"

    if verbose:
        print("Loading records from all sources...")

    load_start = time.perf_counter()
    records, load_stats = load_all_sources(
        data_dir, record_types=record_types, sources=sources, verbose=verbose
    )
    timing_sec["load"] = _elapsed(load_start)
    stats: Dict[str, Any] = {
        "load": load_stats,
        "timing_sec": timing_sec,
    }

    if verbose:
        print(
            f"  Loaded {len(records)} total records in {timing_sec['load']:.2f}s "
            f"({_fmt_rate(len(records), timing_sec['load'])})"
        )
        if load_stats["by_source"]:
            print("  Source totals:")
            for source, count in load_stats["by_source"].items():
                print(f"    {source}: {count}")
        if load_stats["by_assay"]:
            print("  Assay totals:")
            for assay, count in load_stats["by_assay"].items():
                print(f"    {assay}: {count}")
        if load_stats["by_source_assay"]:
            print("  Source x assay breakdown:")
            for source, assay_counts in load_stats["by_source_assay"].items():
                summary = ", ".join(f"{assay}:{count}" for assay, count in assay_counts.items())
                print(f"    {source}: {summary}")

    # Deduplicate
    if verbose:
        print("Deduplicating...")

    dedup_start = time.perf_counter()
    deduper = CrossSourceDeduplicator()
    deduped_pre_filter = deduper.deduplicate(records, show_progress=verbose)
    timing_sec["dedup"] = _elapsed(dedup_start)
    stats.update(deduper.get_stats())

    deduped_by_assay = _count_records_by_assay(deduped_pre_filter)
    stats["deduped_by_assay"] = deduped_by_assay

    if verbose:
        print(
            f"  Dedup stage: {len(records)} -> {len(deduped_pre_filter)} in "
            f"{timing_sec['dedup']:.2f}s ({_fmt_rate(len(records), timing_sec['dedup'])})"
        )
        print(
            "  Dedup groups: "
            f"{stats.get('dedup_groups', 0)} total, "
            f"{stats.get('dedup_groups_multi_record', 0)} multi-record, "
            f"largest={stats.get('dedup_largest_group', 0)}, "
            f"sample-buckets={stats.get('dedup_sample_buckets', 0)}"
        )

    if verbose:
        print("Building elution cell-HLA lookup...")
    cell_hla_lookup_start = time.perf_counter()
    cell_hla_lookup, cell_hla_lookup_stats = _build_elution_cell_hla_lookup(
        records,
        show_progress=verbose,
    )
    timing_sec["cell_hla_lookup"] = _elapsed(cell_hla_lookup_start)
    stats["cell_hla_lookup"] = {
        **cell_hla_lookup_stats,
        "pmid_cell_context_pairs_with_allele_set": len(cell_hla_lookup),
        "cell_contexts_with_allele_set": len(cell_hla_lookup),  # backward-compatible key
    }
    if verbose:
        print(
            f"  Cell-HLA lookup built for {len(cell_hla_lookup)} PMID+cell pairs in "
            f"{timing_sec['cell_hla_lookup']:.2f}s"
        )

    if verbose:
        print("Annotating cellular rows and filtering elution rows...")
    annotate_filter_start = time.perf_counter()
    deduped, cell_hla_annot_stats, elution_filter_stats = _annotate_and_filter_cell_hla(
        deduped_pre_filter,
        cell_hla_lookup,
        show_progress=verbose,
    )
    timing_sec["cell_hla_annotate_filter"] = _elapsed(annotate_filter_start)
    stats["cell_hla_annotation"] = cell_hla_annot_stats
    stats["elution_cell_hla_filter"] = elution_filter_stats

    # Final output reflects the post-annotation elution filter.
    stats["total_output_pre_elution_cell_hla_filter"] = int(
        stats.get("total_output", len(deduped_pre_filter))
    )
    stats["total_output"] = len(deduped)
    stats["final_output_by_assay"] = _count_records_by_assay(deduped)

    if verbose:
        print(
            f"  Cell-HLA annotate/filter completed in {timing_sec['cell_hla_annotate_filter']:.2f}s "
            f"({_fmt_rate(len(deduped_pre_filter), timing_sec['cell_hla_annotate_filter'])})"
        )

    stage_counts: List[Tuple[str, Dict[str, int]]] = [
        ("loaded", dict(load_stats.get("by_assay", {}))),
        ("deduped", deduped_by_assay),
        ("final", dict(stats["final_output_by_assay"])),
    ]
    stats["funnel_stage_counts"] = stage_counts

    if verbose:
        print(f"  Input: {stats['total_input']}")
        print(f"  Output after dedup: {stats['total_output_pre_elution_cell_hla_filter']}")
        print(f"  Output after elution cell-HLA filter: {stats['total_output']}")
        print(f"  Cross-source duplicates removed: {stats['cross_source_duplicates']}")
        print(f"  Same-source duplicates removed: {stats['same_source_duplicates']}")
        print(f"  Unique PMIDs: {stats['unique_pmids']}")
        input_by_source = stats.get("input_by_source", {})
        output_by_source = stats.get("by_source", {})
        output_by_assay = stats.get("output_by_assay", {})
        final_output_by_assay = stats.get("final_output_by_assay", {})

        if input_by_source:
            print("  Input records by source:")
            for source, count in input_by_source.items():
                print(f"    {source}: {count}")
        if output_by_source:
            print("  Output records by source:")
            for source, count in output_by_source.items():
                print(f"    {source}: {count}")

        if input_by_source and output_by_source:
            print("  Source retention:")
            for source, input_count in input_by_source.items():
                kept = int(output_by_source.get(source, 0))
                retention = (kept / float(input_count)) if input_count else 0.0
                pct = retention * 100.0
                warning = " [WARN: >90% dropped]" if input_count >= 100 and pct < 10.0 else ""
                print(f"    {source}: kept {kept}/{input_count} ({pct:.2f}%){warning}")

        if output_by_assay:
            print("  Output records by assay (post-dedup):")
            for assay, count in output_by_assay.items():
                print(f"    {assay}: {count}")
        if final_output_by_assay:
            print("  Output records by assay (final):")
            for assay, count in final_output_by_assay.items():
                print(f"    {assay}: {count}")

        input_by_assay = load_stats.get("by_assay", {})
        if input_by_assay and final_output_by_assay:
            print("  Assay retention:")
            for assay, input_count in input_by_assay.items():
                kept = int(final_output_by_assay.get(assay, 0))
                retention = (kept / float(input_count)) if input_count else 0.0
                pct = retention * 100.0
                warning = " [WARN: >90% dropped]" if input_count >= 100 and pct < 10.0 else ""
                print(f"    {assay}: kept {kept}/{input_count} ({pct:.2f}%){warning}")

        ms_total = int(elution_filter_stats.get("elution_rows_total", 0))
        ms_with = int(elution_filter_stats.get("elution_rows_kept_with_allele_set", 0))
        ms_frac = float(elution_filter_stats.get("elution_rows_with_allele_set_fraction", 0.0))
        print(
            "  Elution cell-HLA coverage: "
            f"{ms_with}/{ms_total} ({ms_frac * 100.0:.2f}%) rows retained"
        )

    # Save full merged table if requested.
    if output_path:
        if verbose:
            print(f"Saving to {output_path}...")
        write_tsv_start = time.perf_counter()
        _write_tsv(deduped, output_path)
        timing_sec["write_tsv"] = _elapsed(write_tsv_start)

        if verbose:
            print(f"  Saved {len(deduped)} records in {timing_sec['write_tsv']:.2f}s")

    if assay_output_dir is not None:
        if verbose:
            print(f"Writing assay CSVs to {assay_output_dir}...")
        write_assays_start = time.perf_counter()
        file_map = write_assay_csvs(deduped, assay_output_dir)
        timing_sec["write_assay_csvs"] = _elapsed(write_assays_start)
        stats["assay_csv_files"] = file_map
        if verbose:
            print(
                f"  Wrote {len(file_map)} assay CSV files in "
                f"{timing_sec['write_assay_csvs']:.2f}s"
            )

    funnel_anchor = output_path if output_path is not None else (data_dir / "merged_deduped.tsv")
    funnel_start = time.perf_counter()
    funnel_files = _write_merge_funnel_artifacts(
        output_path=funnel_anchor,
        stage_counts=stage_counts,
    )
    timing_sec["write_funnel"] = _elapsed(funnel_start)
    stats["funnel_files"] = funnel_files
    if verbose:
        funnel_text = ", ".join(f"{k}={v}" for k, v in funnel_files.items())
        print(f"Funnel artifacts: {funnel_text}")

    timing_sec["total"] = _elapsed(run_start)
    if verbose:
        print("Stage timings (s):")
        for key in (
            "load",
            "dedup",
            "cell_hla_lookup",
            "cell_hla_annotate_filter",
            "write_tsv",
            "write_assay_csvs",
            "write_funnel",
            "total",
        ):
            if key in timing_sec:
                print(f"  {key}: {timing_sec[key]:.2f}")

    return deduped, stats
