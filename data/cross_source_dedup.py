"""Cross-source deduplication for immunology datasets.

Handles merging and deduplicating data from multiple sources (IEDB, CEDAR,
VDJdb, McPAS, etc.) based on publication references (PubMed IDs).

Each source has different file formats but often cites the same papers.
This module normalizes the data and removes duplicate entries that appear
in multiple databases from the same original publication.
"""

import csv
import io
import re
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple, Any
import zipfile
from tqdm.auto import tqdm

from .allele_resolver import infer_species
from .vocab import normalize_organism

try:
    import mhcgnomes

    _HAS_MHCGNOMES = True
except ImportError:
    _HAS_MHCGNOMES = False


@dataclass
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

    # Metadata
    species: Optional[str] = None
    antigen_species: Optional[str] = None
    raw_data: Dict[str, str] = field(default_factory=dict)

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


@lru_cache(maxsize=50000)
def normalize_allele(raw_allele: str) -> str:
    """Normalize MHC allele to standard format using mhcgnomes.

    Uses mhcgnomes library for robust parsing of MHC allele names across
    species and naming conventions. Falls back to simple normalization
    if mhcgnomes is not available.

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

    if _HAS_MHCGNOMES:
        try:
            result = mhcgnomes.parse(allele)
            return result.to_string() if result else allele
        except Exception:
            # mhcgnomes couldn't parse - return as-is
            return allele

    # Fallback: simple normalization without mhcgnomes
    return _normalize_allele_simple(allele)


def _normalize_allele_simple(allele: str) -> str:
    """Simple allele normalization without mhcgnomes."""
    # Already in good format (has HLA-, *, and colon)
    if allele.startswith("HLA-") and "*" in allele and ":" in allele:
        return allele

    # Add HLA prefix if missing
    if not allele.startswith("HLA"):
        if allele.startswith(("A*", "B*", "C*", "DR", "DQ", "DP")):
            allele = "HLA-" + allele

    # Normalize A2 -> A*02 format
    match = re.match(r"HLA-([A-Z]+)(\d+)$", allele)
    if match:
        gene, num = match.groups()
        allele = f"HLA-{gene}*{num.zfill(2)}"

    # Normalize 0201 -> 02:01 format
    match = re.match(r"(HLA-[A-Z]+\*)(\d{4})$", allele)
    if match:
        prefix, digits = match.groups()
        allele = f"{prefix}{digits[:2]}:{digits[2:]}"

    return allele


def _infer_mhc_class_from_allele(allele: str, default: str = "I") -> str:
    """Infer MHC class from allele text using common naming patterns."""
    text = (allele or "").strip().upper()
    if not text:
        return default
    if any(token in text for token in ("DR", "DQ", "DP", "H2-I", "H2-E", "RT1-B", "RT1-D")):
        return "II"
    if any(
        text.startswith(prefix)
        for prefix in (
            "HLA-A",
            "HLA-B",
            "HLA-C",
            "HLA-E",
            "HLA-F",
            "HLA-G",
            "H2-K",
            "H2-D",
            "MAMU-A",
            "MAMU-B",
            "MAMU-E",
            "MAMU-I",
        )
    ):
        return "I"
    return default


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

            pmid_raw = row[col_indices["pmid"]].strip() if "pmid" in col_indices else ""
            doi_raw = row[col_indices["doi"]].strip() if "doi" in col_indices else ""
            reference_parts = [
                row[idx].strip()
                for idx in reference_text_cols
                if idx < len(row) and row[idx].strip()
            ]
            reference_raw = " | ".join(dict.fromkeys(reference_parts))

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
                mhc_allele=normalize_allele(allele),
                mhc_class=_normalize_mhc_class(mhc_class, allele),
                pmid=normalize_pmid(pmid_raw),
                doi=normalize_doi(doi_raw),
                reference_text=_normalize_reference_text(reference_raw),
                source=source,
                record_type="binding",
                value=value,
                value_type=response_measured or method or "IC50",
                qualifier=qualifier,
                response=qualitative or None,
                assay_type=response_measured or None,
                assay_method=method or None,
                species=normalize_organism(species_raw),
                antigen_species=normalize_organism(antigen_species_raw) if antigen_species_raw else None,
                raw_data={
                    "reference_raw": reference_raw,
                    "doi_raw": doi_raw,
                },
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
            reference_parts = [
                row[idx].strip()
                for idx in reference_text_cols
                if idx < len(row) and row[idx].strip()
            ]
            reference_raw = " | ".join(dict.fromkeys(reference_parts))

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
                pmid=normalize_pmid(pmid_raw),
                doi=normalize_doi(doi_raw),
                reference_text=_normalize_reference_text(reference_raw),
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
            reference_parts = [
                row[idx].strip()
                for idx in reference_text_cols
                if idx < len(row) and row[idx].strip()
            ]
            reference_raw = " | ".join(dict.fromkeys(reference_parts))

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
                mhc_allele=normalize_allele(allele),
                mhc_class=_normalize_mhc_class(mhc_class_raw, allele),
                pmid=normalize_pmid(pmid_raw),
                doi=normalize_doi(doi_raw),
                reference_text=_normalize_reference_text(reference_raw),
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
                raw_data={
                    "reference_raw": reference_raw,
                    "doi_raw": doi_raw,
                },
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
            antigen_species_raw = (
                row[header_map["antigen.species"]]
                if "antigen.species" in header_map and header_map["antigen.species"] < len(row)
                else ""
            ).strip()
            normalized_pmid = normalize_pmid(ref_id)
            normalized_doi = normalize_doi(ref_id)

            species_raw = species if species else (infer_species(mhc_a) if mhc_a else None)

            yield UnifiedRecord(
                peptide=epitope,
                mhc_allele=normalize_allele(mhc_a),
                mhc_class=_normalize_mhc_class(mhc_class, mhc_a),
                pmid=normalized_pmid,
                doi=normalized_doi,
                reference_text=_normalize_reference_text(ref_id),
                source="vdjdb",
                record_type="tcr",
                cdr3_alpha=cdr3 if gene == "TRA" else None,
                cdr3_beta=cdr3 if gene == "TRB" else None,
                trav=v_gene if gene == "TRA" else None,
                trbv=v_gene if gene == "TRB" else None,
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
                    mhc_allele=normalize_allele(row.get("MHC", "")),
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
        for rec in records:
            groups[rec.dedup_key()].append(rec)

        # Deduplicate each group
        deduped = []
        group_items = groups.items()
        if show_progress:
            group_items = tqdm(
                group_items,
                total=len(groups),
                desc="Deduplicating groups",
                unit="group",
                leave=False,
            )

        for key, group_records in group_items:
            if len(group_records) == 1:
                deduped.append(group_records[0])
                continue

            selected_group: List[UnifiedRecord] = []
            for rec in group_records:
                if not rec.pmid:
                    self.stats["records_without_pmid"] += 1

                matched_idx: Optional[int] = None
                for idx, existing in enumerate(selected_group):
                    if self._records_equivalent(rec, existing):
                        matched_idx = idx
                        break

                if matched_idx is None:
                    selected_group.append(rec)
                    continue

                existing = selected_group[matched_idx]
                if rec.source == existing.source:
                    self.stats["same_source_duplicates"] += 1
                else:
                    self.stats["cross_source_duplicates"] += 1
                if (
                    not (rec.pmid and existing.pmid and rec.pmid == existing.pmid)
                    and self._references_match(rec, existing)
                ):
                    self.stats["fuzzy_reference_duplicates"] += 1
                selected_group[matched_idx] = self._select_best([existing, rec])

            deduped.extend(selected_group)

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
            return (
                rec.record_type,
                (rec.value_type or "").strip().lower(),
                str(rec.qualifier),
                value_bucket,
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

    def _records_equivalent(self, left: UnifiedRecord, right: UnifiedRecord) -> bool:
        """Whether two records should collapse to one canonical row."""
        if left.dedup_key() != right.dedup_key():
            return False
        if self._sample_signature(left) != self._sample_signature(right):
            return False
        if self._references_match(left, right):
            return True
        # If neither has reference information, only collapse exact same-source duplicates.
        # This prevents aggressive dedup on sparse rows (common in export files).
        left_has_ref = bool(left.pmid or left.doi or left.reference_text)
        right_has_ref = bool(right.pmid or right.doi or right.reference_text)
        if not left_has_ref and not right_has_ref:
            return (
                left.source == right.source
                and self._record_payload_signature(left) == self._record_payload_signature(right)
            )
        return False

    def _select_best(self, records: List[UnifiedRecord]) -> UnifiedRecord:
        """Select best record from duplicates."""
        # Sort by source preference
        source_order = {s: i for i, s in enumerate(self.prefer_sources)}
        records.sort(key=lambda r: source_order.get(r.source, 999))

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

        records.sort(key=completeness, reverse=True)
        return records[0]

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
            )

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
                pbar.update(1)

        if pbar is not None:
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
        "effector_culture_condition": rec.effector_culture_condition or "",
        "apc_culture_condition": rec.apc_culture_condition or "",
        "in_vitro_process_type": rec.in_vitro_process_type or "",
        "in_vitro_responder_cell": rec.in_vitro_responder_cell or "",
        "in_vitro_stimulator_cell": rec.in_vitro_stimulator_cell or "",
        "cdr3_alpha": rec.cdr3_alpha or "",
        "cdr3_beta": rec.cdr3_beta or "",
        "trav": rec.trav or "",
        "trbv": rec.trbv or "",
        "species": rec.species,
        "antigen_species": rec.antigen_species or "",
    }


def classify_assay_type(rec: UnifiedRecord) -> str:
    """Map a unified record into a simplified assay bucket."""
    if rec.record_type == "elution":
        return "elution_ms"
    if rec.record_type == "processing":
        return "processing"
    if rec.record_type == "binding":
        value_type = (rec.value_type or "").strip().lower()
        method_text = " ".join(
            token
            for token in (
                (rec.value_type or "").strip().lower(),
                (rec.assay_method or "").strip().lower(),
            )
            if token
        )
        if any(token in method_text for token in ("processing", "cleavage", "tap transport", "tap assay", "erap")):
            return "processing"
        if any(token in method_text for token in ("kon", "on rate", "association rate", "ka")):
            return "binding_kon"
        if any(token in method_text for token in ("koff", "off rate", "dissociation rate")):
            return "binding_koff"
        if any(token in method_text for token in ("t_half", "half life", "half-life", "t1/2")):
            return "binding_t_half"
        if any(token in method_text for token in ("tm", "melt", "dissociation temperature")):
            return "binding_tm"
        if rec.value is None:
            if any(token in method_text for token in ("dia", "data-independent")):
                return "elution_ms_dia"
            if any(token in method_text for token in ("dda", "data-dependent")):
                return "elution_ms_dda"
            if any(token in method_text for token in ("targeted", "prm", "srm", "srm/ms", "mrm")):
                return "elution_ms_targeted"
            return "elution_ms"
        return "binding_affinity"
    if rec.record_type == "tcell":
        return "tcell_response"
    if rec.record_type == "tcr":
        return "tcr_pmhc"
    if rec.record_type == "bcell":
        return "bcell_response"
    return rec.record_type or "unknown"


def _write_tsv(records: List[UnifiedRecord], output_path: Path) -> None:
    fieldnames = [
        "peptide",
        "mhc_allele",
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for rec in records:
            writer.writerow(record_to_row(rec))


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
        "effector_culture_condition",
        "apc_culture_condition",
        "in_vitro_process_type",
        "in_vitro_responder_cell",
        "in_vitro_stimulator_cell",
        "cdr3_alpha",
        "cdr3_beta",
        "trav",
        "trbv",
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
    if verbose:
        print("Loading records from all sources...")

    records, load_stats = load_all_sources(
        data_dir, record_types=record_types, sources=sources, verbose=verbose
    )
    stats: Dict[str, Any] = {"load": load_stats}

    if verbose:
        print(f"  Loaded {len(records)} total records")
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

    deduper = CrossSourceDeduplicator()
    deduped = deduper.deduplicate(records, show_progress=verbose)
    stats.update(deduper.get_stats())

    if verbose:
        print(f"  Input: {stats['total_input']}")
        print(f"  Output: {stats['total_output']}")
        print(f"  Cross-source duplicates removed: {stats['cross_source_duplicates']}")
        print(f"  Same-source duplicates removed: {stats['same_source_duplicates']}")
        print(f"  Unique PMIDs: {stats['unique_pmids']}")
        input_by_source = stats.get("input_by_source", {})
        output_by_source = stats.get("by_source", {})
        output_by_assay = stats.get("output_by_assay", {})

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
            print("  Output records by assay:")
            for assay, count in output_by_assay.items():
                print(f"    {assay}: {count}")

        input_by_assay = load_stats.get("by_assay", {})
        if input_by_assay and output_by_assay:
            print("  Assay retention:")
            for assay, input_count in input_by_assay.items():
                kept = int(output_by_assay.get(assay, 0))
                retention = (kept / float(input_count)) if input_count else 0.0
                pct = retention * 100.0
                warning = " [WARN: >90% dropped]" if input_count >= 100 and pct < 10.0 else ""
                print(f"    {assay}: kept {kept}/{input_count} ({pct:.2f}%){warning}")

    # Save full merged table if requested.
    if output_path:
        if verbose:
            print(f"Saving to {output_path}...")
        _write_tsv(deduped, output_path)

        if verbose:
            print(f"  Saved {len(deduped)} records")

    if assay_output_dir is not None:
        if verbose:
            print(f"Writing assay CSVs to {assay_output_dir}...")
        file_map = write_assay_csvs(deduped, assay_output_dir)
        stats["assay_csv_files"] = file_map
        if verbose:
            print(f"  Wrote {len(file_map)} assay CSV files")

    return deduped, stats
