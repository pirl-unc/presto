"""Data loaders for Presto training data.

Supports multiple data sources with flexible column name detection:
- IEDB binding data (IC50, KD, EC50)
- IEDB kinetics data (kon, koff)
- IEDB stability data (t_half, Tm)
- IEDB processing data (cleavage, TAP)
- IEDB MS/elution data
- IEDB T-cell assay data
- VDJdb TCR-pMHC data

All loaders support:
- Flexible column name detection (handles IEDB naming variations)
- gzip and zip compressed files
- CSV and TSV formats
"""

import csv
import gzip
import zipfile
import io
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Union, Any, Tuple
from dataclasses import dataclass, field
import random
import re

import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from .collate import PrestoSample, PrestoCollator
from .vocab import normalize_organism, FOREIGN_CATEGORIES
from .allele_resolver import (
    AlleleResolver,
    HUMAN_B2M_SEQUENCE,
    class_i_beta2m_sequence,
    infer_mhc_class_optional,
    infer_species as infer_species_from_allele,
    normalize_mhc_class,
    normalize_species_label,
)


# =============================================================================
# Record Types - One for each data modality
# =============================================================================

MHC_ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWYX")
MIN_MHC_CHAIN_LENGTH = 70  # allow groove-bearing fragments, reject trivial truncations
NUCLEOTIDE_LIKE_SEQUENCE_CHARS = set("ACGTUNWSMKRYBDHV")


def _looks_like_nucleotide_sequence(sequence: str) -> bool:
    """Detect DNA/RNA-like full-length chains that would silently pass AA validation."""
    seq = "".join(ch for ch in str(sequence or "").strip().upper() if ch.isalpha())
    chars = set(seq)
    nucleotide_chars = chars & set("ACGTU")
    return bool(seq) and chars <= NUCLEOTIDE_LIKE_SEQUENCE_CHARS and len(nucleotide_chars) >= 3

@dataclass
class BindingRecord:
    """Binding affinity measurement (IC50, KD, EC50)."""
    peptide: str
    mhc_allele: str
    value: float                    # In nM
    qualifier: int = 0              # -1='<', 0='=', 1='>'
    measurement_type: str = "IC50"  # IC50, KD, EC50
    unit: str = "nM"
    assay_type: Optional[str] = None
    mhc_sequence: Optional[str] = None
    mhc_class: Optional[str] = None
    species: Optional[str] = None
    antigen_species: Optional[str] = None  # Source organism of the epitope
    source: str = "iedb"


@dataclass
class KineticsRecord:
    """Binding kinetics measurement (kon, koff)."""
    peptide: str
    mhc_allele: str
    kon: Optional[float] = None     # Association rate (1/Ms)
    koff: Optional[float] = None    # Dissociation rate (1/s)
    kon_qualifier: int = 0
    koff_qualifier: int = 0
    assay_type: Optional[str] = None
    mhc_sequence: Optional[str] = None
    mhc_class: Optional[str] = None
    species: Optional[str] = None
    antigen_species: Optional[str] = None
    source: str = "iedb"


@dataclass
class StabilityRecord:
    """pMHC stability measurement (t_half, Tm)."""
    peptide: str
    mhc_allele: str
    t_half: Optional[float] = None  # Half-life (hours)
    tm: Optional[float] = None      # Melting temperature (C)
    t_half_qualifier: int = 0
    tm_qualifier: int = 0
    assay_type: Optional[str] = None
    mhc_sequence: Optional[str] = None
    mhc_class: Optional[str] = None
    species: Optional[str] = None
    antigen_species: Optional[str] = None
    source: str = "iedb"


@dataclass
class ProcessingRecord:
    """Antigen processing data (cleavage, TAP transport)."""
    peptide: str
    flank_n: str = ""               # N-terminal flanking sequence
    flank_c: str = ""               # C-terminal flanking sequence
    label: float = 1.0              # Processing outcome (0-1)
    processing_type: str = "cleavage"  # cleavage, tap, processing
    mhc_allele: Optional[str] = None
    mhc_class: Optional[str] = None
    species: Optional[str] = None
    antigen_species: Optional[str] = None
    source: str = "iedb"


@dataclass
class ElutionRecord:
    """Mass spectrometry / elution data."""
    peptide: str
    alleles: List[str]              # Can be multiple (deconvolution needed)
    detected: bool = True           # Was peptide detected?
    cell_type: Optional[str] = None
    tissue: Optional[str] = None
    mhc_class: Optional[str] = None
    species: Optional[str] = None
    antigen_species: Optional[str] = None
    source: str = "iedb"


@dataclass
class TCellRecord:
    """T-cell assay data (IEDB format)."""
    peptide: str
    mhc_allele: str
    response: float                 # 0 or 1
    assay_type: Optional[str] = None  # Assay response measured (e.g. IFNg release)
    assay_method: Optional[str] = None  # Method (ELISPOT/ICS/multimer/etc.)
    effector_culture_condition: Optional[str] = None
    apc_name: Optional[str] = None
    apc_culture_condition: Optional[str] = None
    in_vitro_process_type: Optional[str] = None
    in_vitro_responder_cell: Optional[str] = None
    in_vitro_stimulator_cell: Optional[str] = None
    tcr_a_cdr3: Optional[str] = None
    tcr_b_cdr3: Optional[str] = None
    tcr_a_full: Optional[str] = None
    tcr_b_full: Optional[str] = None
    v_alpha: Optional[str] = None
    j_alpha: Optional[str] = None
    v_beta: Optional[str] = None
    j_beta: Optional[str] = None
    mhc_sequence: Optional[str] = None
    mhc_class: Optional[str] = None
    species: Optional[str] = None
    antigen_species: Optional[str] = None
    source: str = "iedb"


@dataclass
class BCellRecord:
    """B-cell assay data with inferred BCR chain classes."""
    peptide: str
    response: float                 # 0 or 1
    assay_type: Optional[str] = None
    heavy_chain_isotype: Optional[str] = None
    light_chain_isotype: Optional[str] = None
    heavy_chain_type: Optional[str] = None  # IGH
    light_chain_type: Optional[str] = None  # IGK or IGL
    species: Optional[str] = None
    antigen_species: Optional[str] = None
    source: str = "iedb"


@dataclass
class Sc10xVDJRecord:
    """10x VDJ contig record with normalized chain labels."""
    barcode: str
    chain_type: str                 # TRA/TRB/TRG/TRD/IGH/IGK/IGL
    cdr3: Optional[str] = None
    cdr3_nt: Optional[str] = None
    v_gene: Optional[str] = None
    j_gene: Optional[str] = None
    c_gene: Optional[str] = None
    productive: bool = True
    high_confidence: bool = True
    is_cell: bool = True
    phenotype: Optional[str] = None  # ab_T, gd_T, B_cell
    species: Optional[str] = None
    source: str = "10x"


@dataclass
class VDJdbRecord:
    """VDJdb TCR-pMHC data with V/J gene annotations."""
    peptide: str
    mhc_a: str                      # MHC alpha chain allele
    mhc_b: Optional[str] = None     # MHC beta chain (Class II) or B2M
    cdr3_alpha: Optional[str] = None
    cdr3_beta: Optional[str] = None
    v_alpha: Optional[str] = None
    j_alpha: Optional[str] = None
    v_beta: Optional[str] = None
    j_beta: Optional[str] = None
    gene: str = "TRB"               # TRA, TRB, etc.
    mhc_class: Optional[str] = None
    species: Optional[str] = None
    antigen_species: Optional[str] = None  # Pathogen species
    source: str = "vdjdb"


# Backward compatibility alias
TCRpMHCRecord = VDJdbRecord


@dataclass
class UniProtProtein:
    """A protein record from UniProt SwissProt with organism category."""
    accession: str
    sequence: str
    category: str   # one of ORGANISM_CATEGORIES
    organism: str = ""


# =============================================================================
# Flexible Column Detection
# =============================================================================

def _sniff_column(header: List[str], candidates: List[str]) -> Optional[int]:
    """Find column index matching any candidate name (case-insensitive)."""
    header_lower = [h.lower().strip() for h in header]
    for candidate in candidates:
        candidate_lower = candidate.lower().strip()
        for i, h in enumerate(header_lower):
            if candidate_lower == h or candidate_lower in h:
                return i
    return None


def _get_column(row: List[str], idx: Optional[int], default: str = "") -> str:
    """Safely get column value."""
    if idx is None or idx >= len(row):
        return default
    return row[idx].strip()


def _parse_qualifier(value: str) -> tuple:
    """Extract qualifier and numeric value from strings like '<500' or '>10000'."""
    value = value.strip()
    if not value:
        return 0, None

    qualifier = 0
    if value.startswith('<='):
        qualifier = -1
        value = value[2:]
    elif value.startswith('>='):
        qualifier = 1
        value = value[2:]
    elif value.startswith('<'):
        qualifier = -1
        value = value[1:]
    elif value.startswith('>'):
        qualifier = 1
        value = value[1:]
    elif value.startswith('='):
        qualifier = 0
        value = value[1:]

    try:
        return qualifier, float(value)
    except (ValueError, TypeError):
        return 0, None


def _parse_outcome(value: str) -> Optional[float]:
    """Parse T-cell response outcome to 0/1."""
    value = value.lower().strip()
    if value in ('positive', 'pos', '1', 'true', 'yes', '+'):
        return 1.0
    elif value in ('negative', 'neg', '0', 'false', 'no', '-'):
        return 0.0
    return None


def _parse_bool(value: str) -> bool:
    """Parse permissive boolean strings used by public data dumps."""
    normalized = value.strip().lower()
    return normalized in {"1", "true", "t", "yes", "y"}


def _infer_bcr_light_chain_type(isotype: str) -> Optional[str]:
    """Map light-chain isotype text to IGK/IGL."""
    value = (isotype or "").strip().lower()
    if not value:
        return None
    if "kappa" in value or value.startswith("k"):
        return "IGK"
    if "lambda" in value or value.startswith("l"):
        return "IGL"
    return None


def _infer_receptor_phenotype(chain_type: str) -> Optional[str]:
    """Infer cell phenotype class from receptor chain type."""
    chain = (chain_type or "").upper()
    if chain in {"TRA", "TRB"}:
        return "ab_T"
    if chain in {"TRG", "TRD"}:
        return "gd_T"
    if chain in {"IGH", "IGK", "IGL"}:
        return "B_cell"
    return None


def _parse_alleles(value: str) -> List[str]:
    """Parse comma/semicolon separated allele list."""
    if not value:
        return []
    # Split on comma, semicolon, or slash
    alleles = re.split(r'[,;/]', value)
    return [a.strip() for a in alleles if a.strip()]


def _is_binding_affinity_measurement(measurement: str, unit: str) -> bool:
    """Whether a measurement likely represents binding affinity in concentration units."""
    m = (measurement or "").lower()
    u = (unit or "").lower().strip()
    if any(token in m for token in ("ic50", "ec50", "kd", "inhibitory concentration", "effective concentration", "dissociation constant")):
        return True
    if u in {"nm", "nanomolar", "n m"}:
        return True
    return False


def _infer_mhc_class(allele: Optional[str]) -> Optional[str]:
    """Infer MHC class from allele name, preferring mhcgnomes."""
    if not allele:
        return None
    inferred = infer_mhc_class_optional(allele)
    if inferred is not None:
        return inferred
    allele_upper = allele.upper()
    class2_genes = ['DRA', 'DRB', 'DQA', 'DQB', 'DPA', 'DPB', 'DR', 'DQ', 'DP']
    for gene in class2_genes:
        if gene in allele_upper:
            return "II"
    return "I"


def _resolve_mhc_class(value: Optional[str], allele: Optional[str] = None) -> Optional[str]:
    """Normalize explicit MHC class labels and otherwise infer from allele."""
    normalized = normalize_mhc_class(value, default=None)
    if normalized is not None:
        return normalized
    return _infer_mhc_class(allele)


def _open_file(path: Union[str, Path]) -> Iterator[str]:
    """Open file, handling gzip and zip compression."""
    path = Path(path)

    if path.suffix == '.gz':
        with gzip.open(path, 'rt', encoding='utf-8') as f:
            yield from f
    elif path.suffix == '.zip':
        with zipfile.ZipFile(path, 'r') as zf:
            # Find first CSV/TSV file
            for name in zf.namelist():
                if name.endswith(('.csv', '.tsv', '.txt')):
                    with zf.open(name) as f:
                        yield from io.TextIOWrapper(f, encoding='utf-8')
                    break
    else:
        with open(path, 'r', encoding='utf-8') as f:
            yield from f


def _detect_delimiter(first_line: str) -> str:
    """Detect CSV vs TSV."""
    if '\t' in first_line:
        return '\t'
    return ','


def _looks_like_multilevel_header(header_1: List[str], header_2: List[str]) -> bool:
    """Detect IEDB/CEDAR two-row header exports."""
    if not header_1 or not header_2:
        return False
    if len(header_1) != len(header_2):
        return False
    joined = " ".join(cell.lower().strip() for cell in header_2 if cell)
    markers = (
        "quantitative measurement",
        "qualitative measurement",
        "qualitative measure",
        "measurement inequality",
        "response measured",
        "mhc restriction name",
        "cedar iri",
    )
    return any(marker in joined for marker in markers)


def _combine_multilevel_header(header_1: List[str], header_2: List[str]) -> List[str]:
    """Combine two header rows into a single, searchable header row."""
    combined: List[str] = []
    for upper, lower in zip(header_1, header_2):
        upper_clean = upper.strip()
        lower_clean = lower.strip()
        if upper_clean and lower_clean and upper_clean.lower() != lower_clean.lower():
            combined.append(f"{upper_clean} {lower_clean}")
        elif lower_clean:
            combined.append(lower_clean)
        else:
            combined.append(upper_clean)
    return combined


def _parse_header_and_rows(lines: List[str]) -> tuple[List[str], List[List[str]]]:
    """Parse CSV lines and normalize one-row/two-row headers."""
    if not lines:
        return [], []
    delimiter = _detect_delimiter(lines[0])
    rows = list(csv.reader(lines, delimiter=delimiter))
    if not rows:
        return [], []
    if len(rows) >= 2 and _looks_like_multilevel_header(rows[0], rows[1]):
        return _combine_multilevel_header(rows[0], rows[1]), rows[2:]
    return rows[0], rows[1:]


def _parse_inequality_qualifier(value: str) -> int:
    """Parse inequality symbols into qualifier codes."""
    normalized = value.strip().lower()
    if normalized in ("<", "<=", "lt", "le"):
        return -1
    if normalized in (">", ">=", "gt", "ge"):
        return 1
    return 0


# =============================================================================
# IEDB Loaders
# =============================================================================

def load_iedb_binding(path: Union[str, Path]) -> Iterator[BindingRecord]:
    """Load IEDB binding data with flexible column detection.

    Handles various IEDB export formats and column naming conventions.
    """
    lines = list(_open_file(path))
    if not lines:
        return

    header, rows = _parse_header_and_rows(lines)
    if not header:
        return

    # Detect columns
    pep_idx = _sniff_column(
        header,
        ['epitope name', 'linear peptide sequence', 'peptide', 'description'],
    )
    allele_idx = _sniff_column(
        header,
        ['mhc restriction name', 'mhc allele name', 'allele', 'mhc allele', 'allele name'],
    )
    value_idx = _sniff_column(
        header,
        ['assay quantitative measurement', 'quantitative measurement', 'measurement value', 'value', 'ic50 (nm)', 'kd (nm)', 'ec50 (nm)'],
    )
    qual_idx = _sniff_column(
        header,
        ['assay measurement inequality', 'measurement inequality', 'inequality', 'qualifier'],
    )
    type_idx = _sniff_column(
        header,
        ['assay response measured', 'measurement type', 'method', 'assay group'],
    )
    unit_idx = _sniff_column(header, ['assay units', 'measurement unit', 'units', 'unit'])
    assay_idx = _sniff_column(header, ['assay method', 'assay group', 'assay', 'assay type'])
    class_idx = _sniff_column(header, ['mhc restriction class', 'mhc class', 'class', 'mhc_class'])
    species_idx = _sniff_column(header, ['epitope species', 'host organism', 'organism', 'species', 'host'])
    antigen_species_idx = _sniff_column(header, [
        'antigen source organism', 'source organism', 'epitope source organism',
        'antigen.species', 'pathogen',
    ])

    for row in rows:
        peptide = _get_column(row, pep_idx)
        if not peptide:
            continue

        allele = _get_column(row, allele_idx)
        value_str = _get_column(row, value_idx)
        qual_str = _get_column(row, qual_idx, '=')

        # Parse qualifier from value string if not separate
        if qual_idx is None:
            qualifier, value = _parse_qualifier(value_str)
        else:
            qualifier = {'>': 1, '<': -1, '>=': 1, '<=': -1}.get(qual_str, 0)
            try:
                value = float(value_str) if value_str else None
            except ValueError:
                value = None

        if value is None:
            continue
        measurement_type = _get_column(row, type_idx, 'IC50')
        unit = _get_column(row, unit_idx, 'nM')
        if not _is_binding_affinity_measurement(measurement_type, unit):
            continue

        mhc_class = _resolve_mhc_class(_get_column(row, class_idx) or None, allele)

        yield BindingRecord(
            peptide=peptide,
            mhc_allele=allele,
            value=value,
            qualifier=qualifier,
            measurement_type=measurement_type,
            unit=unit,
            assay_type=_get_column(row, assay_idx) or None,
            mhc_class=mhc_class,
            species=_get_column(row, species_idx) or infer_species_from_allele(allele) or None,
            antigen_species=_get_column(row, antigen_species_idx) or None,
            source='iedb',
        )


def load_iedb_kinetics(path: Union[str, Path]) -> Iterator[KineticsRecord]:
    """Load IEDB kinetics data (kon, koff)."""
    lines = list(_open_file(path))
    if not lines:
        return

    header, rows = _parse_header_and_rows(lines)
    if not header:
        return

    pep_idx = _sniff_column(header, ['epitope name', 'linear peptide sequence', 'peptide', 'epitope'])
    allele_idx = _sniff_column(header, ['mhc restriction name', 'mhc allele name', 'allele', 'mhc allele'])
    kon_idx = _sniff_column(header, ['kon', 'ka', 'k_on', 'association rate'])
    koff_idx = _sniff_column(header, ['koff', 'kd', 'k_off', 'dissociation rate'])
    response_idx = _sniff_column(
        header,
        ['response measured', 'measurement type', 'assay response measured'],
    )
    units_idx = _sniff_column(header, ['units', 'measurement unit', 'assay units'])
    value_idx = _sniff_column(
        header,
        ['quantitative measurement', 'measurement value', 'value'],
    )
    ineq_idx = _sniff_column(header, ['measurement inequality', 'inequality', 'qualifier'])
    assay_idx = _sniff_column(header, ['assay method', 'assay group', 'assay', 'assay type'])
    class_idx = _sniff_column(header, ['mhc restriction class', 'mhc class', 'class'])
    species_idx = _sniff_column(header, ['epitope species', 'host organism', 'organism', 'species', 'host'])
    antigen_species_idx = _sniff_column(header, [
        'antigen source organism', 'source organism', 'epitope source organism',
        'antigen.species', 'pathogen',
    ])

    def _normalize_rate_units(value: Optional[float], units: str, is_kon: bool) -> Optional[float]:
        if value is None:
            return None
        unit = units.lower().strip().replace(" ", "")
        normalized = float(value)

        # Normalize time base to per-second where possible.
        if unit in {'1/min', 'min^-1', '/min'}:
            normalized = normalized / 60.0
        elif unit in {'1/h', 'h^-1', 'hr^-1', '/h'}:
            normalized = normalized / 3600.0

        # Normalize concentration base for on-rates to M^-1 s^-1.
        if is_kon and unit in {'nm^-1s^-1', '1/nm/s', 'nm-1s-1'}:
            normalized = normalized * 1e9

        return normalized

    for row in rows:
        peptide = _get_column(row, pep_idx)
        if not peptide:
            continue

        kon_str = _get_column(row, kon_idx)
        koff_str = _get_column(row, koff_idx)

        kon_qual, kon = _parse_qualifier(kon_str)
        koff_qual, koff = _parse_qualifier(koff_str)

        # Fallback for IEDB/CEDAR exports where kinetics are encoded as one
        # "response measured" row per assay (e.g. "on rate", "off rate").
        if kon is None and koff is None:
            measurement = _get_column(row, response_idx).lower()
            units = _get_column(row, units_idx)
            value_raw = _get_column(row, value_idx)
            ineq = _get_column(row, ineq_idx)
            generic_qual = _parse_inequality_qualifier(ineq)
            if generic_qual == 0:
                generic_qual, generic_value = _parse_qualifier(value_raw)
            else:
                try:
                    generic_value = float(value_raw) if value_raw else None
                except ValueError:
                    generic_value = None

            is_koff = (
                'off rate' in measurement
                or 'koff' in measurement
                or 'k_off' in measurement
                or 'dissociation rate' in measurement
            )
            is_kon = (
                'on rate' in measurement
                or 'kon' in measurement
                or 'k_on' in measurement
                or 'association rate' in measurement
            )

            if is_kon and generic_value is not None:
                kon = _normalize_rate_units(generic_value, units, is_kon=True)
                kon_qual = generic_qual
            if is_koff and generic_value is not None:
                koff = _normalize_rate_units(generic_value, units, is_kon=False)
                koff_qual = generic_qual

        if kon is None and koff is None:
            continue

        allele = _get_column(row, allele_idx)

        yield KineticsRecord(
            peptide=peptide,
            mhc_allele=allele,
            kon=kon,
            koff=koff,
            kon_qualifier=kon_qual,
            koff_qualifier=koff_qual,
            assay_type=_get_column(row, assay_idx) or None,
            mhc_class=_resolve_mhc_class(_get_column(row, class_idx) or None, allele),
            species=_get_column(row, species_idx) or infer_species_from_allele(allele) or None,
            antigen_species=_get_column(row, antigen_species_idx) or None,
            source='iedb',
        )


def load_iedb_stability(path: Union[str, Path]) -> Iterator[StabilityRecord]:
    """Load IEDB stability data (t_half, Tm)."""
    lines = list(_open_file(path))
    if not lines:
        return

    header, rows = _parse_header_and_rows(lines)
    if not header:
        return

    pep_idx = _sniff_column(
        header,
        ['peptide', 'linear peptide sequence', 'epitope name', 'epitope'],
    )
    allele_idx = _sniff_column(
        header,
        ['mhc allele name', 'mhc restriction name', 'allele', 'mhc allele'],
    )
    # Dedicated stability columns (simple exports).
    thalf_idx = _sniff_column(header, ['t_half', 'half-life', 't1/2', 'half life'])
    tm_idx = _sniff_column(header, ['tm', 'melting temperature', 'melting point'])
    # Generic IEDB/CEDAR assay columns.
    response_idx = _sniff_column(
        header,
        ['response measured', 'measurement type', 'assay response measured'],
    )
    units_idx = _sniff_column(header, ['units', 'measurement unit', 'assay units'])
    value_idx = _sniff_column(
        header,
        [
            'quantitative measurement',
            'measurement value',
            'value',
            't_half',
            'half-life',
            't1/2',
            'half life',
            'tm',
            'melting temperature',
            'melting point',
        ],
    )
    ineq_idx = _sniff_column(header, ['measurement inequality', 'inequality', 'qualifier'])
    assay_idx = _sniff_column(header, ['assay group', 'assay', 'assay type', 'method'])
    class_idx = _sniff_column(header, ['mhc class', 'class'])
    species_idx = _sniff_column(header, ['epitope species', 'host organism', 'organism', 'species', 'host'])
    antigen_species_idx = _sniff_column(header, [
        'antigen source organism', 'source organism', 'epitope source organism',
        'antigen.species', 'pathogen',
    ])

    def _convert_time_to_hours(value: float, units: str) -> float:
        unit = units.lower().strip()
        if unit in {'h', 'hr', 'hrs', 'hour', 'hours'}:
            return value
        if unit in {'m', 'min', 'mins', 'minute', 'minutes'}:
            return value / 60.0
        if unit in {'s', 'sec', 'secs', 'second', 'seconds'}:
            return value / 3600.0
        # Unknown or missing unit: default to hours for backward compatibility.
        return value

    for row in rows:
        peptide = _get_column(row, pep_idx)
        if not peptide:
            continue

        # Parse dedicated columns first.
        thalf_str = _get_column(row, thalf_idx)
        tm_str = _get_column(row, tm_idx)
        thalf_qual, thalf = _parse_qualifier(thalf_str)
        tm_qual, tm = _parse_qualifier(tm_str)

        # Parse generic IEDB assay-style columns when dedicated columns are absent.
        measurement = _get_column(row, response_idx).lower()
        units = _get_column(row, units_idx)
        value_raw = _get_column(row, value_idx)
        ineq = _get_column(row, ineq_idx)
        generic_qual = _parse_inequality_qualifier(ineq)
        if generic_qual == 0:
            generic_qual, generic_value = _parse_qualifier(value_raw)
        else:
            try:
                generic_value = float(value_raw) if value_raw else None
            except ValueError:
                generic_value = None

        is_ic50_like = (
            'ic50' in measurement
            or 'ec50' in measurement
            or 'kd' in measurement
            or 'inhibitory concentration' in measurement
            or 'effective concentration' in measurement
            or 'dissociation constant' in measurement
        )
        is_half_life = (
            not is_ic50_like
            and (
                'half life' in measurement
                or 'half-life' in measurement
                or 't1/2' in measurement
                or 'dissociation half life' in measurement
            )
        )
        is_tm = (
            'dissociation temperature' in measurement
            or 'melting temperature' in measurement
            or 'melting point' in measurement
            or measurement == 'tm'
        )

        if thalf is None and is_half_life and generic_value is not None:
            thalf = _convert_time_to_hours(generic_value, units)
            thalf_qual = generic_qual

        if tm is None and is_tm and generic_value is not None:
            tm = generic_value
            # If units are Kelvin, convert to Celsius.
            if units.lower().strip() in {'k', 'kelvin'} and tm > 200:
                tm = tm - 273.15
            tm_qual = generic_qual

        if thalf is None and tm is None:
            continue

        allele = _get_column(row, allele_idx)

        yield StabilityRecord(
            peptide=peptide,
            mhc_allele=allele,
            t_half=thalf,
            tm=tm,
            t_half_qualifier=thalf_qual,
            tm_qualifier=tm_qual,
            assay_type=_get_column(row, assay_idx) or None,
            mhc_class=_resolve_mhc_class(_get_column(row, class_idx) or None, allele),
            species=_get_column(row, species_idx) or infer_species_from_allele(allele) or None,
            antigen_species=_get_column(row, antigen_species_idx) or None,
            source='iedb',
        )


def load_iedb_processing(path: Union[str, Path]) -> Iterator[ProcessingRecord]:
    """Load IEDB antigen processing data."""
    lines = list(_open_file(path))
    if not lines:
        return

    header, rows = _parse_header_and_rows(lines)
    if not header:
        return

    pep_idx = _sniff_column(header, ['epitope name', 'linear peptide sequence', 'peptide', 'epitope'])
    flankn_idx = _sniff_column(header, ['n-terminal', 'flank_n', 'n_flank', 'upstream'])
    flankc_idx = _sniff_column(header, ['c-terminal', 'flank_c', 'c_flank', 'downstream'])
    label_idx = _sniff_column(header, ['assay qualitative measurement', 'outcome', 'label', 'processed', 'cleaved'])
    type_idx = _sniff_column(header, ['assay response measured', 'processing type', 'type', 'method'])
    allele_idx = _sniff_column(header, ['mhc restriction name', 'mhc allele name', 'allele'])
    class_idx = _sniff_column(header, ['mhc restriction class', 'mhc class', 'class'])
    species_idx = _sniff_column(header, ['epitope species', 'host organism', 'organism', 'species', 'host'])
    antigen_species_idx = _sniff_column(header, [
        'antigen source organism', 'source organism', 'epitope source organism',
        'antigen.species', 'pathogen',
    ])

    for row in rows:
        peptide = _get_column(row, pep_idx)
        if not peptide:
            continue

        processing_type = _get_column(row, type_idx, 'processing')
        ptype_lower = processing_type.lower()
        if not any(term in ptype_lower for term in ('processing', 'cleavage', 'tap', 'proteasome', 'transport')):
            continue

        label_str = _get_column(row, label_idx, '1')
        label = _parse_outcome(label_str)
        if label is None:
            try:
                label = float(label_str)
            except ValueError:
                label = 1.0

        allele = _get_column(row, allele_idx)

        yield ProcessingRecord(
            peptide=peptide,
            flank_n=_get_column(row, flankn_idx),
            flank_c=_get_column(row, flankc_idx),
            label=label,
            processing_type=processing_type,
            mhc_allele=allele or None,
            mhc_class=_resolve_mhc_class(_get_column(row, class_idx) or None, allele),
            species=_get_column(row, species_idx) or infer_species_from_allele(allele) or None,
            antigen_species=_get_column(row, antigen_species_idx) or None,
            source='iedb',
        )


def load_iedb_elution(path: Union[str, Path]) -> Iterator[ElutionRecord]:
    """Load IEDB MS/elution data."""
    lines = list(_open_file(path))
    if not lines:
        return

    header, rows = _parse_header_and_rows(lines)
    if not header:
        return

    pep_idx = _sniff_column(header, ['epitope name', 'linear peptide sequence', 'peptide', 'epitope'])
    allele_idx = _sniff_column(header, ['mhc restriction name', 'mhc allele name', 'allele', 'mhc allele', 'alleles'])
    cell_idx = _sniff_column(header, ['antigen presenting cell name', 'cell type', 'cell line', 'apc name', 'cell'])
    tissue_idx = _sniff_column(header, ['tissue', 'tissue type', 'sample'])
    class_idx = _sniff_column(header, ['mhc restriction class', 'mhc class', 'class'])
    species_idx = _sniff_column(header, ['epitope species', 'host organism', 'organism', 'species'])
    response_idx = _sniff_column(header, ['assay response measured', 'response measured'])
    assay_method_idx = _sniff_column(header, ['assay method', 'assay type', 'assay group'])
    detected_idx = _sniff_column(header, ['assay qualitative measurement', 'detected', 'positive', 'hit'])
    antigen_species_idx = _sniff_column(header, [
        'antigen source organism', 'source organism', 'epitope source organism',
        'antigen.species', 'pathogen',
    ])

    for row in rows:
        peptide = _get_column(row, pep_idx)
        if not peptide:
            continue

        response_measured = _get_column(row, response_idx)
        assay_method = _get_column(row, assay_method_idx)
        lowered_context = f"{response_measured} {assay_method}".lower()
        if not any(term in lowered_context for term in ('ligand presentation', 'mass spectrometry', 'elution', 'immunopeptid')):
            continue

        alleles = _parse_alleles(_get_column(row, allele_idx))
        if not alleles:
            continue

        # Elution data is typically positive-only
        detected_str = _get_column(row, detected_idx, '1')
        detected = _parse_outcome(detected_str)
        if detected is None:
            detected = 1.0

        mhc_class = _resolve_mhc_class(_get_column(row, class_idx) or None, alleles[0])

        yield ElutionRecord(
            peptide=peptide,
            alleles=alleles,
            detected=bool(detected),
            cell_type=_get_column(row, cell_idx) or None,
            tissue=_get_column(row, tissue_idx) or None,
            mhc_class=mhc_class,
            species=_get_column(row, species_idx) or infer_species_from_allele(alleles[0]) or None,
            antigen_species=_get_column(row, antigen_species_idx) or None,
            source='iedb',
        )


def load_iedb_tcell(path: Union[str, Path]) -> Iterator[TCellRecord]:
    """Load IEDB T-cell assay data."""
    lines = list(_open_file(path))
    if not lines:
        return

    header, rows = _parse_header_and_rows(lines)
    if not header:
        return

    pep_idx = _sniff_column(
        header,
        ['epitope name', 'linear peptide sequence', 'peptide', 'description'],
    )
    allele_idx = _sniff_column(
        header,
        ['mhc restriction name', 'mhc allele name', 'mhc allele', 'allele', 'mhc.a'],
    )
    outcome_idx = _sniff_column(
        header,
        ['assay qualitative measurement', 'qualitative measurement', 'positive/negative', 'outcome', 'response'],
    )
    assay_idx = _sniff_column(
        header,
        ['assay response measured', 'assay group', 'assay type', 'assay'],
    )
    assay_method_idx = _sniff_column(
        header,
        ['assay method', 'method'],
    )
    class_idx = _sniff_column(header, ['mhc restriction class', 'mhc class', 'class'])
    species_idx = _sniff_column(
        header,
        ['epitope species', 'host organism', 'organism', 'species'],
    )
    antigen_species_idx = _sniff_column(header, [
        'antigen source organism', 'source organism', 'epitope source organism',
        'antigen.species', 'pathogen',
    ])
    effector_culture_idx = _sniff_column(
        header,
        ['effector cell culture condition', 'effector culture condition'],
    )
    apc_name_idx = _sniff_column(
        header,
        ['antigen presenting cell name', 'apc name', 'presenting cell'],
    )
    apc_culture_idx = _sniff_column(
        header,
        ['antigen presenting cell culture condition', 'apc culture condition'],
    )
    in_vitro_process_idx = _sniff_column(
        header,
        ['in vitro process process type', 'in vitro process'],
    )
    in_vitro_responder_idx = _sniff_column(
        header,
        ['in vitro responder cell name', 'in vitro responder cell'],
    )
    in_vitro_stimulator_idx = _sniff_column(
        header,
        ['in vitro stimulator cell name', 'in vitro stimulator cell'],
    )

    # Optional TCR columns (present in some exports).
    cdr3a_idx = _sniff_column(header, ['cdr3.alpha', 'cdr3a', 'a cdr3', 'tcr alpha cdr3'])
    cdr3b_idx = _sniff_column(header, ['cdr3.beta', 'cdr3b', 'b cdr3', 'tcr beta cdr3'])
    tcra_idx = _sniff_column(header, ['tcr alpha sequence', 'tcr a aa seq', 'tcr_a', 'tcr alpha'])
    tcrb_idx = _sniff_column(header, ['tcr beta sequence', 'tcr b aa seq', 'tcr_b', 'tcr beta'])
    va_idx = _sniff_column(header, ['trav', 'v alpha', 'v.alpha', 'tcr a v gene'])
    ja_idx = _sniff_column(header, ['traj', 'j alpha', 'j.alpha', 'tcr a j gene'])
    vb_idx = _sniff_column(header, ['trbv', 'v beta', 'v.beta', 'tcr b v gene'])
    jb_idx = _sniff_column(header, ['trbj', 'j beta', 'j.beta', 'tcr b j gene'])

    for row in rows:
        peptide = _get_column(row, pep_idx)
        if not peptide:
            continue

        allele_raw = _get_column(row, allele_idx)
        alleles = _parse_alleles(allele_raw) if allele_raw else []
        allele = alleles[0] if alleles else allele_raw
        outcome_str = _get_column(row, outcome_idx)
        response = _parse_outcome(outcome_str)
        if response is None:
            lowered = outcome_str.lower().strip()
            if "positive" in lowered or "reactive" in lowered:
                response = 1.0
            elif "negative" in lowered or "non-reactive" in lowered:
                response = 0.0
            else:
                try:
                    response = 1.0 if float(lowered) > 0 else 0.0
                except ValueError:
                    continue

        yield TCellRecord(
            peptide=peptide,
            mhc_allele=allele,
            response=response,
            assay_type=_get_column(row, assay_idx) or None,
            assay_method=_get_column(row, assay_method_idx) or None,
            effector_culture_condition=_get_column(row, effector_culture_idx) or None,
            apc_name=_get_column(row, apc_name_idx) or None,
            apc_culture_condition=_get_column(row, apc_culture_idx) or None,
            in_vitro_process_type=_get_column(row, in_vitro_process_idx) or None,
            in_vitro_responder_cell=_get_column(row, in_vitro_responder_idx) or None,
            in_vitro_stimulator_cell=_get_column(row, in_vitro_stimulator_idx) or None,
            tcr_a_cdr3=_get_column(row, cdr3a_idx) or None,
            tcr_b_cdr3=_get_column(row, cdr3b_idx) or None,
            tcr_a_full=_get_column(row, tcra_idx) or None,
            tcr_b_full=_get_column(row, tcrb_idx) or None,
            v_alpha=_get_column(row, va_idx) or None,
            j_alpha=_get_column(row, ja_idx) or None,
            v_beta=_get_column(row, vb_idx) or None,
            j_beta=_get_column(row, jb_idx) or None,
            mhc_class=_resolve_mhc_class(_get_column(row, class_idx) or None, allele),
            species=_get_column(row, species_idx) or infer_species_from_allele(allele) or None,
            antigen_species=_get_column(row, antigen_species_idx) or None,
            source='iedb',
        )


def load_iedb_bcell(path: Union[str, Path]) -> Iterator[BCellRecord]:
    """Load IEDB B-cell assay data with inferred BCR chain classes."""
    lines = list(_open_file(path))
    if not lines:
        return

    header, rows = _parse_header_and_rows(lines)
    if not header:
        return

    pep_idx = _sniff_column(
        header,
        ['epitope name', 'linear peptide sequence', 'peptide', 'description'],
    )
    outcome_idx = _sniff_column(
        header,
        [
            'assay qualitative measure',
            'assay qualitative measurement',
            'qualitative measure',
            'qualitative measurement',
            'positive/negative',
            'outcome',
            'response',
        ],
    )
    assay_idx = _sniff_column(
        header,
        ['assay response measured', 'assay group', 'assay type', 'assay'],
    )
    heavy_idx = _sniff_column(
        header,
        ['heavy chain isotype', 'assay antibody heavy chain isotype', 'heavy isotype'],
    )
    light_idx = _sniff_column(
        header,
        ['light chain isotype', 'assay antibody light chain isotype', 'light isotype'],
    )
    species_idx = _sniff_column(
        header,
        ['host name', 'host organism', 'host species'],
    )
    antigen_species_idx = _sniff_column(
        header,
        ['epitope source organism', 'epitope species', 'source organism'],
    )

    for row in rows:
        peptide = _get_column(row, pep_idx)
        if not peptide:
            continue

        outcome_str = _get_column(row, outcome_idx)
        response = _parse_outcome(outcome_str)
        if response is None:
            lowered = outcome_str.lower().strip()
            if "positive" in lowered or "reactive" in lowered:
                response = 1.0
            elif "negative" in lowered or "non-reactive" in lowered:
                response = 0.0
            else:
                try:
                    response = 1.0 if float(lowered) > 0 else 0.0
                except ValueError:
                    continue

        heavy_isotype = _get_column(row, heavy_idx)
        light_isotype = _get_column(row, light_idx)
        heavy_chain_type = "IGH" if heavy_isotype else None
        light_chain_type = _infer_bcr_light_chain_type(light_isotype)

        yield BCellRecord(
            peptide=peptide,
            response=response,
            assay_type=_get_column(row, assay_idx) or None,
            heavy_chain_isotype=heavy_isotype or None,
            light_chain_isotype=light_isotype or None,
            heavy_chain_type=heavy_chain_type,
            light_chain_type=light_chain_type,
            species=_get_column(row, species_idx) or None,
            antigen_species=_get_column(row, antigen_species_idx) or None,
            source='iedb',
        )


def load_10x_vdj(path: Union[str, Path]) -> Iterator[Sc10xVDJRecord]:
    """Load 10x Genomics VDJ contig annotations."""
    lines = list(_open_file(path))
    if not lines:
        return

    delimiter = _detect_delimiter(lines[0])
    reader = csv.reader(lines, delimiter=delimiter)
    header = next(reader)

    barcode_idx = _sniff_column(header, ['barcode', 'cell_id'])
    chain_idx = _sniff_column(header, ['chain'])
    cdr3_idx = _sniff_column(header, ['cdr3'])
    cdr3_nt_idx = _sniff_column(header, ['cdr3_nt'])
    v_idx = _sniff_column(header, ['v_gene', 'v'])
    j_idx = _sniff_column(header, ['j_gene', 'j'])
    c_idx = _sniff_column(header, ['c_gene', 'c'])
    productive_idx = _sniff_column(header, ['productive'])
    high_conf_idx = _sniff_column(header, ['high_confidence'])
    is_cell_idx = _sniff_column(header, ['is_cell'])

    for row in reader:
        chain = _get_column(row, chain_idx).upper()
        if chain not in {"TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL"}:
            continue

        yield Sc10xVDJRecord(
            barcode=_get_column(row, barcode_idx),
            chain_type=chain,
            cdr3=_get_column(row, cdr3_idx) or None,
            cdr3_nt=_get_column(row, cdr3_nt_idx) or None,
            v_gene=_get_column(row, v_idx) or None,
            j_gene=_get_column(row, j_idx) or None,
            c_gene=_get_column(row, c_idx) or None,
            productive=_parse_bool(_get_column(row, productive_idx, 'true')),
            high_confidence=_parse_bool(_get_column(row, high_conf_idx, 'true')),
            is_cell=_parse_bool(_get_column(row, is_cell_idx, 'true')),
            phenotype=_infer_receptor_phenotype(chain),
            species='human',
            source='10x',
        )


# =============================================================================
# VDJdb Loader
# =============================================================================

def load_vdjdb(path: Union[str, Path]) -> Iterator[VDJdbRecord]:
    """Load VDJdb TCR-pMHC data."""
    lines = list(_open_file(path))
    if not lines:
        return

    delimiter = _detect_delimiter(lines[0])
    reader = csv.reader(lines, delimiter=delimiter)
    header = next(reader)

    gene_idx = _sniff_column(header, ['gene'])
    cdr3_idx = _sniff_column(header, ['cdr3', 'cdr3fix'])
    v_idx = _sniff_column(header, ['v', 'v.segm', 'v.segment'])
    j_idx = _sniff_column(header, ['j', 'j.segm', 'j.segment'])
    pep_idx = _sniff_column(header, ['antigen.epitope', 'epitope', 'peptide'])
    mhca_idx = _sniff_column(header, ['mhc.a', 'mhc_a', 'mhca', 'allele'])
    mhcb_idx = _sniff_column(header, ['mhc.b', 'mhc_b', 'mhcb'])
    class_idx = _sniff_column(header, ['mhc.class', 'mhc_class', 'class'])
    species_idx = _sniff_column(header, ['species', 'organism'])
    antigen_species_idx = _sniff_column(header, ['antigen.species', 'pathogen'])

    for row in reader:
        peptide = _get_column(row, pep_idx)
        cdr3 = _get_column(row, cdr3_idx)
        if not peptide or not cdr3:
            continue

        gene = _get_column(row, gene_idx, 'TRB')
        mhc_a = _get_column(row, mhca_idx)
        mhc_class = _resolve_mhc_class(_get_column(row, class_idx) or None, mhc_a)

        # Assign CDR3 to alpha or beta based on gene
        cdr3_alpha = cdr3 if gene.upper() in ('TRA', 'TRAD') else None
        cdr3_beta = cdr3 if gene.upper() in ('TRB', 'TRBD') else None

        yield VDJdbRecord(
            peptide=peptide,
            mhc_a=mhc_a,
            mhc_b=_get_column(row, mhcb_idx) or None,
            cdr3_alpha=cdr3_alpha,
            cdr3_beta=cdr3_beta,
            v_alpha=_get_column(row, v_idx) if gene.upper() == 'TRA' else None,
            j_alpha=_get_column(row, j_idx) if gene.upper() == 'TRA' else None,
            v_beta=_get_column(row, v_idx) if gene.upper() == 'TRB' else None,
            j_beta=_get_column(row, j_idx) if gene.upper() == 'TRB' else None,
            gene=gene,
            mhc_class=mhc_class,
            species=_get_column(row, species_idx) or infer_species_from_allele(mhc_a) or None,
            antigen_species=_get_column(row, antigen_species_idx) or None,
            source='vdjdb',
        )


# =============================================================================
# Simple CSV Loaders (for backward compatibility)
# =============================================================================

def load_binding_csv(path: str) -> List[BindingRecord]:
    """Load binding data from simple CSV format."""
    return list(load_iedb_binding(path))


def load_elution_csv(path: str) -> List[ElutionRecord]:
    """Load elution data from simple CSV format."""
    return list(load_iedb_elution(path))


def load_tcr_pmhc_csv(path: str) -> List[TCellRecord]:
    """Load TCR-pMHC data from simple CSV format."""
    return list(load_iedb_tcell(path))


def load_mhc_fasta(path: str) -> Dict[str, str]:
    """Load MHC sequences from FASTA file."""
    sequences = {}
    current_name = None
    current_seq = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_name and current_seq:
                    sequences[current_name] = ''.join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_name and current_seq:
            sequences[current_name] = ''.join(current_seq)

    return sequences


def load_uniprot_proteins(path: Union[str, Path]) -> List[UniProtProtein]:
    """Load parsed UniProt SwissProt proteins from TSV.

    Expects the TSV produced by :func:`downloaders.parse_uniprot_swissprot`
    with columns: accession, sequence, category, organism.
    """
    proteins: List[UniProtProtein] = []
    path = Path(path)

    open_fn = gzip.open if str(path).endswith(".gz") else open
    with open_fn(path, "rt", encoding="utf-8") as f:  # type: ignore[call-overload]
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        if header is None:
            return proteins
        # Map columns by name
        h_lower = [h.lower().strip() for h in header]
        acc_idx = h_lower.index("accession") if "accession" in h_lower else 0
        seq_idx = h_lower.index("sequence") if "sequence" in h_lower else 1
        cat_idx = h_lower.index("category") if "category" in h_lower else 2
        org_idx = h_lower.index("organism") if "organism" in h_lower else 3

        for row in reader:
            if len(row) <= max(acc_idx, seq_idx, cat_idx):
                continue
            proteins.append(UniProtProtein(
                accession=row[acc_idx],
                sequence=row[seq_idx],
                category=row[cat_idx],
                organism=row[org_idx] if org_idx < len(row) else "",
            ))
    return proteins


# =============================================================================
# Unified Dataset
# =============================================================================

# Type alias for all record types
AnyRecord = Union[BindingRecord, KineticsRecord, StabilityRecord, ProcessingRecord,
                  ElutionRecord, TCellRecord, BCellRecord, Sc10xVDJRecord, VDJdbRecord]


class PrestoDataset(Dataset):
    """PyTorch dataset for Presto training.

    Combines all data modalities into a unified dataset.
    """

    def __init__(
        self,
        binding_records: List[BindingRecord] = None,
        kinetics_records: List[KineticsRecord] = None,
        stability_records: List[StabilityRecord] = None,
        processing_records: List[ProcessingRecord] = None,
        elution_records: List[ElutionRecord] = None,
        tcell_records: List[TCellRecord] = None,
        vdjdb_records: List[VDJdbRecord] = None,
        sc10x_records: List[Sc10xVDJRecord] = None,
        mhc_sequences: Dict[str, str] = None,
        allele_resolver: AlleleResolver = None,
        strict_mhc_resolution: bool = True,
        # Backward compatibility
        tcr_records: List[TCellRecord] = None,
    ):
        # Handle backward compatibility: tcr_records -> tcell_records
        if tcr_records is not None and tcell_records is None:
            tcell_records = tcr_records
        self.mhc_sequences = mhc_sequences or {}
        self.allele_resolver = allele_resolver
        self.strict_mhc_resolution = bool(strict_mhc_resolution)
        self._mhc_x_sequence_count = 0
        self._mhc_x_residue_total = 0
        self._mhc_x_allele_examples: List[str] = []

        # Create unified sample list
        self.samples = []

        def _preferred_chain_type(
            tcr_a_cdr3: Optional[str],
            tcr_b_cdr3: Optional[str],
            tcr_a_full: Optional[str],
            tcr_b_full: Optional[str],
            default_gene: Optional[str] = None,
        ) -> Optional[str]:
            if tcr_b_cdr3:
                return "TRB_CDR3"
            if tcr_a_cdr3:
                return "TRA_CDR3"
            if tcr_b_full:
                return "TRB"
            if tcr_a_full:
                return "TRA"
            gene = (default_gene or "").upper()
            if gene in {"TRA", "TRAD"}:
                return "TRA_CDR3"
            if gene in {"TRB", "TRBD"}:
                return "TRB_CDR3"
            return None

        def _source_label(source: Optional[str]) -> str:
            src = (source or "").strip()
            return src if src else "unknown"

        def _synthetic_kind(source: Optional[str]) -> Optional[str]:
            src = _source_label(source)
            return src if src.startswith("synthetic_negative") else None

        def _organism_fields(antigen_species: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
            """Map raw antigen_species → (species_of_origin, foreignness_label)."""
            cat = normalize_organism(antigen_species)
            if cat is None:
                return None, None
            foreign = 1.0 if cat in FOREIGN_CATEGORIES else 0.0
            return cat, foreign

        def _binary_bucket(value: Optional[float]) -> Optional[str]:
            if value is None:
                return None
            return "positive" if float(value) > 0.5 else "negative"

        def _binding_assay_group(measurement_type: Optional[str], source: Optional[str]) -> str:
            label = (measurement_type or "").strip().lower()
            if "kon" in label or "association rate" in label or "on rate" in label:
                return "binding_kon"
            if "koff" in label or "dissociation rate" in label or "off rate" in label:
                return "binding_koff"
            if "t_half" in label or "half-life" in label or "half life" in label or "t1/2" in label:
                return "binding_t_half"
            if label == "tm" or "melt" in label or "dissociation temperature" in label:
                return "binding_tm"
            if "ic50" in label:
                return "binding_ic50"
            if "ec50" in label:
                return "binding_ec50"
            if "kd" in label:
                return "binding_kd"
            return "binding_affinity"

        # Add binding samples
        for rec in (binding_records or []):
            mhc_class = self._resolve_mhc_class_value(rec.mhc_class, rec.mhc_allele)
            src = _source_label(rec.source)
            is_no_mhc_alpha = src == "synthetic_negative_no_mhc_alpha"
            is_no_mhc_beta = src in {"synthetic_negative_no_mhc_beta", "synthetic_negative_no_b2m"}
            mhc_seq = "" if is_no_mhc_alpha else self._get_mhc_sequence(rec.mhc_allele, rec.mhc_sequence)
            mhc_b_seq = self._resolve_mhc_b_sequence(
                mhc_class=mhc_class,
                species=rec.species,
                allele=rec.mhc_allele,
                allow_default_class_i_beta=not is_no_mhc_beta,
            )
            is_synth = src.startswith("synthetic_negative")
            bind_label = "negative" if is_synth or float(rec.value) >= 50000.0 else "positive"
            so, fl = _organism_fields(rec.antigen_species)
            self.samples.append(PrestoSample(
                peptide=rec.peptide,
                mhc_a=mhc_seq,
                mhc_b=mhc_b_seq,
                mhc_class=mhc_class,
                bind_value=rec.value,
                bind_qual=rec.qualifier,
                bind_measurement_type=rec.measurement_type,
                species=rec.species,
                species_of_origin=so,
                foreignness_label=fl,
                sample_source=src,
                assay_group=_binding_assay_group(rec.measurement_type, rec.source),
                label_bucket=bind_label,
                primary_allele=rec.mhc_allele,
                synthetic_kind=_synthetic_kind(rec.source),
                sample_id=f"bind_{len(self.samples)}",
            ))

        # Add kinetics samples
        for rec in (kinetics_records or []):
            mhc_class = self._resolve_mhc_class_value(rec.mhc_class, rec.mhc_allele)
            mhc_seq = self._get_mhc_sequence(rec.mhc_allele, rec.mhc_sequence)
            mhc_b_seq = self._resolve_mhc_b_sequence(
                mhc_class=mhc_class,
                species=rec.species,
                allele=rec.mhc_allele,
            )
            so, fl = _organism_fields(rec.antigen_species)
            self.samples.append(PrestoSample(
                peptide=rec.peptide,
                mhc_a=mhc_seq,
                mhc_b=mhc_b_seq,
                mhc_class=mhc_class,
                kon=rec.kon,
                koff=rec.koff,
                species=rec.species,
                species_of_origin=so,
                foreignness_label=fl,
                sample_source=_source_label(rec.source),
                assay_group="binding_kinetics",
                label_bucket=None,
                primary_allele=rec.mhc_allele,
                synthetic_kind=_synthetic_kind(rec.source),
                sample_id=f"kin_{len(self.samples)}",
            ))

        # Add stability samples
        for rec in (stability_records or []):
            mhc_class = self._resolve_mhc_class_value(rec.mhc_class, rec.mhc_allele)
            mhc_seq = self._get_mhc_sequence(rec.mhc_allele, rec.mhc_sequence)
            mhc_b_seq = self._resolve_mhc_b_sequence(
                mhc_class=mhc_class,
                species=rec.species,
                allele=rec.mhc_allele,
            )
            so, fl = _organism_fields(rec.antigen_species)
            self.samples.append(PrestoSample(
                peptide=rec.peptide,
                mhc_a=mhc_seq,
                mhc_b=mhc_b_seq,
                mhc_class=mhc_class,
                t_half=rec.t_half,
                tm=rec.tm,
                species=rec.species,
                species_of_origin=so,
                foreignness_label=fl,
                sample_source=_source_label(rec.source),
                assay_group="binding_stability",
                label_bucket=None,
                primary_allele=rec.mhc_allele,
                synthetic_kind=_synthetic_kind(rec.source),
                sample_id=f"stab_{len(self.samples)}",
            ))

        # Add processing samples
        for rec in (processing_records or []):
            mhc_class = self._resolve_mhc_class_value(rec.mhc_class, rec.mhc_allele)
            mhc_seq = self._get_mhc_sequence(rec.mhc_allele, None) if rec.mhc_allele else ""
            mhc_b_seq = self._resolve_mhc_b_sequence(
                mhc_class=mhc_class,
                species=rec.species,
                allele=rec.mhc_allele,
            )
            so, fl = _organism_fields(rec.antigen_species)
            self.samples.append(PrestoSample(
                peptide=rec.peptide,
                flank_n=rec.flank_n,
                flank_c=rec.flank_c,
                mhc_a=mhc_seq,
                mhc_b=mhc_b_seq,
                mhc_class=mhc_class,
                processing_label=rec.label,
                species=rec.species,
                species_of_origin=so,
                foreignness_label=fl,
                sample_source=_source_label(rec.source),
                assay_group="processing",
                label_bucket=_binary_bucket(rec.label),
                primary_allele=rec.mhc_allele,
                synthetic_kind=_synthetic_kind(rec.source),
                sample_id=f"proc_{len(self.samples)}",
            ))

        # Add elution samples
        for rec in (elution_records or []):
            mil_mhc_a_list: List[str] = []
            mil_mhc_b_list: List[str] = []
            mil_mhc_class_list: List[str] = []
            mil_species_list: List[str] = []

            alleles = rec.alleles if rec.alleles else [None]
            for allele in alleles:
                mhc_class_i = self._resolve_mhc_class_value(rec.mhc_class, allele)
                mhc_seq_i = self._get_mhc_sequence(allele, None) if allele else ""
                mhc_b_seq_i = self._resolve_mhc_b_sequence(
                    mhc_class=mhc_class_i,
                    species=rec.species,
                    allele=allele,
                )
                mil_mhc_a_list.append(mhc_seq_i)
                mil_mhc_b_list.append(mhc_b_seq_i)
                mil_mhc_class_list.append(mhc_class_i)
                mil_species_list.append(rec.species)

            mhc_seq = mil_mhc_a_list[0] if mil_mhc_a_list else ""
            mhc_b_seq = mil_mhc_b_list[0] if mil_mhc_b_list else ""
            mhc_class = mil_mhc_class_list[0] if mil_mhc_class_list else "I"
            so, fl = _organism_fields(rec.antigen_species)
            self.samples.append(PrestoSample(
                peptide=rec.peptide,
                mhc_a=mhc_seq,
                mhc_b=mhc_b_seq,
                mhc_class=mhc_class,
                elution_label=1.0 if rec.detected else 0.0,
                mil_mhc_a_list=mil_mhc_a_list,
                mil_mhc_b_list=mil_mhc_b_list,
                mil_mhc_class_list=mil_mhc_class_list,
                mil_species_list=mil_species_list,
                species=rec.species,
                species_of_origin=so,
                foreignness_label=fl,
                sample_source=_source_label(rec.source),
                assay_group="elution_ms",
                label_bucket="positive" if rec.detected else "negative",
                primary_allele=(alleles[0] if alleles else None),
                synthetic_kind=_synthetic_kind(rec.source),
                sample_id=f"elut_{len(self.samples)}",
            ))

        # Add T-cell samples
        for rec in (tcell_records or []):
            mhc_class = self._resolve_mhc_class_value(rec.mhc_class, rec.mhc_allele)
            mhc_seq = self._get_mhc_sequence(rec.mhc_allele, rec.mhc_sequence)
            mhc_b_seq = self._resolve_mhc_b_sequence(
                mhc_class=mhc_class,
                species=rec.species,
                allele=rec.mhc_allele,
            )
            chain_type = _preferred_chain_type(
                rec.tcr_a_cdr3,
                rec.tcr_b_cdr3,
                rec.tcr_a_full,
                rec.tcr_b_full,
            )
            phenotype = "ab_T" if chain_type is not None else None
            so, fl = _organism_fields(rec.antigen_species)
            self.samples.append(PrestoSample(
                peptide=rec.peptide,
                mhc_a=mhc_seq,
                mhc_b=mhc_b_seq,
                mhc_class=mhc_class,
                tcr_a=rec.tcr_a_cdr3 or rec.tcr_a_full,
                tcr_b=rec.tcr_b_cdr3 or rec.tcr_b_full,
                tcell_label=rec.response,
                tcell_assay_method=rec.assay_method,
                tcell_assay_readout=rec.assay_type,
                tcell_apc_name=rec.apc_name,
                tcell_effector_culture=rec.effector_culture_condition,
                tcell_apc_culture=rec.apc_culture_condition,
                tcell_in_vitro_process=rec.in_vitro_process_type,
                tcell_in_vitro_responder=rec.in_vitro_responder_cell,
                tcell_in_vitro_stimulator=rec.in_vitro_stimulator_cell,
                chain_type=chain_type,
                phenotype=phenotype,
                species=rec.species,
                species_of_origin=so,
                foreignness_label=fl,
                sample_source=_source_label(rec.source),
                assay_group="tcell_response",
                label_bucket=_binary_bucket(rec.response),
                primary_allele=rec.mhc_allele,
                synthetic_kind=_synthetic_kind(rec.source),
                sample_id=f"tcell_{len(self.samples)}",
            ))

        # Add VDJdb samples
        for rec in (vdjdb_records or []):
            mhc_class = self._resolve_mhc_class_value(rec.mhc_class, rec.mhc_a)
            mhc_seq = self._get_mhc_sequence(rec.mhc_a, None)
            mhc_b_seq = self._resolve_mhc_b_sequence(
                mhc_class=mhc_class,
                species=rec.species,
                allele=rec.mhc_a,
                mhc_b=rec.mhc_b,
            )
            chain_type = _preferred_chain_type(
                rec.cdr3_alpha,
                rec.cdr3_beta,
                None,
                None,
                default_gene=rec.gene,
            )
            phenotype = "ab_T" if chain_type is not None else None
            so, fl = _organism_fields(rec.antigen_species)
            self.samples.append(PrestoSample(
                peptide=rec.peptide,
                mhc_a=mhc_seq,
                mhc_b=mhc_b_seq,
                mhc_class=mhc_class,
                tcr_a=rec.cdr3_alpha,
                tcr_b=rec.cdr3_beta,
                tcell_label=1.0,  # VDJdb entries are positive by definition
                chain_type=chain_type,
                phenotype=phenotype,
                species=rec.species,
                species_of_origin=so,
                foreignness_label=fl,
                sample_source=_source_label(rec.source),
                assay_group="tcr_pmhc",
                label_bucket="positive",
                primary_allele=rec.mhc_a,
                synthetic_kind=_synthetic_kind(rec.source),
                sample_id=f"vdjdb_{len(self.samples)}",
            ))

        # Add 10x VDJ chain-classification samples (chain/species/phenotype supervision).
        for rec in (sc10x_records or []):
            chain_type = (rec.chain_type or "").strip().upper()
            if not chain_type:
                continue
            seq = (rec.cdr3 or "").strip().upper()
            if not seq:
                continue
            phenotype = rec.phenotype or _infer_receptor_phenotype(chain_type)
            chain_type_label = f"{chain_type}_CDR3" if not chain_type.endswith("_CDR3") else chain_type

            # Use minimally valid placeholders for pMHC inputs; these samples
            # primarily supervise chain heads.
            self.samples.append(PrestoSample(
                peptide="A",
                mhc_a="",
                mhc_b="",
                mhc_class="",
                tcr_a=seq if chain_type in {"TRA", "TRG", "IGK", "IGL"} else None,
                tcr_b=seq if chain_type in {"TRB", "TRD", "IGH"} else None,
                chain_type=chain_type_label,
                phenotype=phenotype,
                species=rec.species,
                sample_source=_source_label(rec.source),
                assay_group="chain_aux",
                label_bucket=None,
                primary_allele=None,
                synthetic_kind=_synthetic_kind(rec.source),
                sample_id=f"10x_{len(self.samples)}",
            ))

        if self._mhc_x_sequence_count > 0:
            example_text = ", ".join(self._mhc_x_allele_examples[:5]) or "(unlabeled)"
            warnings.warn(
                "Detected ambiguous residue 'X' in loaded MHC sequences: "
                f"sequences={self._mhc_x_sequence_count}, residues={self._mhc_x_residue_total}, "
                f"examples={example_text}",
                RuntimeWarning,
            )

    @staticmethod
    def _resolve_mhc_class_value(
        mhc_class: Optional[str],
        allele: Optional[str],
    ) -> str:
        """Resolve canonical MHC class label."""
        resolved = normalize_mhc_class(mhc_class)
        if resolved is not None:
            return resolved
        if allele:
            inferred = _infer_mhc_class(allele)
            if inferred is not None:
                return inferred
        return "I"

    def _default_class_i_beta2m(
        self,
        species: Optional[str],
        allele: Optional[str],
    ) -> str:
        """Return species-aware class-I beta2m with a safe fallback."""
        resolved_species = normalize_species_label(species)
        if resolved_species is None and allele:
            resolved_species = normalize_species_label(infer_species_from_allele(allele))
        return class_i_beta2m_sequence(resolved_species) or HUMAN_B2M_SEQUENCE

    def _resolve_mhc_b_sequence(
        self,
        mhc_class: Optional[str],
        species: Optional[str],
        allele: Optional[str],
        mhc_b: Optional[str] = None,
        allow_default_class_i_beta: bool = True,
    ) -> str:
        """Resolve MHC beta chain sequence for class-I/class-II samples."""
        cls = self._resolve_mhc_class_value(mhc_class, allele)

        if mhc_b:
            mhc_b_clean = str(mhc_b).strip()
            if mhc_b_clean:
                if mhc_b_clean.upper() in {
                    "B2M",
                    "BETA2M",
                    "BETA-2-MICROGLOBULIN",
                    "BETA2-MICROGLOBULIN",
                }:
                    seq = self._default_class_i_beta2m(species, allele)
                    return self._validate_mhc_chain_sequence(
                        sequence=seq,
                        chain_label="mhc_b",
                        mhc_class=cls,
                        allele=allele,
                        allow_short_class_i_beta=True,
                    )
                return self._validate_mhc_chain_sequence(
                    sequence=mhc_b_clean,
                    chain_label="mhc_b",
                    mhc_class=cls,
                    allele=allele,
                    allow_short_class_i_beta=(cls == "I"),
                )

        if cls == "I" and allow_default_class_i_beta:
            seq = self._default_class_i_beta2m(species, allele)
            return self._validate_mhc_chain_sequence(
                sequence=seq,
                chain_label="mhc_b",
                mhc_class=cls,
                allele=allele,
                allow_short_class_i_beta=True,
            )
        return ""

    def _get_mhc_sequence(self, allele: str, direct_seq: Optional[str]) -> str:
        """Resolve MHC sequence from allele name or direct sequence."""
        if direct_seq:
            return self._validate_mhc_chain_sequence(
                sequence=direct_seq,
                chain_label="mhc_a",
                mhc_class=None,
                allele=allele,
                allow_short_class_i_beta=False,
            )
        if allele and allele in self.mhc_sequences:
            return self._validate_mhc_chain_sequence(
                sequence=self.mhc_sequences[allele],
                chain_label="mhc_a",
                mhc_class=None,
                allele=allele,
                allow_short_class_i_beta=False,
            )
        if allele and self.allele_resolver:
            seq = self.allele_resolver.get_sequence(allele)
            if seq:
                return self._validate_mhc_chain_sequence(
                    sequence=seq,
                    chain_label="mhc_a",
                    mhc_class=None,
                    allele=allele,
                    allow_short_class_i_beta=False,
                )
        if allele and self.strict_mhc_resolution:
            raise ValueError(
                "Unresolved MHC allele without sequence: "
                f"{allele}. Provide explicit sequence via index/record and "
                "do not rely on allele-string fallback."
            )
        return ""

    def _validate_mhc_chain_sequence(
        self,
        *,
        sequence: Optional[str],
        chain_label: str,
        mhc_class: Optional[str],
        allele: Optional[str],
        allow_short_class_i_beta: bool,
    ) -> str:
        """Validate loaded MHC chain sequence quality and biologic length constraints."""
        seq = str(sequence or "").strip().upper()
        if not seq:
            return ""

        bad = sorted({ch for ch in seq if ch not in MHC_ALLOWED_AA})
        if bad:
            allele_text = str(allele or "").strip() or "<unknown>"
            raise ValueError(
                "Non-canonical residue(s) in MHC sequence "
                f"(allele={allele_text}, chain={chain_label}): {''.join(bad)}"
            )

        if "X" in seq:
            self._mhc_x_sequence_count += 1
            self._mhc_x_residue_total += int(seq.count("X"))
            allele_text = str(allele or "").strip()
            if allele_text and allele_text not in self._mhc_x_allele_examples and len(self._mhc_x_allele_examples) < 8:
                self._mhc_x_allele_examples.append(allele_text)

        if len(seq) >= MIN_MHC_CHAIN_LENGTH and _looks_like_nucleotide_sequence(seq):
            allele_text = str(allele or "").strip() or "<unknown>"
            raise ValueError(
                "Likely nucleotide sequence loaded for MHC chain: "
                f"allele={allele_text}, chain={chain_label}, len={len(seq)}"
            )

        if len(seq) < MIN_MHC_CHAIN_LENGTH:
            normalized_class = normalize_mhc_class(mhc_class)
            is_class_i_beta = (
                allow_short_class_i_beta
                and chain_label == "mhc_b"
                and normalized_class == "I"
            )
            if not is_class_i_beta:
                allele_text = str(allele or "").strip() or "<unknown>"
                if self.strict_mhc_resolution:
                    raise ValueError(
                        "MHC chain shorter than minimum accepted groove-bearing fragment: "
                        f"allele={allele_text}, chain={chain_label}, len={len(seq)}"
                    )
                # Non-strict mode: accept the short sequence with a warning
                # (logged once per allele via _mhc_x_allele_examples)
                return seq

        return seq

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> PrestoSample:
        return self.samples[idx]


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    collator: PrestoCollator = None,
    balanced: bool = False,
    seed: int = 42,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader for Presto training."""
    collator = collator or PrestoCollator()
    num_workers = max(int(num_workers), 0)
    loader_kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": collator,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    if balanced and shuffle:
        batch_sampler = BalancedMiniBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        **loader_kwargs,
    )


class BalancedMiniBatchSampler(Sampler[List[int]]):
    """Balanced mini-batch sampler for multi-assay unified training.

    Guarantees per-batch task mixing and encourages balance across source,
    label polarity, MHC allele identity, and synthetic-negative types.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
        seed: int = 42,
        max_candidates_per_draw: int = 2048,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.max_candidates_per_draw = max(16, int(max_candidates_per_draw))
        self._epoch = 0

        self._all_indices = list(range(len(dataset)))
        self._task_to_indices: Dict[str, List[int]] = defaultdict(list)
        self._task_label_to_indices: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._index_weight: Dict[int, float] = {}
        self._metadata_by_index: Dict[int, tuple[str, str, str, str, str]] = {}

        task_counts: Dict[str, int] = defaultdict(int)
        source_counts: Dict[str, int] = defaultdict(int)
        label_counts: Dict[str, int] = defaultdict(int)
        allele_counts: Dict[str, int] = defaultdict(int)
        synthetic_counts: Dict[str, int] = defaultdict(int)
        for idx in self._all_indices:
            sample = dataset[idx]
            task = self._sample_task_group(sample)
            source = self._sample_source(sample)
            label = self._sample_label_bucket(sample)
            allele = self._sample_primary_allele(sample)
            synthetic = self._sample_synthetic_kind(sample)

            self._metadata_by_index[idx] = (task, source, label, allele, synthetic)
            self._task_to_indices[task].append(idx)
            self._task_label_to_indices[task][label].append(idx)

            task_counts[task] += 1
            source_counts[source] += 1
            label_counts[label] += 1
            allele_counts[allele] += 1
            synthetic_counts[synthetic] += 1

        for idx, (task, source, label, allele, synthetic) in self._metadata_by_index.items():
            # Product of inverse frequencies to upweight underrepresented strata.
            weight = 1.0
            weight *= 1.0 / float(task_counts[task] + 1)
            weight *= 1.0 / float(source_counts[source] + 1)
            weight *= 1.0 / float(label_counts[label] + 1)
            weight *= 1.0 / float(allele_counts[allele] + 1)
            weight *= 1.0 / float(synthetic_counts[synthetic] + 1)
            self._index_weight[idx] = weight

        self._tasks = sorted(self._task_to_indices.keys())
        if not self._tasks:
            raise ValueError("Cannot build balanced sampler: dataset is empty")
        self._task_sizes = {task: len(indices) for task, indices in self._task_to_indices.items()}

    @staticmethod
    def _sample_task_group(sample: PrestoSample) -> str:
        if sample.assay_group:
            return sample.assay_group
        if (sample.sample_source or "").strip() == "mhc_augmentation":
            return "mhc_aux"
        if sample.bind_value is not None:
            return "binding_affinity"
        if sample.kon is not None or sample.koff is not None:
            return "binding_kinetics"
        if sample.t_half is not None or sample.tm is not None:
            return "binding_stability"
        if sample.processing_label is not None:
            return "processing"
        if sample.elution_label is not None:
            return "elution_ms"
        if sample.tcell_label is not None:
            return "tcell_response"
        if sample.chain_type is not None or sample.phenotype is not None:
            return "chain_aux"
        return "other"

    @staticmethod
    def _sample_source(sample: PrestoSample) -> str:
        source = (sample.sample_source or "").strip()
        return source if source else "unknown"

    @staticmethod
    def _sample_label_bucket(sample: PrestoSample) -> str:
        if sample.label_bucket:
            return sample.label_bucket
        if sample.elution_label is not None:
            return "positive" if float(sample.elution_label) > 0.5 else "negative"
        if sample.tcell_label is not None:
            return "positive" if float(sample.tcell_label) > 0.5 else "negative"
        if sample.processing_label is not None:
            return "positive" if float(sample.processing_label) > 0.5 else "negative"
        return "unknown"

    @staticmethod
    def _sample_primary_allele(sample: PrestoSample) -> str:
        allele = (sample.primary_allele or "").strip()
        if allele:
            return allele
        if sample.mhc_a:
            return sample.mhc_a[:24]
        return "unknown_allele"

    @staticmethod
    def _sample_synthetic_kind(sample: PrestoSample) -> str:
        kind = (sample.synthetic_kind or "").strip()
        return kind if kind else "none"

    def __len__(self) -> int:
        n = len(self._all_indices)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def _weighted_choice(self, candidates: List[int], rng: random.Random) -> int:
        if len(candidates) == 1:
            return candidates[0]
        total = 0.0
        for idx in candidates:
            total += self._index_weight.get(idx, 1.0)
        if total <= 0:
            return rng.choice(candidates)
        needle = rng.random() * total
        running = 0.0
        for idx in candidates:
            running += self._index_weight.get(idx, 1.0)
            if running >= needle:
                return idx
        return candidates[-1]

    def _batch_balanced_choice(
        self,
        candidates: List[int],
        rng: random.Random,
        batch_source_counts: Dict[str, int],
        batch_label_counts: Dict[str, int],
        batch_allele_counts: Dict[str, int],
        batch_synthetic_counts: Dict[str, int],
    ) -> int:
        """Sample one index while actively balancing batch-level strata."""
        if len(candidates) == 1:
            return candidates[0]

        total = 0.0
        weighted: List[Tuple[int, float]] = []
        for idx in candidates:
            _, source, label, allele, synthetic = self._metadata_by_index[idx]
            weight = self._index_weight.get(idx, 1.0)
            weight *= 1.0 / float(batch_source_counts.get(source, 0) + 1)
            weight *= 1.0 / float(batch_label_counts.get(label, 0) + 1)
            weight *= 1.0 / float(batch_allele_counts.get(allele, 0) + 1)
            weight *= 1.0 / float(batch_synthetic_counts.get(synthetic, 0) + 1)
            weight = max(weight, 1e-12)
            weighted.append((idx, weight))
            total += weight

        if total <= 0:
            return self._weighted_choice(candidates, rng)

        needle = rng.random() * total
        running = 0.0
        for idx, weight in weighted:
            running += weight
            if running >= needle:
                return idx
        return weighted[-1][0]

    def _draw_from_task(
        self,
        task: str,
        rng: random.Random,
        task_label_cursor: Dict[str, int],
        in_batch: set[int],
        batch_source_counts: Dict[str, int],
        batch_label_counts: Dict[str, int],
        batch_allele_counts: Dict[str, int],
        batch_synthetic_counts: Dict[str, int],
    ) -> int:
        label_pools = self._task_label_to_indices[task]
        preferred_labels = [label for label in ("positive", "negative") if label in label_pools]
        if preferred_labels:
            cursor = task_label_cursor[task]
            label = preferred_labels[cursor % len(preferred_labels)]
            task_label_cursor[task] = cursor + 1
            candidates = label_pools[label]
        else:
            labels = sorted(label_pools.keys())
            cursor = task_label_cursor[task]
            label = labels[cursor % len(labels)]
            task_label_cursor[task] = cursor + 1
            candidates = label_pools[label]

        choice_pool = self._choose_candidate_pool(
            candidates=candidates,
            in_batch=in_batch,
            rng=rng,
        )

        return self._batch_balanced_choice(
            choice_pool,
            rng,
            batch_source_counts,
            batch_label_counts,
            batch_allele_counts,
            batch_synthetic_counts,
        )

    def _choose_candidate_pool(
        self,
        candidates: List[int],
        in_batch: set[int],
        rng: random.Random,
    ) -> List[int]:
        """Bound per-draw candidate set to avoid scanning huge pools each sample."""
        n = len(candidates)
        if n <= self.max_candidates_per_draw:
            filtered = [idx for idx in candidates if idx not in in_batch]
            return filtered or candidates

        target = self.max_candidates_per_draw
        sampled: List[int] = []
        seen: set[int] = set()
        attempts = 0
        max_attempts = target * 4
        while len(sampled) < target and attempts < max_attempts:
            idx = candidates[rng.randrange(n)]
            attempts += 1
            if idx in seen or idx in in_batch:
                continue
            seen.add(idx)
            sampled.append(idx)

        if sampled:
            return sampled

        # Fallback if all random draws hit already-selected batch members.
        quick = rng.sample(candidates, k=min(target, n))
        filtered = [idx for idx in quick if idx not in in_batch]
        return filtered or quick

    def _task_quotas(self, tasks: List[str], rng: random.Random) -> Dict[str, int]:
        """Allocate per-task batch quotas proportionally with min-1 per task."""
        if not tasks:
            return {}

        # When there are more tasks than batch slots, sample tasks by prevalence.
        if len(tasks) > self.batch_size:
            remaining_tasks = list(tasks)
            remaining_weights = [float(self._task_sizes.get(task, 1)) for task in remaining_tasks]
            selected: List[str] = []
            for _ in range(self.batch_size):
                if not remaining_tasks:
                    break
                total = sum(remaining_weights)
                if total <= 0:
                    pick_idx = rng.randrange(len(remaining_tasks))
                else:
                    needle = rng.random() * total
                    running = 0.0
                    pick_idx = len(remaining_tasks) - 1
                    for idx, weight in enumerate(remaining_weights):
                        running += weight
                        if running >= needle:
                            pick_idx = idx
                            break
                selected.append(remaining_tasks.pop(pick_idx))
                remaining_weights.pop(pick_idx)
            return {task: 1 for task in selected}

        quotas = {task: 1 for task in tasks}
        remaining_slots = self.batch_size - len(tasks)
        if remaining_slots <= 0:
            return quotas

        total_task_samples = sum(max(1, self._task_sizes.get(task, 0)) for task in tasks)
        if total_task_samples <= 0:
            task_order = list(tasks)
            rng.shuffle(task_order)
            for idx in range(remaining_slots):
                quotas[task_order[idx % len(task_order)]] += 1
            return quotas

        shuffled_tasks = list(tasks)
        rng.shuffle(shuffled_tasks)
        fractional: List[Tuple[float, str]] = []
        assigned = 0
        for task in shuffled_tasks:
            raw = remaining_slots * (max(1, self._task_sizes.get(task, 0)) / float(total_task_samples))
            extra = int(raw)
            quotas[task] += extra
            assigned += extra
            fractional.append((raw - float(extra), task))

        leftovers = max(0, remaining_slots - assigned)
        fractional.sort(key=lambda item: item[0], reverse=True)
        if fractional:
            for idx in range(leftovers):
                quotas[fractional[idx % len(fractional)][1]] += 1
        return quotas

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        n_batches = len(self)
        tasks = list(self._tasks)

        for _ in range(n_batches):
            batch: List[int] = []
            in_batch: set[int] = set()
            task_label_cursor: Dict[str, int] = defaultdict(int)
            batch_source_counts: Dict[str, int] = defaultdict(int)
            batch_label_counts: Dict[str, int] = defaultdict(int)
            batch_allele_counts: Dict[str, int] = defaultdict(int)
            batch_synthetic_counts: Dict[str, int] = defaultdict(int)

            quotas = self._task_quotas(tasks, rng)

            for task, task_quota in quotas.items():
                for _ in range(task_quota):
                    idx = self._draw_from_task(
                        task,
                        rng,
                        task_label_cursor,
                        in_batch,
                        batch_source_counts,
                        batch_label_counts,
                        batch_allele_counts,
                        batch_synthetic_counts,
                    )
                    batch.append(idx)
                    in_batch.add(idx)
                    _, source, label, allele, synthetic = self._metadata_by_index[idx]
                    batch_source_counts[source] += 1
                    batch_label_counts[label] += 1
                    batch_allele_counts[allele] += 1
                    batch_synthetic_counts[synthetic] += 1

            rng.shuffle(batch)
            if self.drop_last and len(batch) < self.batch_size:
                continue
            if not self.drop_last and len(batch) < self.batch_size:
                while len(batch) < self.batch_size:
                    filler_pool = [idx for idx in self._all_indices if idx not in in_batch]
                    if not filler_pool:
                        filler_pool = self._all_indices
                    idx = self._batch_balanced_choice(
                        filler_pool,
                        rng,
                        batch_source_counts,
                        batch_label_counts,
                        batch_allele_counts,
                        batch_synthetic_counts,
                    )
                    batch.append(idx)
                    in_batch.add(idx)
                    _, source, label, allele, synthetic = self._metadata_by_index[idx]
                    batch_source_counts[source] += 1
                    batch_label_counts[label] += 1
                    batch_allele_counts[allele] += 1
                    batch_synthetic_counts[synthetic] += 1
            yield batch


# =============================================================================
# Synthetic Data Generation (Complete and Realistic)
# =============================================================================

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Realistic allele pools
CLASS_I_ALLELES = [
    "HLA-A*01:01", "HLA-A*02:01", "HLA-A*02:03", "HLA-A*02:06", "HLA-A*03:01",
    "HLA-A*11:01", "HLA-A*23:01", "HLA-A*24:02", "HLA-A*26:01", "HLA-A*29:02",
    "HLA-A*30:01", "HLA-A*30:02", "HLA-A*31:01", "HLA-A*32:01", "HLA-A*33:01",
    "HLA-A*68:01", "HLA-A*68:02",
    "HLA-B*07:02", "HLA-B*08:01", "HLA-B*13:02", "HLA-B*14:02", "HLA-B*15:01",
    "HLA-B*18:01", "HLA-B*27:05", "HLA-B*35:01", "HLA-B*38:01", "HLA-B*39:01",
    "HLA-B*40:01", "HLA-B*44:02", "HLA-B*44:03", "HLA-B*45:01", "HLA-B*46:01",
    "HLA-B*49:01", "HLA-B*50:01", "HLA-B*51:01", "HLA-B*52:01", "HLA-B*53:01",
    "HLA-B*55:01", "HLA-B*56:01", "HLA-B*57:01", "HLA-B*58:01",
    "HLA-C*01:02", "HLA-C*02:02", "HLA-C*03:03", "HLA-C*03:04", "HLA-C*04:01",
    "HLA-C*05:01", "HLA-C*06:02", "HLA-C*07:01", "HLA-C*07:02", "HLA-C*08:02",
    "HLA-C*12:03", "HLA-C*14:02", "HLA-C*15:02", "HLA-C*16:01",
]

CLASS_II_ALLELES = [
    "HLA-DRB1*01:01", "HLA-DRB1*03:01", "HLA-DRB1*04:01", "HLA-DRB1*04:05",
    "HLA-DRB1*07:01", "HLA-DRB1*08:02", "HLA-DRB1*09:01", "HLA-DRB1*10:01",
    "HLA-DRB1*11:01", "HLA-DRB1*12:01", "HLA-DRB1*13:02", "HLA-DRB1*14:01",
    "HLA-DRB1*15:01", "HLA-DRB1*16:02",
    "HLA-DPA1*01:03/DPB1*02:01", "HLA-DPA1*01:03/DPB1*04:01",
    "HLA-DPA1*02:01/DPB1*01:01", "HLA-DPA1*02:01/DPB1*05:01",
    "HLA-DQA1*01:02/DQB1*06:02", "HLA-DQA1*05:01/DQB1*02:01",
    "HLA-DQA1*05:01/DQB1*03:01", "HLA-DQA1*03:01/DQB1*03:02",
]

# V gene families
V_ALPHA_GENES = ["TRAV1-1", "TRAV1-2", "TRAV2", "TRAV3", "TRAV4", "TRAV5",
                 "TRAV6", "TRAV8-1", "TRAV8-2", "TRAV8-3", "TRAV9-1", "TRAV9-2",
                 "TRAV10", "TRAV12-1", "TRAV12-2", "TRAV12-3", "TRAV13-1",
                 "TRAV14/DV4", "TRAV16", "TRAV17", "TRAV19", "TRAV20", "TRAV21",
                 "TRAV22", "TRAV23/DV6", "TRAV24", "TRAV25", "TRAV26-1", "TRAV26-2",
                 "TRAV27", "TRAV29/DV5", "TRAV30", "TRAV34", "TRAV35", "TRAV36/DV7",
                 "TRAV38-1", "TRAV38-2/DV8", "TRAV39", "TRAV40", "TRAV41"]

V_BETA_GENES = ["TRBV2", "TRBV3-1", "TRBV4-1", "TRBV4-2", "TRBV4-3", "TRBV5-1",
                "TRBV5-4", "TRBV5-5", "TRBV5-6", "TRBV5-8", "TRBV6-1", "TRBV6-2",
                "TRBV6-3", "TRBV6-4", "TRBV6-5", "TRBV6-6", "TRBV6-8", "TRBV6-9",
                "TRBV7-2", "TRBV7-3", "TRBV7-4", "TRBV7-6", "TRBV7-7", "TRBV7-8",
                "TRBV7-9", "TRBV9", "TRBV10-1", "TRBV10-2", "TRBV10-3", "TRBV11-1",
                "TRBV11-2", "TRBV11-3", "TRBV12-3", "TRBV12-4", "TRBV12-5",
                "TRBV13", "TRBV14", "TRBV15", "TRBV16", "TRBV18", "TRBV19",
                "TRBV20-1", "TRBV24-1", "TRBV25-1", "TRBV27", "TRBV28", "TRBV29-1",
                "TRBV30"]

J_ALPHA_GENES = [f"TRAJ{i}" for i in range(1, 62)]
J_BETA_GENES = ["TRBJ1-1", "TRBJ1-2", "TRBJ1-3", "TRBJ1-4", "TRBJ1-5", "TRBJ1-6",
                "TRBJ2-1", "TRBJ2-2", "TRBJ2-3", "TRBJ2-4", "TRBJ2-5", "TRBJ2-6", "TRBJ2-7"]

# Cell types and tissues
CELL_TYPES = ["PBMC", "B-LCL", "DC", "Monocyte", "CD8+ T cell", "CD4+ T cell",
              "A375", "HeLa", "K562", "JY", "C1R", "T2"]

TISSUES = ["blood", "tumor", "thymus", "spleen", "lymph node", "lung", "liver"]

# Assay types
BINDING_ASSAYS = ["competitive binding", "fluorescence polarization",
                  "purified MHC/direct", "cellular MHC/competitive"]
TCELL_ASSAYS = ["IFNg release", "ELISPOT", "51Cr release", "cytotoxicity",
                "proliferation", "intracellular cytokine staining",
                "tetramer staining", "multimer staining"]


def random_peptide(min_len: int = 8, max_len: int = 15) -> str:
    """Generate a random peptide sequence."""
    length = random.randint(min_len, max_len)
    return ''.join(random.choices(AMINO_ACIDS, k=length))


def random_mhc_sequence(length: int = 275) -> str:
    """Generate a random MHC-like sequence."""
    return ''.join(random.choices(AMINO_ACIDS, k=length))


def random_tcr_cdr3(chain: str = "beta", min_len: int = 10, max_len: int = 18) -> str:
    """Generate a realistic TCR CDR3 sequence.

    Alpha CDR3s typically: C...F (TRAJ genes end in F)
    Beta CDR3s typically: C...F (TRBJ genes end in F)
    """
    length = random.randint(min_len, max_len)
    middle = ''.join(random.choices(AMINO_ACIDS, k=length - 2))
    return f"C{middle}F"


def random_flank(length: int = 10) -> str:
    """Generate random flanking sequence."""
    return ''.join(random.choices(AMINO_ACIDS, k=length))


def generate_synthetic_binding_data(
    n_samples: int = 100,
    alleles: List[str] = None,
    include_class_ii: bool = True,
) -> List[BindingRecord]:
    """Generate realistic synthetic binding data."""
    if alleles is None:
        alleles = CLASS_I_ALLELES[:10]
        if include_class_ii:
            alleles += CLASS_II_ALLELES[:5]

    records = []
    for _ in range(n_samples):
        allele = random.choice(alleles)
        mhc_class = "II" if "DR" in allele or "DP" in allele or "DQ" in allele else "I"

        # Class I: 8-11mer, Class II: 12-25mer
        if mhc_class == "I":
            pep_len = random.randint(8, 11)
        else:
            pep_len = random.randint(12, 20)

        # Realistic IC50 distribution (log-normal)
        log_ic50 = random.gauss(3.0, 1.5)  # Mean ~1000 nM
        ic50 = 10 ** max(0, min(6, log_ic50))  # Clamp to 1-1,000,000 nM

        # More common to have '=' qualifier
        qual_weights = [0.15, 0.7, 0.15]
        qualifier = random.choices([-1, 0, 1], weights=qual_weights)[0]

        records.append(BindingRecord(
            peptide=random_peptide(pep_len, pep_len),
            mhc_allele=allele,
            value=ic50,
            qualifier=qualifier,
            measurement_type=random.choice(["IC50", "KD", "EC50"]),
            unit="nM",
            assay_type=random.choice(BINDING_ASSAYS),
            mhc_class=mhc_class,
            species="human",
            source="synthetic",
        ))

    return records


def generate_synthetic_kinetics_data(
    n_samples: int = 50,
    alleles: List[str] = None,
) -> List[KineticsRecord]:
    """Generate realistic synthetic kinetics data."""
    alleles = alleles or CLASS_I_ALLELES[:10]

    records = []
    for _ in range(n_samples):
        allele = random.choice(alleles)
        mhc_class = "II" if "DR" in allele or "DP" in allele or "DQ" in allele else "I"
        pep_len = random.randint(8, 11) if mhc_class == "I" else random.randint(12, 20)

        # Typical kon: 10^3 - 10^6 M^-1 s^-1
        # Typical koff: 10^-4 - 10^-1 s^-1
        kon = 10 ** random.uniform(3, 6)
        koff = 10 ** random.uniform(-4, -1)

        records.append(KineticsRecord(
            peptide=random_peptide(pep_len, pep_len),
            mhc_allele=allele,
            kon=kon,
            koff=koff,
            assay_type="surface plasmon resonance",
            mhc_class=mhc_class,
            source="synthetic",
        ))

    return records


def generate_synthetic_stability_data(
    n_samples: int = 50,
    alleles: List[str] = None,
) -> List[StabilityRecord]:
    """Generate realistic synthetic stability data."""
    alleles = alleles or CLASS_I_ALLELES[:10]

    records = []
    for _ in range(n_samples):
        allele = random.choice(alleles)
        mhc_class = "II" if "DR" in allele or "DP" in allele or "DQ" in allele else "I"
        pep_len = random.randint(8, 11) if mhc_class == "I" else random.randint(12, 20)

        # t_half: typically 0.1 - 100 hours
        # Tm: typically 30-70 C
        t_half = 10 ** random.uniform(-1, 2)
        tm = random.gauss(50, 10)

        records.append(StabilityRecord(
            peptide=random_peptide(pep_len, pep_len),
            mhc_allele=allele,
            t_half=t_half,
            tm=max(20, min(80, tm)),
            assay_type="thermal stability",
            mhc_class=mhc_class,
            source="synthetic",
        ))

    return records


def generate_synthetic_processing_data(
    n_samples: int = 50,
) -> List[ProcessingRecord]:
    """Generate realistic synthetic processing data."""
    records = []
    for _ in range(n_samples):
        peptide = random_peptide(9, 11)

        records.append(ProcessingRecord(
            peptide=peptide,
            flank_n=random_flank(10),
            flank_c=random_flank(10),
            label=float(random.random() > 0.3),  # 70% positive
            processing_type=random.choice(["cleavage", "tap", "processing"]),
            source="synthetic",
        ))

    return records


def generate_synthetic_elution_data(
    n_samples: int = 50,
    alleles: List[str] = None,
) -> List[ElutionRecord]:
    """Generate realistic synthetic MS/elution data."""
    alleles = alleles or CLASS_I_ALLELES[:10]
    if not alleles:
        return []

    records = []
    for _ in range(n_samples):
        # Cell lines typically express 3-6 alleles
        max_alleles = min(6, len(alleles))
        min_alleles = min(3, max_alleles)
        n_alleles = random.randint(min_alleles, max_alleles)
        sample_alleles = random.sample(alleles, n_alleles)

        records.append(ElutionRecord(
            peptide=random_peptide(8, 12),
            alleles=sample_alleles,
            detected=True,  # MS data is typically positive-only
            cell_type=random.choice(CELL_TYPES),
            tissue=random.choice(TISSUES),
            mhc_class="I",
            species="human",
            source="synthetic",
        ))

    return records


def generate_synthetic_tcell_data(
    n_samples: int = 50,
    alleles: List[str] = None,
    include_tcr: bool = True,
) -> List[TCellRecord]:
    """Generate realistic synthetic T-cell assay data."""
    alleles = alleles or CLASS_I_ALLELES[:10]

    records = []
    for _ in range(n_samples):
        allele = random.choice(alleles)
        mhc_class = "II" if "DR" in allele or "DP" in allele or "DQ" in allele else "I"
        pep_len = random.randint(8, 11) if mhc_class == "I" else random.randint(12, 20)

        # Include TCR info for some samples
        has_tcr = include_tcr and random.random() > 0.5

        records.append(TCellRecord(
            peptide=random_peptide(pep_len, pep_len),
            mhc_allele=allele,
            response=float(random.random() > 0.4),  # 60% positive
            assay_type=random.choice(TCELL_ASSAYS),
            tcr_a_cdr3=random_tcr_cdr3("alpha") if has_tcr else None,
            tcr_b_cdr3=random_tcr_cdr3("beta") if has_tcr else None,
            v_alpha=random.choice(V_ALPHA_GENES) if has_tcr else None,
            j_alpha=random.choice(J_ALPHA_GENES) if has_tcr else None,
            v_beta=random.choice(V_BETA_GENES) if has_tcr else None,
            j_beta=random.choice(J_BETA_GENES) if has_tcr else None,
            mhc_class=mhc_class,
            species="human",
            source="synthetic",
        ))

    return records


def generate_synthetic_vdjdb_data(
    n_samples: int = 50,
    alleles: List[str] = None,
) -> List[VDJdbRecord]:
    """Generate realistic synthetic VDJdb data."""
    alleles = alleles or CLASS_I_ALLELES[:10]
    pathogens = ["CMV", "EBV", "Influenza", "SARS-CoV-2", "HIV", "HCV", "HPV"]

    records = []
    for _ in range(n_samples):
        allele = random.choice(alleles)
        mhc_class = "II" if "DR" in allele or "DP" in allele or "DQ" in allele else "I"
        pep_len = random.randint(8, 11) if mhc_class == "I" else random.randint(12, 20)

        # VDJdb has separate alpha and beta entries
        gene = random.choice(["TRA", "TRB"])
        is_alpha = gene == "TRA"

        records.append(VDJdbRecord(
            peptide=random_peptide(pep_len, pep_len),
            mhc_a=allele,
            mhc_b="B2M" if mhc_class == "I" else None,
            cdr3_alpha=random_tcr_cdr3("alpha") if is_alpha else None,
            cdr3_beta=random_tcr_cdr3("beta") if not is_alpha else None,
            v_alpha=random.choice(V_ALPHA_GENES) if is_alpha else None,
            j_alpha=random.choice(J_ALPHA_GENES) if is_alpha else None,
            v_beta=random.choice(V_BETA_GENES) if not is_alpha else None,
            j_beta=random.choice(J_BETA_GENES) if not is_alpha else None,
            gene=gene,
            mhc_class=mhc_class,
            species="human",
            antigen_species=random.choice(pathogens),
            source="synthetic",
        ))

    return records


def generate_synthetic_mhc_sequences(
    alleles: List[str] = None,
) -> Dict[str, str]:
    """Generate synthetic MHC sequences for testing."""
    if alleles is None:
        alleles = CLASS_I_ALLELES[:20] + CLASS_II_ALLELES[:10]

    sequences = {}
    for allele in alleles:
        # Class I alpha: ~275 aa, Class II alpha: ~180 aa, Class II beta: ~220 aa
        if "DR" in allele or "DP" in allele or "DQ" in allele:
            length = 200
        else:
            length = 275
        sequences[allele] = random_mhc_sequence(length)

    # Add B2M
    sequences["B2M"] = random_mhc_sequence(99)

    return sequences


# =============================================================================
# CSV Writers (for testing/export)
# =============================================================================

def write_binding_csv(records: List[BindingRecord], path: str) -> None:
    """Write binding records to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'peptide', 'mhc_allele', 'measurement_value', 'measurement_inequality',
            'measurement_type', 'unit', 'assay_type', 'mhc_class', 'species'
        ])
        for rec in records:
            qual = {-1: '<', 0: '=', 1: '>'}.get(rec.qualifier, '=')
            writer.writerow([
                rec.peptide, rec.mhc_allele, rec.value, qual,
                rec.measurement_type, rec.unit, rec.assay_type or '',
                rec.mhc_class, rec.species
            ])


def write_elution_csv(records: List[ElutionRecord], path: str) -> None:
    """Write elution records to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['peptide', 'alleles', 'detected', 'cell_type', 'tissue', 'mhc_class', 'species'])
        for rec in records:
            writer.writerow([
                rec.peptide, ','.join(rec.alleles), 1 if rec.detected else 0,
                rec.cell_type or '', rec.tissue or '', rec.mhc_class, rec.species
            ])


def write_tcr_csv(records: List[TCellRecord], path: str) -> None:
    """Write T-cell records to CSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'peptide', 'mhc_allele', 'response', 'assay_type',
            'tcr_a_cdr3', 'tcr_b_cdr3', 'v_alpha', 'j_alpha', 'v_beta', 'j_beta',
            'mhc_class', 'species'
        ])
        for rec in records:
            writer.writerow([
                rec.peptide, rec.mhc_allele, rec.response, rec.assay_type or '',
                rec.tcr_a_cdr3 or '', rec.tcr_b_cdr3 or '',
                rec.v_alpha or '', rec.j_alpha or '', rec.v_beta or '', rec.j_beta or '',
                rec.mhc_class, rec.species
            ])


def write_vdjdb_tsv(records: List[VDJdbRecord], path: str) -> None:
    """Write VDJdb records to TSV."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'gene', 'cdr3', 'v.segm', 'j.segm', 'antigen.epitope',
            'mhc.a', 'mhc.b', 'mhc.class', 'species', 'antigen.species'
        ])
        for rec in records:
            cdr3 = rec.cdr3_alpha or rec.cdr3_beta or ''
            v = rec.v_alpha or rec.v_beta or ''
            j = rec.j_alpha or rec.j_beta or ''
            writer.writerow([
                rec.gene, cdr3, v, j, rec.peptide,
                rec.mhc_a, rec.mhc_b or '', rec.mhc_class, rec.species,
                rec.antigen_species or ''
            ])


def write_mhc_fasta(sequences: Dict[str, str], path: str) -> None:
    """Write MHC sequences to FASTA."""
    with open(path, 'w') as f:
        for name, seq in sequences.items():
            f.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")


# Backward compatibility alias
generate_synthetic_tcr_data = generate_synthetic_tcell_data
