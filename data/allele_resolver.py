"""Allele resolver for MHC sequences.

Resolves MHC allele names to sequences using IMGT/HLA and IPD-MHC databases.
"""

import csv
import importlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .vocab import (
    CHAIN_SPECIES_CATEGORIES,
    CHAIN_SPECIES_TO_IDX,
    FINE_TO_B2M_KEY,
    FINE_TO_CHAIN_SPECIES,
    MHC_SPECIES_CATEGORIES,
    N_MHC_SPECIES,
    normalize_organism,
    normalize_species,
)

logger = logging.getLogger(__name__)


_MHC_CLASS_I_ALIASES = {
    "I",
    "IA",
    "IB",
    "IC",
    "CLASSI",
    "CLASS-I",
    "MHCI",
    "MHC-I",
}
_MHC_CLASS_II_ALIASES = {
    "II",
    "IIA",
    "IIB",
    "CLASSII",
    "CLASS-II",
    "MHCII",
    "MHC-II",
}

# Coarse network buckets. Keep these distinct from fine-grained species identity
# returned by `infer_species_identity`.
PROCESSING_SPECIES_BUCKETS = tuple(MHC_SPECIES_CATEGORIES)
PROCESSING_SPECIES_TO_IDX = {
    name: idx for idx, name in enumerate(PROCESSING_SPECIES_BUCKETS)
}

# ---------------------------------------------------------------------------
# B2M sequences — loaded from external CSV
# ---------------------------------------------------------------------------
_B2M_CSV = Path(__file__).parent / "b2m_sequences.csv"


def _load_b2m_sequences() -> Dict[str, str]:
    """Load B2M sequences from CSV alongside this module."""
    seqs: Dict[str, str] = {}
    with open(_B2M_CSV, newline="") as f:
        for row in csv.DictReader(f):
            seqs[row["species_key"]] = row["sequence"]
    return seqs


_B2M_SEQUENCES: Dict[str, str] = _load_b2m_sequences()

# Backward-compatible aliases
HUMAN_B2M_SEQUENCE = _B2M_SEQUENCES["human"]
MOUSE_B2M_SEQUENCE = _B2M_SEQUENCES["mouse"]
MACAQUE_B2M_SEQUENCE = _B2M_SEQUENCES["human"]  # chimp/macaque ≈ human

_RAW_DEFAULT_DR_ALPHA_BY_PREFIX: Dict[str, str] = {
    # Use two-field names where they are sequence-unique in the local index.
    # Preserve higher resolution only where two-field collapse is protein-ambiguous.
    "Aona": "Aona-DRA*01:01",
    "Aovo": "Aovo-DRA*01:01",
    "BoLA": "BoLA-DRA*001:01",
    "Chsa": "Chsa-DRA*01:01:01",
    "Eqca": "Eqca-DRA*001:01",
    "Gogo": "Gogo-DRA*01:01",
    "HLA": "HLA-DRA*01:01",
    "Mafa": "Mafa-DRA*01:01:01:01",
    "Malo": "Malo-DRA*01:01",
    "Mamu": "Mamu-DRA*01:01",
    "Mane": "Mane-DRA*01:01",
    "Ovar": "Ovar-DRA*01:01",
    "Ovca": "Ovca-DRA*01:01",
    "Paan": "Paan-DRA*01:01",
    "Patr": "Patr-DRA*01:01",
    "Poab": "Poab-DRA*01:01",
    "Popy": "Popy-DRA*01:01",
    "SLA": "SLA-DRA*01:01",
}

_DEFAULT_DR_ALPHA_PREFIX_BY_SPECIES: Dict[str, str] = {
    "Aotus nancymaae": "Aona",
    "Aotus vociferans": "Aovo",
    "Bos sp.": "BoLA",
    "Chlorocebus sabaeus": "Chsa",
    "Equus caballus": "Eqca",
    "Gorilla gorilla": "Gogo",
    "Homo sapiens": "HLA",
    "Macaca fascicularis": "Mafa",
    "Macaca leonina": "Malo",
    "Macaca mulatta": "Mamu",
    "Macaca nemestrina": "Mane",
    "Ovar aries": "Ovar",
    "Ovis aries": "Ovar",
    "Ovis canadensis": "Ovca",
    "Pan troglodytes": "Patr",
    "Papio anubis": "Paan",
    "Pongo abelii": "Poab",
    "Pongo pygmaeus": "Popy",
    "Sus sp.": "SLA",
}

_DEFAULT_DR_ALPHA_PREFIX_BY_FINE_SPECIES: Dict[str, str] = {
    "cattle": "BoLA",
    "chimpanzee": "Patr",
    "gorilla": "Gogo",
    "horse": "Eqca",
    "human": "HLA",
    "pig": "SLA",
    "sheep": "Ovar",
}


def _require_mhcgnomes() -> Any:
    try:
        mhcgnomes = importlib.import_module("mhcgnomes")
    except ImportError as exc:
        raise RuntimeError(
            "mhcgnomes is a required dependency for allele parsing and class inference."
        ) from exc
    if callable(getattr(mhcgnomes, "parse", None)):
        return mhcgnomes
    try:
        function_api = importlib.import_module("mhcgnomes.function_api")
    except ImportError as exc:
        raise RuntimeError(
            "mhcgnomes is installed but its parse API could not be imported. "
            "Expected `mhcgnomes.parse` or `mhcgnomes.function_api.parse`."
        ) from exc
    parse_fn = getattr(function_api, "parse", None)
    if not callable(parse_fn):
        raise RuntimeError(
            "mhcgnomes is installed but does not expose a callable parse API."
        )
    setattr(mhcgnomes, "parse", parse_fn)
    return mhcgnomes


def require_mhcgnomes() -> Any:
    """Return the canonical mhcgnomes module with a guaranteed parse API."""
    return _require_mhcgnomes()


def _coerce_allele_name_for_parse(allele: Optional[str]) -> str:
    token = str(allele or "").strip().strip(",;")
    if not token:
        return ""
    token = token.replace("_", "-")
    upper = token.upper()
    if upper.startswith("H-2-"):
        token = "H2-" + token[4:]
        upper = token.upper()
    if upper.startswith("H-2"):
        token = "H2-" + token[3:]
        upper = token.upper()
    short_match = re.match(r"^(?:HLA-)?([A-Z]+)(\d)$", upper)
    if short_match:
        gene, field = short_match.groups()
        return f"HLA-{gene}*0{field}"
    return token


def _canonicalize_parsed_allele(parsed: Any, allele_fields: int = 2) -> str:
    if parsed is None:
        raise ValueError("Cannot canonicalize an empty mhcgnomes parse result")
    target_fields = max(1, int(allele_fields))
    restrict_fn = getattr(parsed, "restrict_allele_fields", None)
    if callable(restrict_fn):
        parsed = restrict_fn(target_fields)
    to_string = getattr(parsed, "to_string", None)
    if callable(to_string):
        return str(to_string())
    return str(parsed)


def parse_allele_name(allele: Optional[str]) -> Optional[Any]:
    """Parse an allele string with mhcgnomes.

    Tries the raw token first, then a lightly coerced allele-like form when the
    raw parse returns no object. This keeps mhcgnomes as the source of truth
    while still accepting compact user/data-entry formats and minor separator
    inconsistencies without recursing through `normalize_allele_name()`.
    """
    if not allele:
        return None
    parse_fn = _require_mhcgnomes().parse
    raw = str(allele).strip()
    if not raw:
        return None
    coerced = _coerce_allele_name_for_parse(raw)
    candidates = []
    if coerced:
        candidates.append(coerced)
    if raw not in candidates:
        candidates.append(raw)
    for candidate in candidates:
        parsed = parse_fn(candidate)
        if parsed is not None:
            return parsed
    return None


def _parse_with_mhcgnomes(allele: Optional[str]) -> Optional[Any]:
    return parse_allele_name(allele)


def normalize_mhc_class(value: Optional[str], default: Optional[str] = None) -> Optional[str]:
    """Normalize MHC class labels to canonical "I" / "II".

    Accepts aliases such as Ia, Ib, IIa, Class-I, etc.
    """
    if value is None:
        return default
    normalized = str(value).strip().upper().replace("_", "").replace(" ", "")
    normalized = normalized.replace("/", "").replace("*", "")
    if normalized in _MHC_CLASS_I_ALIASES or (
        normalized.startswith("I") and not normalized.startswith("II")
    ):
        return "I"
    if normalized in _MHC_CLASS_II_ALIASES or normalized.startswith("II"):
        return "II"
    return default


def normalize_species_label(species: Optional[str]) -> Optional[str]:
    """Normalize species labels to 6-class chain species.

    Returns one of the coarse network buckets:
    `human`, `nhp`, `murine`, `other_mammal`, `bird`, `other_vertebrate`.
    Returns None for non-animal or unrecognizable inputs.
    Delegates to the unified fine-grained parser in vocab.py.
    """
    if species is None:
        return None
    fine = normalize_species(species)
    if fine is None:
        return None
    return FINE_TO_CHAIN_SPECIES[fine]


def normalize_processing_species_label(
    species: Optional[str],
    default: Optional[str] = None,
) -> Optional[str]:
    """Normalize species labels to MHC species buckets (6-class).

    Buckets: `human`, `nhp`, `murine`, `other_mammal`, `bird`,
    `other_vertebrate`.
    Non-animal categories (viruses, bacteria, etc.) map to default.
    Delegates to the unified fine-grained parser.
    """
    if species is None:
        return default
    fine = normalize_species(species)
    if fine is None:
        return default
    chain_sp = FINE_TO_CHAIN_SPECIES[fine]
    return chain_sp if chain_sp is not None else default


def infer_processing_species_from_allele(allele: Optional[str]) -> Optional[str]:
    """Infer processing species bucket from allele naming conventions."""
    if not allele:
        return None
    normalized = normalize_processing_species_label(infer_species(str(allele)))
    return normalized or None


def _canonical_mhc_prefix(prefix: Optional[str]) -> Optional[str]:
    token = str(prefix or "").strip()
    if not token:
        return None
    return _DEFAULT_DR_ALPHA_CANONICAL_PREFIXES.get(token.upper())


def _infer_mhc_prefix(allele: Optional[str]) -> Optional[str]:
    if not allele:
        return None
    try:
        parsed = parse_allele_name(allele)
    except Exception:
        return None
    species = getattr(parsed, "species", None)
    prefix = getattr(species, "mhc_prefix", None)
    return _canonical_mhc_prefix(prefix)


def _resolve_native_dr_alpha_prefix(species: Optional[str]) -> Optional[str]:
    raw = str(species or "").strip()
    if not raw:
        return None
    direct_prefix = _canonical_mhc_prefix(raw)
    if direct_prefix is not None:
        return direct_prefix
    if raw in _DEFAULT_DR_ALPHA_PREFIX_BY_SPECIES:
        return _DEFAULT_DR_ALPHA_PREFIX_BY_SPECIES[raw]
    fine = normalize_species(raw)
    if fine is None:
        return None
    return _DEFAULT_DR_ALPHA_PREFIX_BY_FINE_SPECIES.get(fine)


def is_class_ii_dr_beta_allele(allele: Optional[str]) -> bool:
    """Whether an allele names a class-II DR beta chain."""
    if not allele:
        return False
    try:
        parsed = parse_allele_name(allele)
    except Exception:
        parsed = None
    gene_name = getattr(getattr(parsed, "gene", None), "name", None)
    if gene_name:
        return str(gene_name).upper().startswith("DRB")
    try:
        return infer_gene(str(allele)).upper().startswith("DRB")
    except Exception:
        return False


def class_ii_default_dra_allele(
    species: Optional[str] = None,
    beta_allele: Optional[str] = None,
) -> Optional[str]:
    """Return the native default DRA allele for a DR beta chain.

    Resolution order:
    1. Fine-grained species inferred from the beta allele name
    2. Exact species identity string
    3. Stable MHC prefix (`HLA`, `SLA`, `Mamu`, etc.)
    4. Unambiguous fine-species aliases (human, cattle, pig, horse, sheep, chimp, gorilla)
    """
    if beta_allele and not is_class_ii_dr_beta_allele(beta_allele):
        return None
    prefix = _infer_mhc_prefix(beta_allele) if beta_allele else None
    if prefix is None:
        prefix = _resolve_native_dr_alpha_prefix(species)
    if prefix is None:
        return None
    return DEFAULT_DR_ALPHA_BY_PREFIX.get(prefix)


def class_i_beta2m_sequence(species: Optional[str]) -> Optional[str]:
    """Return species-matched beta2m sequence for Class I MHC.

    Delegates to the unified fine-grained parser to determine the B2M key,
    then looks up the sequence from the externalized CSV.
    """
    if species is None:
        return None
    fine = normalize_species(species)
    if fine is None:
        return None
    b2m_key = FINE_TO_B2M_KEY.get(fine)
    if b2m_key is None:
        return None
    return _B2M_SEQUENCES.get(b2m_key)


@dataclass
class AlleleRecord:
    """A resolved MHC allele."""
    name: str
    sequence: str
    gene: str
    mhc_class: str
    species: str = "human"


@dataclass(frozen=True)
class MHCRestrictionExpansion:
    """Canonical parsed representation of a scalar or bagged MHC restriction."""

    raw: str
    normalized_token: str
    exact_alleles: Tuple[str, ...]
    provenance: str
    parsed_type: str = ""
    mhc_class: Optional[str] = None
    species_identity: Optional[str] = None

    @property
    def bag_size(self) -> int:
        return len(self.exact_alleles)

    @property
    def is_exact(self) -> bool:
        return self.provenance == "exact"


def _canonicalize_exact_alleles(alleles: Sequence[Any]) -> Tuple[str, ...]:
    values: List[str] = []
    for allele in alleles:
        try:
            values.append(_canonicalize_parsed_allele(allele, allele_fields=2))
        except Exception:
            continue
    if not values:
        return tuple()
    return tuple(sorted(set(values)))


def expand_mhc_restriction(allele: Optional[str]) -> MHCRestrictionExpansion:
    """Expand an MHC restriction token into exact candidate alleles when possible.

    Exact alleles remain singleton bags. Serotypes, haplotypes, and pair-like
    restrictions retain their normalized coarse token but also expose the exact
    candidate allele set for downstream MIL-capable paths.
    """
    raw = str(allele or "").strip()
    if not raw:
        return MHCRestrictionExpansion(
            raw="",
            normalized_token="",
            exact_alleles=tuple(),
            provenance="unresolved",
        )

    try:
        parsed = _require_mhcgnomes().parse(raw)
    except Exception:
        parsed = None
    if parsed is None:
        try:
            parsed = parse_allele_name(raw)
        except Exception:
            parsed = None
    if parsed is None:
        return MHCRestrictionExpansion(
            raw=raw,
            normalized_token=raw,
            exact_alleles=tuple(),
            provenance="unresolved",
        )

    parsed_type = type(parsed).__name__
    to_string = getattr(parsed, "to_string", None)
    normalized_token = str(to_string()) if callable(to_string) else str(parsed)
    species_name = getattr(getattr(parsed, "species", None), "name", None)
    mhc_class = normalize_mhc_class(getattr(parsed, "mhc_class", None), default=None)

    if parsed_type == "Allele":
        exact_alleles = _canonicalize_exact_alleles((parsed,))
        return MHCRestrictionExpansion(
            raw=raw,
            normalized_token=normalized_token,
            exact_alleles=exact_alleles,
            provenance="exact",
            parsed_type=parsed_type,
            mhc_class=mhc_class or infer_mhc_class_optional(normalized_token),
            species_identity=str(species_name).strip() or None,
        )

    exact_alleles = _canonicalize_exact_alleles(tuple(getattr(parsed, "alleles", ()) or ()))
    if parsed_type == "Serotype":
        provenance = "serotype_expanded"
    elif parsed_type == "Haplotype":
        provenance = "haplotype_expanded"
    elif parsed_type == "Pair":
        provenance = "pair_expanded"
    elif exact_alleles:
        provenance = f"{parsed_type.lower()}_expanded"
    else:
        provenance = "unresolved"

    if mhc_class is None and exact_alleles:
        inferred_classes = {
            infer_mhc_class_optional(token)
            for token in exact_alleles
            if infer_mhc_class_optional(token)
        }
        if len(inferred_classes) == 1:
            mhc_class = next(iter(inferred_classes))

    return MHCRestrictionExpansion(
        raw=raw,
        normalized_token=normalized_token,
        exact_alleles=exact_alleles,
        provenance=provenance,
        parsed_type=parsed_type,
        mhc_class=mhc_class,
        species_identity=str(species_name).strip() or None,
    )


def normalize_allele_name(name: str) -> str:
    """Normalize an allele name to canonical two-field protein resolution.

    Examples:
        "HLA-A*02:01" -> "HLA-A*02:01"
        "A*02:01" -> "HLA-A*02:01"
        "A0201" -> "HLA-A*02:01"
        "HLA-A2" -> "HLA-A*02"
        "HLA-A*02:01:01:02L" -> "HLA-A*02:01L"
    """
    parsed = parse_allele_name(name)
    if parsed is None:
        raise ValueError(f"mhcgnomes failed to parse allele: {name!r}")
    return _canonicalize_parsed_allele(parsed, allele_fields=2)


# Default DR alpha-chain alleles selected from the local MHC index.
# Canonical keys are stable MHC prefixes; species names are convenience aliases.
DEFAULT_DR_ALPHA_BY_PREFIX: Dict[str, str] = dict(_RAW_DEFAULT_DR_ALPHA_BY_PREFIX)
DEFAULT_DR_ALPHA_BY_SPECIES: Dict[str, str] = {
    species: DEFAULT_DR_ALPHA_BY_PREFIX[prefix]
    for species, prefix in _DEFAULT_DR_ALPHA_PREFIX_BY_SPECIES.items()
}
_DEFAULT_DR_ALPHA_CANONICAL_PREFIXES = {
    prefix.upper(): prefix for prefix in DEFAULT_DR_ALPHA_BY_PREFIX
}


def _infer_mhc_class_with_mhcgnomes(allele: Optional[str]) -> Optional[str]:
    """Infer MHC class via mhcgnomes."""
    if not allele:
        return None
    try:
        parsed = _parse_with_mhcgnomes(allele)
    except Exception:
        return None
    if parsed is None:
        return None
    return normalize_mhc_class(getattr(parsed, "mhc_class", None), default=None)


def infer_mhc_class_optional(allele: Optional[str]) -> Optional[str]:
    """Infer MHC class from an allele name via mhcgnomes only."""
    return _infer_mhc_class_with_mhcgnomes(allele)


def infer_mhc_class(allele: Optional[str]) -> str:
    """Infer MHC class from allele name.

    Args:
        allele: Allele name

    Returns:
        "I" or "II"
    """
    inferred = infer_mhc_class_optional(allele)
    if inferred is None:
        raise ValueError(f"mhcgnomes failed to infer MHC class for allele: {allele!r}")
    return inferred


def infer_gene(allele: str) -> str:
    """Extract gene name from allele.

    Args:
        allele: Allele name

    Returns:
        Gene name (e.g., "A", "B", "DRB1")
    """
    try:
        parsed = _parse_with_mhcgnomes(allele)
    except Exception:
        parsed = None
    if parsed is not None and getattr(parsed, "gene", None) is not None:
        gene_name = getattr(parsed.gene, "name", None)
        if gene_name:
            return str(gene_name).upper()

    try:
        allele = normalize_allele_name(allele)
    except Exception:
        allele = str(allele).strip()

    # Remove species prefix (HLA-, H2-, MAMU-, or generic species codes)
    if "-" in allele:
        _, remainder = allele.split("-", 1)
        if remainder:
            allele = remainder

    # Extract gene (everything before *)
    if "*" in allele:
        return allele.split("*")[0]

    return allele


def infer_species(allele: str) -> Optional[str]:
    """Infer species from allele name.

    Returns one of the MHC species categories:
    human, nhp, murine, other_mammal, bird, other_vertebrate.
    Returns None when the species cannot be determined from the allele name.
    """
    species_identity = infer_species_identity(allele)
    if species_identity is None:
        return None
    return normalize_processing_species_label(species_identity)


def infer_species_identity(allele: Optional[str]) -> Optional[str]:
    """Infer fine-grained species identity from an allele via mhcgnomes."""
    if not allele:
        return None
    try:
        parsed = _parse_with_mhcgnomes(allele)
    except Exception:
        return None
    if parsed is None or getattr(parsed, "species", None) is None:
        return None
    species_name = getattr(parsed.species, "name", None)
    return str(species_name).strip() or None


def validate_mhc_species_coverage(mhc_index: Dict[str, str]) -> Dict[str, int]:
    """Check that all alleles in an MHC index map to a known species.

    Args:
        mhc_index: Dict mapping allele name → sequence.

    Returns:
        Dict mapping species category → count of alleles in that category.
    """
    from collections import Counter
    counts: Counter = Counter()
    unmapped: list = []
    for allele_name in mhc_index:
        sp = infer_processing_species_from_allele(allele_name)
        counts[sp] += 1
        # Log if we couldn't infer a specific species (fell through to default)
        if sp == "human" and not allele_name.upper().startswith("HLA-"):
            unmapped.append(allele_name)
    if unmapped:
        logger.warning(
            "MHC species coverage: %d alleles fell through to default 'human' "
            "but don't start with HLA-: %s",
            len(unmapped),
            unmapped[:10],
        )
    return dict(counts)


class AlleleResolver:
    """Resolves MHC allele names to sequences.

    Can load sequences from IMGT/HLA FASTA files or IPD-MHC.
    """

    def __init__(
        self,
        imgt_fasta: Optional[str] = None,
        ipd_mhc_dir: Optional[str] = None,
    ):
        """Initialize resolver.

        Args:
            imgt_fasta: Path to IMGT/HLA FASTA file
            ipd_mhc_dir: Path to IPD-MHC directory
        """
        self.records: Dict[str, AlleleRecord] = {}
        self._aliases: Dict[str, str] = {}

        if imgt_fasta:
            self.load_imgt_fasta(imgt_fasta)

        if ipd_mhc_dir:
            self.load_ipd_mhc(ipd_mhc_dir)

        # Add common beta2m sequence
        self.beta2m = HUMAN_B2M_SEQUENCE

    def load_imgt_fasta(self, path: str) -> int:
        """Load allele sequences from IMGT/HLA FASTA file.

        Args:
            path: Path to FASTA file

        Returns:
            Number of sequences loaded
        """
        count = 0
        current_header = None
        current_seq = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # Save previous record
                    if current_header and current_seq:
                        self._add_from_header(current_header, "".join(current_seq))
                        count += 1
                    current_header = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)

            # Save last record
            if current_header and current_seq:
                self._add_from_header(current_header, "".join(current_seq))
                count += 1

        return count

    def _add_from_header(self, header: str, sequence: str) -> None:
        """Parse IMGT header and add record."""
        # IMGT header format: HLA:HLA00001 A*01:01:01:01 365 bp
        parts = header.split()
        if len(parts) >= 2:
            allele_name = parts[1] if "HLA" not in parts[0] else parts[0]
            # Try to extract from format like "HLA:HLA00001 A*01:01:01:01"
            for part in parts:
                if "*" in part:
                    allele_name = part
                    break
            self._add_record(allele_name, sequence)

    def _add_record(self, allele_name: str, sequence: str) -> None:
        """Add an allele record and its aliases."""
        parsed = parse_allele_name(allele_name)
        if parsed is None:
            raise ValueError(f"mhcgnomes failed to parse allele: {allele_name!r}")
        normalized = _canonicalize_parsed_allele(parsed, allele_fields=99)
        record = AlleleRecord(
            name=normalized,
            sequence=sequence,
            gene=infer_gene(normalized),
            mhc_class=infer_mhc_class(normalized),
            species=infer_species(normalized),
        )

        self.records[normalized] = record

        # Also add truncated versions as aliases under the protein-resolution key.
        parts = normalized.split(":")
        for i in range(1, len(parts)):
            alias = ":".join(parts[:i])
            if alias not in self._aliases:
                self._aliases[alias] = normalized

    def load_ipd_mhc(self, directory: str) -> int:
        """Load sequences from IPD-MHC directory.

        Args:
            directory: Path to IPD-MHC data directory

        Returns:
            Number of sequences loaded
        """
        path = Path(directory)
        if not path.exists():
            return 0

        # Accept a single FASTA file or a directory of FASTA files
        if path.is_file():
            fasta_paths = [path]
        else:
            fasta_paths = list(path.glob("*.fasta")) + list(path.glob("*.fa"))
            # Prefer protein sequences when both are present
            prot_paths = [p for p in fasta_paths if "prot" in p.name.lower()]
            if prot_paths:
                fasta_paths = prot_paths

        count = 0
        current_header = None
        current_seq: List[str] = []

        for fasta_path in fasta_paths:
            with open(fasta_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if current_header and current_seq:
                            self._add_ipd_from_header(current_header, "".join(current_seq))
                            count += 1
                        current_header = line[1:]
                        current_seq = []
                    else:
                        current_seq.append(line)

                if current_header and current_seq:
                    self._add_ipd_from_header(current_header, "".join(current_seq))
                    count += 1

            # Reset for next file
            current_header = None
            current_seq = []

        return count

    def _add_ipd_from_header(self, header: str, sequence: str) -> None:
        """Parse IPD-MHC header and add record."""
        # IPD header format: IPD-MHC:NHP00001 Aona-DQA1*27:01 73 bp
        parts = header.split()
        allele_name = None
        if len(parts) >= 2:
            allele_name = parts[1]
        else:
            for part in parts:
                if "*" in part or "-" in part:
                    allele_name = part
                    break

        if allele_name:
            self._add_record(allele_name, sequence)

    def resolve(self, allele: str) -> Optional[AlleleRecord]:
        """Resolve allele name to record.

        Args:
            allele: Allele name

        Returns:
            AlleleRecord or None if not found
        """
        normalized = normalize_allele_name(allele)

        # Direct lookup
        if normalized in self.records:
            return self.records[normalized]

        # Try alias
        if normalized in self._aliases:
            return self.records[self._aliases[normalized]]

        # Try progressively shorter versions
        parts = normalized.split(":")
        for i in range(len(parts) - 1, 0, -1):
            shorter = ":".join(parts[:i])
            if shorter in self.records:
                return self.records[shorter]
            if shorter in self._aliases:
                return self.records[self._aliases[shorter]]

        return None

    def get_sequence(self, allele: str) -> Optional[str]:
        """Get sequence for allele.

        Args:
            allele: Allele name

        Returns:
            Sequence string or None
        """
        record = self.resolve(allele)
        return record.sequence if record else None

    def get_mhc_class(self, allele: str) -> str:
        """Get MHC class for allele.

        Args:
            allele: Allele name

        Returns:
            "I" or "II"
        """
        record = self.resolve(allele)
        if record:
            return record.mhc_class
        return infer_mhc_class(allele)

    def nearest(self, sequence: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find nearest known alleles by sequence similarity.

        Args:
            sequence: Query sequence
            top_k: Number of results to return

        Returns:
            List of (allele_name, similarity_score) tuples
        """
        # Simple edit distance based similarity
        scores = []
        for name, record in self.records.items():
            sim = self._sequence_similarity(sequence, record.sequence)
            scores.append((name, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _sequence_similarity(self, s1: str, s2: str) -> float:
        """Compute simple sequence similarity."""
        # Use ratio of matching characters at aligned positions
        min_len = min(len(s1), len(s2))
        if min_len == 0:
            return 0.0
        matches = sum(1 for a, b in zip(s1[:min_len], s2[:min_len]) if a == b)
        return matches / max(len(s1), len(s2))

    def list_alleles(self, gene: str = None, mhc_class: str = None) -> List[str]:
        """List available alleles.

        Args:
            gene: Filter by gene (e.g., "A", "DRB1")
            mhc_class: Filter by class ("I" or "II")

        Returns:
            List of allele names
        """
        results = []
        for name, record in self.records.items():
            if gene and record.gene != gene:
                continue
            if mhc_class and record.mhc_class != mhc_class:
                continue
            results.append(name)
        return sorted(results)
