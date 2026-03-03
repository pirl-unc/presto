"""Allele resolver for MHC sequences.

Resolves MHC allele names to sequences using IMGT/HLA and IPD-MHC databases.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .vocab import MHC_SPECIES_CATEGORIES, N_MHC_SPECIES, normalize_organism

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

# Expanded to 6-class (aligned with MHC_SPECIES_CATEGORIES from vocab.py)
PROCESSING_SPECIES_BUCKETS = tuple(MHC_SPECIES_CATEGORIES)  # human, nhp, murine, other_mammal, bird, fish
PROCESSING_SPECIES_TO_IDX = {
    name: idx for idx, name in enumerate(PROCESSING_SPECIES_BUCKETS)
}

HUMAN_B2M_SEQUENCE = (
    "MSRSVALAVLALLSLSGLEAIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"
)
MOUSE_B2M_SEQUENCE = (
    "MSRSVALAVLALLSLSGLEAIQRTPKIQVYSRHPPENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLSHAEFTPQATDEYACRVNHVTLSQPKIVKWDRDM"
)
MACAQUE_B2M_SEQUENCE = HUMAN_B2M_SEQUENCE


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
    """Normalize species labels used by MHC/B2M helpers."""
    if species is None:
        return None
    s = str(species).strip().lower()
    if not s:
        return None
    if "murine" in s:
        return "mouse"
    if "nhp" in s:
        return "macaque"
    if "human" in s or "homo sapiens" in s:
        return "human"
    if "rat" in s or "rattus" in s:
        return "mouse"
    if "mouse" in s or "mus musculus" in s:
        return "mouse"
    if "chimp" in s or "pan troglodytes" in s:
        return "macaque"
    if "macaque" in s or "macaca" in s:
        return "macaque"
    if s in {"human", "mouse", "macaque", "other"}:
        return s
    return "other"


def normalize_processing_species_label(
    species: Optional[str],
    default: Optional[str] = None,
) -> Optional[str]:
    """Normalize species labels to MHC species buckets (6-class).

    Buckets: `human`, `nhp`, `murine`, `other_mammal`, `bird`, `fish`.
    Non-animal categories (viruses, bacteria, etc.) map to default.
    """
    if species is None:
        return default
    s = str(species).strip()
    if not s:
        return default

    # Delegate to unified normalizer
    category = normalize_organism(s)
    if category is None:
        return default

    # Only animal categories are valid MHC species
    if category in PROCESSING_SPECIES_TO_IDX:
        return category

    # Non-animal organism categories → default
    return default


def infer_processing_species_from_allele(allele: Optional[str]) -> Optional[str]:
    """Infer processing species bucket from allele naming conventions."""
    if not allele:
        return None
    normalized = normalize_processing_species_label(infer_species(str(allele)))
    return normalized or None


def class_i_beta2m_sequence(species: Optional[str]) -> Optional[str]:
    """Return species-matched beta2m sequence for Class I MHC."""
    norm = normalize_species_label(species)
    if norm == "human":
        return HUMAN_B2M_SEQUENCE
    if norm == "mouse":
        return MOUSE_B2M_SEQUENCE
    if norm == "macaque":
        return MACAQUE_B2M_SEQUENCE
    return None


@dataclass
class AlleleRecord:
    """A resolved MHC allele."""
    name: str
    sequence: str
    gene: str
    mhc_class: str
    species: str = "human"


def normalize_allele_name(name: str) -> str:
    """Normalize allele name to standard format.

    Examples:
        "HLA-A*02:01" -> "HLA-A*02:01"
        "A*02:01" -> "HLA-A*02:01"
        "A0201" -> "HLA-A*02:01"
        "HLA-A2" -> "HLA-A*02"
    """
    name = name.strip().upper()

    # Preserve non-human prefixes (e.g., H2-, MAMU-, AONA-)
    if name.startswith("H2-") or name.startswith("MAMU-"):
        return name
    if re.match(r"^[A-Z0-9]{2,}-", name) and not name.startswith("HLA-"):
        return name

    # Remove common HLA prefixes
    name = name.replace("HLA-", "").replace("HLA_", "")

    # Handle compact format (A0201 -> A*02:01)
    compact_match = re.match(r"^([A-Z]+)(\d{2})(\d{2})$", name)
    if compact_match:
        gene, group, protein = compact_match.groups()
        name = f"{gene}*{group}:{protein}"

    # Handle short format (A2 -> A*02)
    short_match = re.match(r"^([A-Z]+)(\d)$", name)
    if short_match:
        gene, group = short_match.groups()
        name = f"{gene}*0{group}"

    # Add HLA- prefix for human alleles
    if not name.startswith("H2-") and not name.startswith("MAMU-"):
        if not name.startswith("HLA-"):
            name = "HLA-" + name

    return name


def infer_mhc_class(allele: str) -> str:
    """Infer MHC class from allele name.

    Args:
        allele: Allele name

    Returns:
        "I" or "II"
    """
    allele = allele.upper()

    # Class II genes
    class_ii_genes = ["DRA", "DRB", "DQA", "DQB", "DPA", "DPB", "DM", "DO"]
    for gene in class_ii_genes:
        if gene in allele:
            return "II"

    # Mouse Class II
    if "H2-A" in allele or "H2-E" in allele:
        # H2-Ab, H2-Eb are Class II
        if re.search(r"H2-[AE][AB]", allele):
            return "II"

    return "I"


def infer_gene(allele: str) -> str:
    """Extract gene name from allele.

    Args:
        allele: Allele name

    Returns:
        Gene name (e.g., "A", "B", "DRB1")
    """
    allele = normalize_allele_name(allele)

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
    human, nhp, murine, other_mammal, bird, fish.
    Returns None when the species cannot be determined from the allele name.
    """
    allele_upper = allele.upper()
    if allele_upper.startswith("HLA-"):
        return "human"
    if allele_upper.startswith("H2-") or allele_upper.startswith("H-2"):
        return "murine"
    # NHP allele prefixes
    _nhp_prefixes = ("MAMU-", "PAAN-", "AONA-", "PATR-", "GOGO-", "PAPA-")
    if any(allele_upper.startswith(p) for p in _nhp_prefixes):
        return "nhp"
    # Other mammal allele prefixes
    _mammal_prefixes = ("BOLA-", "SLA-", "DLA-", "ELA-", "OLA-")
    if any(allele_upper.startswith(p) for p in _mammal_prefixes):
        return "other_mammal"
    # Bird allele prefixes
    if allele_upper.startswith("GAGA-") or allele_upper.startswith("BF-"):
        return "bird"
    # Fish — no standard prefix convention, but check common patterns
    _fish_prefixes = ("ONMY-", "SASA-")
    if any(allele_upper.startswith(p) for p in _fish_prefixes):
        return "fish"
    return None


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
        normalized = normalize_allele_name(allele_name)
        record = AlleleRecord(
            name=normalized,
            sequence=sequence,
            gene=infer_gene(normalized),
            mhc_class=infer_mhc_class(normalized),
            species=infer_species(normalized),
        )

        self.records[normalized] = record

        # Also add truncated versions as aliases
        # HLA-A*02:01:01:01 -> HLA-A*02:01:01 -> HLA-A*02:01 -> HLA-A*02
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
