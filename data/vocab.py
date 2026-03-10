"""Vocabulary constants for Presto.

Defines amino acid vocabulary, chain types, cell types, MHC types, species,
organism categories, and biological compatibility matrices.
"""

import re
from typing import Dict, Optional, Set, Tuple

# Amino acid vocabulary with special tokens
AA_VOCAB = [
    "<PAD>",  # 0 - padding
    "<UNK>",  # 1 - unknown
    "<BOS>",  # 2 - beginning of sequence
    "<EOS>",  # 3 - end of sequence
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    "X",  # unknown/any amino acid
    "<MISSING>",  # dedicated missing-value token
]

AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AA_VOCAB)}

# Chain types for TCR and BCR
# Full-length chains
CHAIN_TYPES_FULL = ["TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL"]
# CDR3-only variants (common in sequencing data)
CHAIN_TYPES_CDR3 = ["TRA_CDR3", "TRB_CDR3", "TRG_CDR3", "TRD_CDR3", "IGH_CDR3", "IGK_CDR3", "IGL_CDR3"]
# Combined
CHAIN_TYPES = CHAIN_TYPES_FULL + CHAIN_TYPES_CDR3
CHAIN_TO_IDX = {ct: i for i, ct in enumerate(CHAIN_TYPES)}
IDX_TO_CHAIN = {i: ct for i, ct in enumerate(CHAIN_TYPES)}

# Map CDR3 types to their full-length parent
CDR3_TO_FULL = {
    "TRA_CDR3": "TRA", "TRB_CDR3": "TRB", "TRG_CDR3": "TRG", "TRD_CDR3": "TRD",
    "IGH_CDR3": "IGH", "IGK_CDR3": "IGK", "IGL_CDR3": "IGL",
}
FULL_TO_CDR3 = {v: k for k, v in CDR3_TO_FULL.items()}

# Cell types
CELL_TYPES = ["CD4_T", "CD8_T", "ab_T", "gd_T", "B_cell"]
CELL_TO_IDX = {ct: i for i, ct in enumerate(CELL_TYPES)}
IDX_TO_CELL = {i: ct for i, ct in enumerate(CELL_TYPES)}

# MHC types (for cell-MHC compatibility)
MHC_TYPES = ["MHC_I", "MHC_II", "HLA_E", "HLA_F", "HLA_G"]
MHC_TO_IDX = {mt: i for i, mt in enumerate(MHC_TYPES)}
IDX_TO_MHC = {i: mt for i, mt in enumerate(MHC_TYPES)}

# =============================================================================
# MHC CHAIN TYPES (for chain-level classification)
# =============================================================================

# Full-length MHC chain types
MHC_CHAIN_TYPES_FULL = [
    # Class I heavy chains (alpha)
    "MHC_I_ALPHA",      # HLA-A, HLA-B, HLA-C, H2-K, H2-D, etc.
    # Class I light chain
    "B2M",              # Beta-2-microglobulin (invariant)
    # Class II chains
    "MHC_II_ALPHA",     # HLA-DRA, HLA-DQA, HLA-DPA, H2-Aa, etc.
    "MHC_II_BETA",      # HLA-DRB, HLA-DQB, HLA-DPB, H2-Ab, etc.
    # Non-classical Class I
    "HLA_E_ALPHA",
    "HLA_F_ALPHA",
    "HLA_G_ALPHA",
    # Class I-like (for future expansion)
    "MR1_ALPHA",        # MR1 (presents microbial metabolites to MAIT cells)
    "CD1_ALPHA",        # CD1 family (presents lipids)
]

# Pseudosequence variants (e.g., from NetMHCpan, MHCflurry)
MHC_CHAIN_TYPES_PSEUDO = [
    "MHC_I_ALPHA_PSEUDO",     # 34-residue pseudosequence
    "MHC_II_ALPHA_PSEUDO",
    "MHC_II_BETA_PSEUDO",
]

# Combined MHC chain types
MHC_CHAIN_TYPES = MHC_CHAIN_TYPES_FULL + MHC_CHAIN_TYPES_PSEUDO
MHC_CHAIN_TO_IDX = {ct: i for i, ct in enumerate(MHC_CHAIN_TYPES)}
IDX_TO_MHC_CHAIN = {i: ct for i, ct in enumerate(MHC_CHAIN_TYPES)}

# Map pseudosequence types to their full-length parent
MHC_PSEUDO_TO_FULL = {
    "MHC_I_ALPHA_PSEUDO": "MHC_I_ALPHA",
    "MHC_II_ALPHA_PSEUDO": "MHC_II_ALPHA",
    "MHC_II_BETA_PSEUDO": "MHC_II_BETA",
}

# Which MHC chain types can pair together
# Key: chain type, Value: set of compatible partner chain types
MHC_CHAIN_PAIRING: dict[str, Set[str]] = {
    # Class I: alpha pairs with B2M
    "MHC_I_ALPHA": {"B2M"},
    "MHC_I_ALPHA_PSEUDO": {"B2M"},
    "B2M": {"MHC_I_ALPHA", "MHC_I_ALPHA_PSEUDO", "HLA_E_ALPHA", "HLA_F_ALPHA", "HLA_G_ALPHA"},
    # Non-classical also pair with B2M
    "HLA_E_ALPHA": {"B2M"},
    "HLA_F_ALPHA": {"B2M"},
    "HLA_G_ALPHA": {"B2M"},
    # Class II: alpha pairs with beta of SAME locus
    "MHC_II_ALPHA": {"MHC_II_BETA", "MHC_II_BETA_PSEUDO"},
    "MHC_II_ALPHA_PSEUDO": {"MHC_II_BETA", "MHC_II_BETA_PSEUDO"},
    "MHC_II_BETA": {"MHC_II_ALPHA", "MHC_II_ALPHA_PSEUDO"},
    "MHC_II_BETA_PSEUDO": {"MHC_II_ALPHA", "MHC_II_ALPHA_PSEUDO"},
    # Special
    "MR1_ALPHA": {"B2M"},
    "CD1_ALPHA": {"B2M"},
}

# Locus-specific pairing rules for Class II (must match locus)
# e.g., DRA pairs with DRB, DQA pairs with DQB, DPA pairs with DPB
MHC_II_LOCI = {
    "human": ["DR", "DQ", "DP"],
    "mouse": ["A", "E"],  # H2-Aa/H2-Ab, H2-Ea/H2-Eb
}

# Fine-grained MHC chain types for per-chain auxiliary heads (5 classes).
# These map the detailed chain types above to a compact label set that
# the model predicts to learn MHC structural distinctions.
MHC_CHAIN_FINE_TYPES = [
    "MHC_I",        # Class I alpha (classical + non-classical: HLA-A,-B,-C,-E,-F,-G, H2-K,-D,-L, Qa, Tla)
    "MHC_IIa",      # Class II alpha (DRA, DQA, DPA, H2-Aa, H2-Ea)
    "MHC_IIb",      # Class II beta (DRB, DQB, DPB, H2-Ab, H2-Eb)
    "B2M",          # Beta-2-microglobulin
    "unknown",      # Unresolvable
]
MHC_CHAIN_FINE_TO_IDX = {ct: i for i, ct in enumerate(MHC_CHAIN_FINE_TYPES)}
IDX_TO_MHC_CHAIN_FINE = {i: ct for i, ct in enumerate(MHC_CHAIN_FINE_TYPES)}
N_MHC_CHAIN_FINE_TYPES = len(MHC_CHAIN_FINE_TYPES)

# =============================================================================
# UNIFIED ORGANISM TAXONOMY (12-class)
# =============================================================================
# Single vocabulary for ALL organism classification: peptide source, MHC source,
# UniProt taxonomy, IEDB organism names, VDJdb antigen_species.
ORGANISM_CATEGORIES = [
    # Animal (MHC sources + host organisms) — first 6
    "human", "nhp", "murine", "other_mammal", "bird", "other_vertebrate",
    # Other animals
    "invertebrate",
    # Foreign (pathogens)
    "fungi", "bacteria", "viruses", "archaea",
]
ORGANISM_TO_IDX = {cat: i for i, cat in enumerate(ORGANISM_CATEGORIES)}
IDX_TO_ORGANISM = {i: cat for i, cat in enumerate(ORGANISM_CATEGORIES)}
N_ORGANISM_CATEGORIES = len(ORGANISM_CATEGORIES)

# =============================================================================
# CHAIN SPECIES (6-class vertebrate subset)
# =============================================================================
# Used for BOTH MHC chain species heads AND TCR/BCR chain attribute classifier.
CHAIN_SPECIES_CATEGORIES = ORGANISM_CATEGORIES[:6]  # human, nhp, murine, other_mammal, bird, other_vertebrate
CHAIN_SPECIES_TO_IDX = {sp: i for i, sp in enumerate(CHAIN_SPECIES_CATEGORIES)}
IDX_TO_CHAIN_SPECIES = {i: sp for i, sp in enumerate(CHAIN_SPECIES_CATEGORIES)}
N_CHAIN_SPECIES = len(CHAIN_SPECIES_CATEGORIES)

# Backward-compat aliases (used by models, training scripts)
MHC_SPECIES_CATEGORIES = CHAIN_SPECIES_CATEGORIES
N_MHC_SPECIES = N_CHAIN_SPECIES

# Foreignness: pathogens vs animals
FOREIGN_CATEGORIES = frozenset({"bacteria", "viruses", "fungi", "archaea"})


# =============================================================================
# UNIFIED FINE-GRAINED SPECIES TAXONOMY (29-class)
# =============================================================================
# Single canonical parser; all coarser views are roll-ups.

FINE_SPECIES = [
    # Primates
    "human", "macaque", "chimpanzee", "gorilla", "orangutan", "baboon", "other_nhp",
    # Rodents
    "mouse", "rat",
    # Mammals
    "cattle", "pig", "horse", "sheep", "goat", "dog", "cat", "rabbit", "other_mammal",
    # Birds
    "chicken", "other_bird",
    # Fish
    "salmon", "zebrafish", "other_fish",
    # Non-animal
    "other_vertebrate", "invertebrate", "viruses", "bacteria", "fungi", "archaea",
]
FINE_SPECIES_TO_IDX = {sp: i for i, sp in enumerate(FINE_SPECIES)}
N_FINE_SPECIES = len(FINE_SPECIES)

# Keyword patterns checked in order; first match wins.
# Each entry: (keywords_tuple, fine_species_label)
_SPECIES_PATTERNS: list[Tuple[Tuple[str, ...], str]] = [
    # --- Human ---
    (("homo sapiens", "human"), "human"),

    # --- NHP: specific species first ---
    (("chimpanzee", "pan troglodytes", "pan paniscus", "patr-"), "chimpanzee"),
    (("gorilla", "gogo-"), "gorilla"),
    (("orangutan", "pongo", "popy-"), "orangutan"),
    (("baboon", "papio", "paan-"), "baboon"),
    (("macaque", "macaca", "rhesus", "mamu-"), "macaque"),
    # Catch-all NHP
    (("nhp", "aotus", "night monkey", "aona-", "cercopithecus",
      "saguinus", "callithrix", "saimiri", "ateles", "pithecia",
      "leontopithecus", "hylobates", "chlorocebus", "cercocebus",
      "primate"), "other_nhp"),

    # --- Rodents ---
    (("mus musculus", "mouse", "c57bl", "balb/c"), "mouse"),
    (("rattus", "rat "), "rat"),
    # Catch-all murine token (maps to mouse if nothing more specific)
    (("murine", "h2-", "h-2"), "mouse"),

    # --- Other mammals (specific first) ---
    (("bos taurus", "bos ", "bovine", "cow", "cattle", "bola-", "bos grunniens"), "cattle"),
    (("sus scrofa", "sus ", "porcine", "pig", "swine", "sla-"), "pig"),
    (("equus", "equine", "horse", "ela-"), "horse"),
    (("ovis aries", "ovine", "sheep", "ola-"), "sheep"),
    (("capra", "caprine", "goat"), "goat"),
    (("canis", "canine", "dog", "dla-"), "dog"),
    (("felis", "feline", "cat "), "cat"),
    (("rabbit", "oryctolagus"), "rabbit"),
    (("mammal",), "other_mammal"),

    # --- Birds ---
    (("gallus", "chicken", "gaga-"), "chicken"),
    (("duck", "turkey", "quail", "bird", "avian", "aves"), "other_bird"),

    # --- Pathogens (BEFORE fish, so "salmonella" → bacteria, not fish) ---

    # --- Viruses ---
    (("virus", "viral", "influenza", "sars", "cov",
      "hiv", "hcv", "hbv", "ebv", "cmv", "hsv", "vzv",
      "htlv", "dengue", "zika", "ebola", "measles",
      "hepatitis", "retrovirus", "coronavirus",
      "adenovirus", "papillomavirus", "herpes",
      "vaccinia", "poxvirus", "flavivirus",
      "paramyxovirus", "orthomyxovirus",
      "phage", "bacteriophage"), "viruses"),

    # --- Bacteria ---
    (("mycobacterium", "tuberculosis", "escherichia", "e. coli",
      "staphylococcus", "streptococcus", "salmonella",
      "clostridium", "listeria", "helicobacter",
      "chlamydia", "borrelia", "treponema",
      "pseudomonas", "bacillus", "legionella",
      "neisseria", "rickettsia", "bartonella",
      "bacterium", "bacteria", "bacterial"), "bacteria"),

    # --- Fungi ---
    (("candida", "aspergillus", "cryptococcus",
      "coccidioides", "histoplasma", "blastomyces",
      "saccharomyces", "yeast", "fungus", "fungi", "fungal",
      "pneumocystis", "trichophyton"), "fungi"),

    # --- Archaea ---
    (("archaea", "archaeal", "methanobacterium",
      "halobacterium", "sulfolobus", "thermococcus"), "archaea"),

    # --- Fish (AFTER bacteria, so "salmonella" doesn't match "salmon") ---
    (("salmo salar", "salmo ", "salmon", "trout", "oncorhynchus"), "salmon"),
    (("danio", "zebrafish"), "zebrafish"),
    (("fish", "pisces"), "other_fish"),

    # --- Other vertebrate ---
    (("reptile", "reptilia", "amphibian", "amphibia",
      "frog", "xenopus", "turtle", "lizard", "snake",
      "alligator", "crocodile", "salamander"), "other_vertebrate"),

    # --- Invertebrate ---
    (("drosophila", "insect", "arthropod", "arachnid",
      "mosquito", "tick", "worm", "nematode", "mollusk",
      "caenorhabditis", "c. elegans", "invertebrate",
      "schistosoma", "plasmodium", "toxoplasma",
      "leishmania", "trypanosoma", "parasite"), "invertebrate"),
]


def normalize_species(raw: Optional[str]) -> Optional[str]:
    """Unified fine-grained species normalizer (29 categories).

    This is the single canonical parser; all coarser views
    (organism, MHC species, legacy species, B2M key) are derived
    via roll-up dicts.

    Returns None if unrecognizable.
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None

    # Direct hit on fine labels
    if s in FINE_SPECIES_TO_IDX:
        return s

    # Pattern scan — first match wins
    for keywords, label in _SPECIES_PATTERNS:
        if any(kw in s for kw in keywords):
            return label

    return None


# ---- Roll-up mappings (fine → coarser views) ----

FINE_TO_ORGANISM: Dict[str, str] = {
    "human": "human",
    "macaque": "nhp", "chimpanzee": "nhp", "gorilla": "nhp",
    "orangutan": "nhp", "baboon": "nhp", "other_nhp": "nhp",
    "mouse": "murine", "rat": "murine",
    "cattle": "other_mammal", "pig": "other_mammal", "horse": "other_mammal",
    "sheep": "other_mammal", "goat": "other_mammal", "dog": "other_mammal",
    "cat": "other_mammal", "rabbit": "other_mammal", "other_mammal": "other_mammal",
    "chicken": "bird", "other_bird": "bird",
    "salmon": "other_vertebrate", "zebrafish": "other_vertebrate", "other_fish": "other_vertebrate",
    "other_vertebrate": "other_vertebrate",
    "invertebrate": "invertebrate",
    "viruses": "viruses", "bacteria": "bacteria",
    "fungi": "fungi", "archaea": "archaea",
}

FINE_TO_CHAIN_SPECIES: Dict[str, Optional[str]] = {
    "human": "human",
    "macaque": "nhp", "chimpanzee": "nhp", "gorilla": "nhp",
    "orangutan": "nhp", "baboon": "nhp", "other_nhp": "nhp",
    "mouse": "murine", "rat": "murine",
    "cattle": "other_mammal", "pig": "other_mammal", "horse": "other_mammal",
    "sheep": "other_mammal", "goat": "other_mammal", "dog": "other_mammal",
    "cat": "other_mammal", "rabbit": "other_mammal", "other_mammal": "other_mammal",
    "chicken": "bird", "other_bird": "bird",
    "salmon": "other_vertebrate", "zebrafish": "other_vertebrate", "other_fish": "other_vertebrate",
    "other_vertebrate": "other_vertebrate",
}
# Non-animal categories → None (not valid chain species)
for _fs in FINE_SPECIES:
    if _fs not in FINE_TO_CHAIN_SPECIES:
        FINE_TO_CHAIN_SPECIES[_fs] = None

# Backward-compat alias
FINE_TO_MHC_SPECIES = FINE_TO_CHAIN_SPECIES

FINE_TO_B2M_KEY: Dict[str, Optional[str]] = {
    "human": "human",
    # NHP → human B2M (highly conserved across primates)
    "macaque": "human", "chimpanzee": "human", "gorilla": "human",
    "orangutan": "human", "baboon": "human", "other_nhp": "human",
    "mouse": "mouse", "rat": "rat",
    "cattle": "cattle", "pig": "pig", "horse": "horse",
    "sheep": "sheep", "goat": "cattle",  # closest available
    "dog": "dog", "cat": "cat",
    "rabbit": "cattle",  # closest available
    "other_mammal": "cattle",
    "chicken": "chicken", "other_bird": "chicken",
    "salmon": "salmon", "zebrafish": "salmon", "other_fish": "salmon",
    "other_vertebrate": "salmon",  # salmon B2M as closest available non-mammal/non-bird
}
# Non-animal categories → None
for _fs in FINE_SPECIES:
    if _fs not in FINE_TO_B2M_KEY:
        FINE_TO_B2M_KEY[_fs] = None

FINE_TO_IS_FOREIGN: Dict[str, bool] = {
    fs: (fs in {"viruses", "bacteria", "fungi", "archaea"})
    for fs in FINE_SPECIES
}


def normalize_organism(raw: Optional[str]) -> Optional[str]:
    """Unified normalizer: map any organism name to one of 12 categories.

    Delegates to `normalize_species()` (29-class fine-grained) and rolls up
    via `FINE_TO_ORGANISM`.

    Returns None if unrecognizable (will result in mask=0, no supervision).
    """
    if raw is None:
        return None
    # Fast path: direct match on 12-class labels
    s = str(raw).strip().lower()
    if s in ORGANISM_TO_IDX:
        return s
    fine = normalize_species(raw)
    if fine is None:
        return None
    return FINE_TO_ORGANISM[fine]

# T-cell assay context vocabularies (IEDB/CEDAR assay metadata).
TCELL_ASSAY_METHODS = [
    "unknown",
    "ELISPOT",
    "ICS",
    "MULTIMER",
    "ELISA",
    "CYTOTOXICITY_ASSAY",
    "PROLIFERATION_ASSAY",
    "IN_VITRO_ASSAY",
    "IN_VIVO_ASSAY",
    "BIOASSAY",
    "OTHER",
]
TCELL_ASSAY_METHOD_TO_IDX = {
    name: i for i, name in enumerate(TCELL_ASSAY_METHODS)
}
IDX_TO_TCELL_ASSAY_METHOD = {
    i: name for i, name in enumerate(TCELL_ASSAY_METHODS)
}

TCELL_ASSAY_READOUTS = [
    "unknown",
    "IFNG",
    "TNFA",
    "IL2",
    "IL4",
    "IL5",
    "IL10",
    "GMCSF",
    "CYTOTOXICITY",
    "PROLIFERATION",
    "ACTIVATION",
    "QUAL_BINDING",
    "KD",
    "MULTIMER_BINDING",
    "OTHER",
]
TCELL_ASSAY_READOUT_TO_IDX = {
    name: i for i, name in enumerate(TCELL_ASSAY_READOUTS)
}
IDX_TO_TCELL_ASSAY_READOUT = {
    i: name for i, name in enumerate(TCELL_ASSAY_READOUTS)
}

TCELL_APC_TYPES = [
    "unknown",
    "DENDRITIC",
    "B_CELL",
    "PBMC",
    "SPLENOCYTE",
    "T2_B_CELL",
    "B_LCL",
    "T_CELL",
    "OTHER",
]
TCELL_APC_TYPE_TO_IDX = {name: i for i, name in enumerate(TCELL_APC_TYPES)}
IDX_TO_TCELL_APC_TYPE = {i: name for i, name in enumerate(TCELL_APC_TYPES)}

TCELL_CULTURE_CONTEXTS = [
    "unknown",
    "DIRECT_EX_VIVO",
    "SHORT_RESTIM",
    "IN_VITRO",
    "IN_VIVO",
    "ENGINEERED",
    "CELL_LINE_CLONE",
    "NON_SPECIFIC_ACTIVATION",
    "OTHER",
]
TCELL_CULTURE_CONTEXT_TO_IDX = {
    name: i for i, name in enumerate(TCELL_CULTURE_CONTEXTS)
}
IDX_TO_TCELL_CULTURE_CONTEXT = {
    i: name for i, name in enumerate(TCELL_CULTURE_CONTEXTS)
}

TCELL_STIM_CONTEXTS = [
    "unknown",
    "EX_VIVO",
    "IN_VITRO_STIM",
    "IN_VIVO",
    "ENGINEERED",
    "OTHER",
]
TCELL_STIM_CONTEXT_TO_IDX = {name: i for i, name in enumerate(TCELL_STIM_CONTEXTS)}
IDX_TO_TCELL_STIM_CONTEXT = {
    i: name for i, name in enumerate(TCELL_STIM_CONTEXTS)
}

TCELL_PEPTIDE_FORMATS = [
    "unknown",
    "MINIMAL_EPITOPE",
    "LONG_PEPTIDE",
    "PEPTIDE_POOL",
    "WHOLE_PROTEIN",
    "OTHER",
]
TCELL_PEPTIDE_FORMAT_TO_IDX = {
    name: i for i, name in enumerate(TCELL_PEPTIDE_FORMATS)
}
IDX_TO_TCELL_PEPTIDE_FORMAT = {
    i: name for i, name in enumerate(TCELL_PEPTIDE_FORMATS)
}

# Binding assay context vocabularies (quantitative affinity metadata).
BINDING_ASSAY_TYPES = [
    "unknown",
    "KD",
    "KD_PROXY_IC50",
    "KD_PROXY_EC50",
    "IC50",
    "EC50",
    "OTHER",
]
BINDING_ASSAY_TYPE_TO_IDX = {
    name: i for i, name in enumerate(BINDING_ASSAY_TYPES)
}
IDX_TO_BINDING_ASSAY_TYPE = {
    i: name for i, name in enumerate(BINDING_ASSAY_TYPES)
}

BINDING_ASSAY_METHODS = [
    "unknown",
    "PURIFIED_COMPETITIVE_RADIOACTIVITY",
    "PURIFIED_DIRECT_FLUORESCENCE",
    "PURIFIED_COMPETITIVE_FLUORESCENCE",
    "CELLULAR_COMPETITIVE_FLUORESCENCE",
    "CELLULAR_DIRECT_FLUORESCENCE",
    "CELLULAR_COMPETITIVE_RADIOACTIVITY",
    "CELLULAR_TCELL_INHIBITION",
    "LYSATE_DIRECT_RADIOACTIVITY",
    "PURIFIED_DIRECT_RADIOACTIVITY",
    "OTHER",
]
BINDING_ASSAY_METHOD_TO_IDX = {
    name: i for i, name in enumerate(BINDING_ASSAY_METHODS)
}
IDX_TO_BINDING_ASSAY_METHOD = {
    i: name for i, name in enumerate(BINDING_ASSAY_METHODS)
}

# Biological validity: which chain types can appear in which cell types
VALID_CHAIN_CELL: dict[str, Set[str]] = {
    # Full-length chains
    "TRA": {"CD4_T", "CD8_T", "ab_T"},
    "TRB": {"CD4_T", "CD8_T", "ab_T"},
    "TRG": {"gd_T"},
    "TRD": {"gd_T"},
    "IGH": {"B_cell"},
    "IGK": {"B_cell"},
    "IGL": {"B_cell"},
    # CDR3-only chains (same cell type mappings)
    "TRA_CDR3": {"CD4_T", "CD8_T", "ab_T"},
    "TRB_CDR3": {"CD4_T", "CD8_T", "ab_T"},
    "TRG_CDR3": {"gd_T"},
    "TRD_CDR3": {"gd_T"},
    "IGH_CDR3": {"B_cell"},
    "IGK_CDR3": {"B_cell"},
    "IGL_CDR3": {"B_cell"},
}

# Biological compatibility: which cell types can recognize which MHC types
# Empty set means the cell type does NOT bind classical pMHC
CELL_MHC_COMPATIBILITY: dict[str, Set[str]] = {
    "CD4_T": {"MHC_II"},
    "CD8_T": {"MHC_I", "HLA_E"},
    "ab_T": {"MHC_I", "MHC_II", "HLA_E", "HLA_F", "HLA_G"},  # unknown restriction
    "gd_T": set(),  # does not bind classical pMHC
    "B_cell": set(),  # does not bind pMHC
}


def is_valid_chain_cell(chain_type: str, cell_type: str) -> bool:
    """Check if a chain type can appear in a cell type."""
    if chain_type not in VALID_CHAIN_CELL:
        return False
    return cell_type in VALID_CHAIN_CELL[chain_type]


def is_compatible_cell_mhc(cell_type: str, mhc_type: str) -> bool:
    """Check if a cell type can recognize an MHC type."""
    if cell_type not in CELL_MHC_COMPATIBILITY:
        return False
    return mhc_type in CELL_MHC_COMPATIBILITY[cell_type]


def is_cdr3_only(chain_type: str) -> bool:
    """Check if a chain type is CDR3-only."""
    return chain_type in CHAIN_TYPES_CDR3


def get_base_chain_type(chain_type: str) -> str:
    """Get the base chain type (strips _CDR3 suffix if present)."""
    return CDR3_TO_FULL.get(chain_type, chain_type)
