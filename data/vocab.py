"""Vocabulary constants for Presto.

Defines amino acid vocabulary, chain types, cell types, MHC types, species,
organism categories, and biological compatibility matrices.
"""

import re
from typing import Optional, Set

# Amino acid vocabulary with special tokens
AA_VOCAB = [
    "<PAD>",  # 0 - padding
    "<UNK>",  # 1 - unknown
    "<BOS>",  # 2 - beginning of sequence
    "<EOS>",  # 3 - end of sequence
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    "X",  # unknown/any amino acid
    "B",  # Asn or Asp
    "Z",  # Gln or Glu
    "U",  # Selenocysteine
    "O",  # Pyrrolysine
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

# Fine-grained MHC chain types for per-chain auxiliary heads (6 classes).
# These map the detailed chain types above to a compact label set that
# the model predicts to learn MHC structural distinctions.
MHC_CHAIN_FINE_TYPES = [
    "MHC_Ia",       # Classical Class I alpha (HLA-A, -B, -C, H2-K, H2-D, H2-L)
    "MHC_Ib",       # Non-classical Class I alpha (HLA-E, -F, -G, Qa, Tla)
    "MHC_IIa",      # Class II alpha (DRA, DQA, DPA, H2-Aa, H2-Ea)
    "MHC_IIb",      # Class II beta (DRB, DQB, DPB, H2-Ab, H2-Eb)
    "B2M",          # Beta-2-microglobulin
    "unknown",      # Unresolvable
]
MHC_CHAIN_FINE_TO_IDX = {ct: i for i, ct in enumerate(MHC_CHAIN_FINE_TYPES)}
IDX_TO_MHC_CHAIN_FINE = {i: ct for i, ct in enumerate(MHC_CHAIN_FINE_TYPES)}
N_MHC_CHAIN_FINE_TYPES = len(MHC_CHAIN_FINE_TYPES)

# Species (legacy 4-class, kept for backward compat with chain attribute classifier)
SPECIES = ["human", "mouse", "macaque", "other"]
SPECIES_TO_IDX = {sp: i for i, sp in enumerate(SPECIES)}
IDX_TO_SPECIES = {i: sp for i, sp in enumerate(SPECIES)}

# =============================================================================
# UNIFIED ORGANISM TAXONOMY (12-class)
# =============================================================================
# Single vocabulary for ALL organism classification: peptide source, MHC source,
# UniProt taxonomy, IEDB organism names, VDJdb antigen_species.
ORGANISM_CATEGORIES = [
    # Animal (MHC sources + host organisms) — first 6
    "human", "nhp", "murine", "other_mammal", "bird", "fish",
    # Other animals
    "other_vertebrate", "invertebrate",
    # Foreign (pathogens)
    "fungi", "bacteria", "viruses", "archaea",
]
ORGANISM_TO_IDX = {cat: i for i, cat in enumerate(ORGANISM_CATEGORIES)}
IDX_TO_ORGANISM = {i: cat for i, cat in enumerate(ORGANISM_CATEGORIES)}
N_ORGANISM_CATEGORIES = len(ORGANISM_CATEGORIES)

# MHC species uses the first 6 (animal subset)
MHC_SPECIES_CATEGORIES = ORGANISM_CATEGORIES[:6]  # human..fish
N_MHC_SPECIES = len(MHC_SPECIES_CATEGORIES)

# Foreignness: pathogens vs animals
FOREIGN_CATEGORIES = frozenset({"bacteria", "viruses", "fungi", "archaea"})


def normalize_organism(raw: Optional[str]) -> Optional[str]:
    """Unified normalizer: map any organism name to one of 12 categories.

    Used for BOTH peptide source organism AND MHC source species.
    Handles IEDB organism names, UniProt OS fields, VDJdb antigen_species,
    and MHC allele prefix inference.

    Returns None if unrecognizable (will result in mask=0, no supervision).
    """
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if not s:
        return None

    # Direct match first
    if s in ORGANISM_TO_IDX:
        return s

    # --- Human ---
    if "homo sapiens" in s or "human" in s:
        return "human"

    # --- Murine (mouse/rat) ---
    _murine = (
        "mus musculus", "mouse", "murine", "rattus", "rat",
        "c57bl", "balb/c", "h2-", "h-2",
    )
    if any(tok in s for tok in _murine):
        return "murine"

    # --- NHP (non-human primates) ---
    _nhp = (
        "macaca", "macaque", "nhp", "chimpanzee", "pan troglodytes",
        "gorilla", "orangutan", "pongo", "baboon", "papio",
        "aotus", "night monkey", "cercopithecus", "rhesus",
        "mamu-", "paan-", "aona-",
    )
    if any(tok in s for tok in _nhp):
        return "nhp"

    # --- Other mammals ---
    _other_mammal = (
        "bos taurus", "bovine", "cow", "cattle",
        "sus scrofa", "porcine", "pig", "swine",
        "equus", "equine", "horse",
        "ovis aries", "ovine", "sheep",
        "capra", "caprine", "goat",
        "canis", "canine", "dog",
        "felis", "feline", "cat",
        "rabbit", "oryctolagus",
        "bola-", "sla-", "dla-", "ela-", "ola-",
        "mammal",
    )
    if any(tok in s for tok in _other_mammal):
        return "other_mammal"

    # --- Bird ---
    _bird = (
        "gallus", "chicken", "bird", "avian", "aves",
        "duck", "turkey", "quail", "gaga-",
    )
    if any(tok in s for tok in _bird):
        return "bird"

    # --- Fish ---
    _fish = (
        "salmo", "salmon", "trout", "oncorhynchus",
        "danio", "zebrafish", "fish", "pisces",
    )
    if any(tok in s for tok in _fish):
        return "fish"

    # --- Viruses ---
    _virus = (
        "virus", "viral", "influenza", "sars", "cov",
        "hiv", "hcv", "hbv", "ebv", "cmv", "hsv", "vzv",
        "htlv", "dengue", "zika", "ebola", "measles",
        "hepatitis", "retrovirus", "coronavirus",
        "adenovirus", "papillomavirus", "herpes",
        "vaccinia", "poxvirus", "flavivirus",
        "paramyxovirus", "orthomyxovirus",
        "phage", "bacteriophage",
    )
    if any(tok in s for tok in _virus):
        return "viruses"

    # --- Bacteria ---
    _bacteria = (
        "mycobacterium", "tuberculosis", "escherichia", "e. coli",
        "staphylococcus", "streptococcus", "salmonella",
        "clostridium", "listeria", "helicobacter",
        "chlamydia", "borrelia", "treponema",
        "pseudomonas", "bacillus", "legionella",
        "neisseria", "rickettsia", "bartonella",
        "bacterium", "bacteria", "bacterial",
    )
    if any(tok in s for tok in _bacteria):
        return "bacteria"

    # --- Fungi ---
    _fungi = (
        "candida", "aspergillus", "cryptococcus",
        "coccidioides", "histoplasma", "blastomyces",
        "saccharomyces", "yeast", "fungus", "fungi", "fungal",
        "pneumocystis", "trichophyton",
    )
    if any(tok in s for tok in _fungi):
        return "fungi"

    # --- Archaea ---
    _archaea = (
        "archaea", "archaeal", "methanobacterium",
        "halobacterium", "sulfolobus", "thermococcus",
    )
    if any(tok in s for tok in _archaea):
        return "archaea"

    # --- Other vertebrate ---
    _other_vert = (
        "reptile", "reptilia", "amphibian", "amphibia",
        "frog", "xenopus", "turtle", "lizard", "snake",
        "alligator", "crocodile", "salamander",
    )
    if any(tok in s for tok in _other_vert):
        return "other_vertebrate"

    # --- Invertebrate ---
    _invertebrate = (
        "drosophila", "insect", "arthropod", "arachnid",
        "mosquito", "tick", "worm", "nematode", "mollusk",
        "caenorhabditis", "c. elegans", "invertebrate",
        "schistosoma", "plasmodium", "toxoplasma",
        "leishmania", "trypanosoma", "parasite",
    )
    if any(tok in s for tok in _invertebrate):
        return "invertebrate"

    return None

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
