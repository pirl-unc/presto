"""Vocabulary constants for Presto.

Defines amino acid vocabulary, chain types, cell types, MHC types, species,
organism categories, and biological compatibility matrices.
"""

import re
from typing import Any, Dict, Optional, Set, Tuple

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

#: Residues the tokenizer can actually encode, derived from AA_VOCAB so there
#: is exactly one source of truth. `X` is included: it is a real placeholder for
#: an unknown residue and has its own embedding.
#:
#: Anything outside this set cannot be represented at all. That includes both
#: annotation junk ("YXGEVXVSV + INDIST(X2, X6)") and genuine but unmodelled
#: residues -- selenocysteine `U` appears in real human selenoproteins and
#: reached the tokenizer through hitlist flanks, aborting training mid-epoch.
ENCODABLE_RESIDUES = frozenset(
    token for token in AA_VOCAB if len(token) == 1
)


def is_encodable_sequence(sequence: str) -> bool:
    """True when every residue can be tokenized. Empty is not encodable."""
    return bool(sequence) and not (set(sequence) - ENCODABLE_RESIDUES)


def drop_unencodable_sequence(sequence: str) -> str:
    """Blank an optional sequence that cannot be tokenized.

    Optional context (flanks, auxiliary sequences) degrades to "absent" rather
    than taking the whole row down with it: a flank with one unrepresentable
    residue still came from a row whose peptide and label are perfectly good,
    and many rows legitimately carry no flank at all. This mirrors what the
    merged-TSV loader already does via `_normalize_optional_aa_sequence`.
    """
    text = str(sequence or "").strip().upper()
    if not text or (set(text) - ENCODABLE_RESIDUES):
        return ""
    return text


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
# Appended entries keep existing indices stable, so an old checkpoint's
# embedding rows still mean what they meant; only the table grows. Without the
# stability/kinetics entries those rows type as "OTHER" and are indistinguishable
# from an unrecognized binding assay.
BINDING_ASSAY_TYPES = [
    "unknown",
    "KD",
    "KD_PROXY_IC50",
    "KD_PROXY_EC50",
    "IC50",
    "EC50",
    "OTHER",
    "T_HALF",
    "TM",
    "KOFF",
    "KON",
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

BINDING_ASSAY_PREP = [
    "unknown",
    "PURIFIED",
    "CELLULAR",
    "LYSATE",
    "BINDING_ASSAY",
    "OTHER",
]
BINDING_ASSAY_PREP_TO_IDX = {name: i for i, name in enumerate(BINDING_ASSAY_PREP)}
IDX_TO_BINDING_ASSAY_PREP = {i: name for i, name in enumerate(BINDING_ASSAY_PREP)}

BINDING_ASSAY_GEOMETRY = [
    "unknown",
    "COMPETITIVE",
    "DIRECT",
    "T_CELL_INHIBITION",
    "OTHER",
]
BINDING_ASSAY_GEOMETRY_TO_IDX = {name: i for i, name in enumerate(BINDING_ASSAY_GEOMETRY)}
IDX_TO_BINDING_ASSAY_GEOMETRY = {i: name for i, name in enumerate(BINDING_ASSAY_GEOMETRY)}

BINDING_ASSAY_READOUT = [
    "unknown",
    "RADIOACTIVITY",
    "FLUORESCENCE",
    "OTHER",
]
BINDING_ASSAY_READOUT_TO_IDX = {name: i for i, name in enumerate(BINDING_ASSAY_READOUT)}
IDX_TO_BINDING_ASSAY_READOUT = {i: name for i, name in enumerate(BINDING_ASSAY_READOUT)}

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


# ---------------------------------------------------------------------------
# Peptide excision machinery
# ---------------------------------------------------------------------------
# The machinery that cut a peptide out of its source protein. In-vivo entries
# are the biological pathways; the rest are in-vitro proteases whose cleavage
# rules are known exactly and are used to pin the corresponding readout.
#
# Rules mirror hitlist/data/bulk_proteomics/sources.yaml so the two stay in
# step. See docs/assay_learning_scheme.md and
# tasks/protease_detectability_spec.md.
EXCISION_MACHINERY = [
    "unknown",
    "proteasome",     # class I in vivo: a mixture over beta1/beta2/beta5
    "cathepsin",      # class II in vivo: endo/lysosomal
    "trypsin",        # C-term K/R, not before P
    "chymotrypsin",   # C-term F/W/Y/L/M, not before P  (MaxQuant Chymotrypsin+)
    "lysc",           # C-term K, P allowed
    "gluc",           # C-term E/D in bicarbonate buffer, not before P
]
EXCISION_MACHINERY_TO_IDX = {name: i for i, name in enumerate(EXCISION_MACHINERY)}
IDX_TO_EXCISION_MACHINERY = {i: name for i, name in enumerate(EXCISION_MACHINERY)}

# P1 residues each in-vitro protease cleaves after. Absent keys are learned
# rather than pinned: the proteasome is modeled as a convex mixture over the
# in-vitro profiles (its beta1/beta2/beta5 sites have exactly these
# specificities), and cathepsin is left free.
EXCISION_P1_RULES: Dict[str, str] = {
    "trypsin": "KR",
    "chymotrypsin": "FWYLM",
    "lysc": "K",
    "gluc": "ED",
}

# P1' residues that block cleavage for a given machinery ("not before P").
# LysC is the exception: its MaxQuant spec allows K-P.
EXCISION_P1_PRIME_BLOCKED: Dict[str, str] = {
    "trypsin": "P",
    "chymotrypsin": "P",
    "gluc": "P",
    "lysc": "",
}

# Machinery whose profile is pinned to a known rule rather than learned.
PINNED_EXCISION_MACHINERY = tuple(sorted(EXCISION_P1_RULES))

# The proteasome's three catalytic specificities map onto in-vitro analogs, so
# its profile is initialized as a blend of them rather than from scratch.
PROTEASOME_MIXTURE_COMPONENTS = ("trypsin", "chymotrypsin", "gluc")


def default_machinery_for_class(mhc_class: Optional[str]) -> str:
    """In-vivo machinery implied by a *declared* MHC class.

    Class I peptides are generated by the proteasome, class II by endo/
    lysosomal cathepsins. Shared by the collator and the dataset so the two
    cannot disagree about what an unlabelled row defaults to.

    This does NOT cover `Presto._resolve_machinery`, which faces a different
    problem: at inference no declared class need be present, so it falls back
    to *predicted* class probabilities. That is a deliberately separate rule,
    not a stale copy of this one -- it answers "what class does the model think
    this is" rather than "what class does the data say it is". Do not fold them
    together; a declared label must never be overridden by a prediction.
    """
    return "cathepsin" if str(mhc_class or "").strip().upper() == "II" else "proteasome"


def excision_machinery_index(name: Optional[str]) -> int:
    """Resolve a machinery name to its vocabulary index."""
    token = str(name or "").strip().lower()
    return EXCISION_MACHINERY_TO_IDX.get(token, EXCISION_MACHINERY_TO_IDX["unknown"])


# ---------------------------------------------------------------------------
# Peptide provenance and cellular state
# ---------------------------------------------------------------------------
# These replace the single flat `machinery` axis, which conflated two things
# that are not alternatives: what was done to the sample in the tube, and what
# state the cell was in. Because MHC ligands are never digested and shotgun
# proteins are extracted whole, one axis is perfectly anti-correlated with the
# corpus -- a pure corpus indicator with no within-corpus variation, which is
# why the in-vivo half never received gradient.
#
# See docs/model_io_contract.md for the permission rules attached to each.

# Tier 2: what was captured. Selects which branch computes the termini; must
# never be fed to a predictor as a feature.
PEPTIDE_SOURCES = ["unknown", "mhc", "protein"]
PEPTIDE_SOURCE_TO_IDX = {name: i for i, name in enumerate(PEPTIDE_SOURCES)}

# Tier 2: post-capture step. Non-`none` only when peptide_source == "protein".
ENZYMATIC_DIGESTS = ["none", "trypsin", "chymotrypsin", "lysc", "gluc"]
ENZYMATIC_DIGEST_TO_IDX = {name: i for i, name in enumerate(ENZYMATIC_DIGESTS)}

# Tier 3: what was applied to the cells. "Stimulus" rather than "cytokine
# state" because TLR agonists are PAMPs, not cytokines.
#
# `none` is a deliberate catch-all and it DOES conflate two things: cells with
# no recorded treatment, and cells whose condition simply was not recorded.
# That conflation is accepted rather than hidden. The earlier name for this
# slot was `basal`, which asserted a specific biological state -- resting cells
# do carry tonic interferon tone -- but we usually have no evidence for that
# claim, only the absence of a recorded treatment. `none` says the weaker,
# true thing. ~98.6% of class I elution rows land here, so treat it as "not
# known to be stimulated", never as a measured resting state.
#
# `ifn_type1` covers IFN-alpha and IFN-beta together: both bind IFNAR1/2 and
# drive the same ISGF3 program, including the immunoproteasome swap that
# matters here. IFN-gamma is type II through a different receptor (IFNGR1/2),
# so it stays separate. Spelled out rather than `ifn_ab`, which reads as
# "antibody" in an immunology codebase.
#
# Corpus volume per token, counted from hitlist `condition_category` (MS
# evidence) and pinned in tests/test_stimulus_vocabulary.py so drift is loud:
#
#   none                  ~3.9M   (1.44M of them an empty category string)
#   ifn_gamma               71,910
#   tlr                     34,434  (TLR_stimulation + bacterial/parasite)
#   cell_activation         37,929
#   ifn_type1               21,481  (all from infection_viral)
#   cytokine_unspecified     6,159
#   tnf_alpha                    0
#
# `tnf_alpha` matches no row today: no deposit in the corpus records a direct
# TNF-alpha treatment, and `IFN_alpha_treatment` / `IFN_beta_treatment` never
# appear either -- the type I signal arrives only as viral infection. That
# embedding row therefore receives no gradient. It is kept as declared
# headroom rather than deleted, because the mapping table must still route a
# TNF-alpha deposit correctly the day one appears.
PROCESSING_STIMULI = [
    "none",
    "ifn_gamma",
    "ifn_type1",
    "tnf_alpha",
    "tlr",
    # Lymphocyte activation (PMA/ionomycin, CD3/CD28, restimulation). Kept
    # distinct from the interferon and TLR axes because it acts through
    # PKC/NF-kB/NFAT rather than a PRR or interferon receptor. Folding it into
    # either would assert a signalling route the experiments do not support.
    # Appended last so existing stimulus indices are unchanged.
    "cell_activation",
    # "A cytokine was applied but the deposit does not name it." Distinct from
    # `none`, which means no treatment is known to have been applied.
    "cytokine_unspecified",
]

#: Superseded spellings, kept so older callers and saved records do not shift
#: to a different embedding row. Index order above is unchanged, so no
#: checkpoint migration is required.
LEGACY_STIMULUS_ALIASES = {"basal": "none", "ifn_ab": "ifn_type1"}
PROCESSING_STIMULUS_TO_IDX = {name: i for i, name in enumerate(PROCESSING_STIMULI)}

# Tier 3: antigen-processing-machinery perturbation, grouped by mechanism.
# Per-gene flags are too thin to learn individually (ERAP1 25 samples, TAP1 16,
# B2M 12, ~12 each for the rest), and a single boolean would make biologically
# opposite interventions identical -- B2M-null abolishes class I outright,
# while ERAP1-KO shifts only the N-terminus.
APM_PERTURBATIONS = [
    "none",
    "peptide_supply",     # TAP1/2, PSMB5/8/9/10 -- what reaches the ER
    "n_term_trimming",    # ERAP1/2 -- shifts the N-terminus specifically
    "loading_complex",    # TAPBP, CALR, CANX, PDIA3 -- stability, not cleavage
    "mhc_null",           # B2M -- abolishes class I presentation
    "class_ii_loading",   # HLA-DM/DO, CD74, CIITA -- editing and register
    "other",
]
APM_PERTURBATION_TO_IDX = {name: i for i, name in enumerate(APM_PERTURBATIONS)}

# hitlist APM gene flag -> mechanism group.
APM_GENE_TO_GROUP: Dict[str, str] = {
    "tap1": "peptide_supply", "tap2": "peptide_supply",
    "tap_inhibitor": "peptide_supply", "tap_deficient_line": "peptide_supply",
    "psmb5": "peptide_supply", "psmb8": "peptide_supply",
    "psmb9": "peptide_supply", "psmb10": "peptide_supply",
    "proteasome_inhibitor": "peptide_supply",
    "erap1": "n_term_trimming", "erap2": "n_term_trimming",
    "erap_inhibitor": "n_term_trimming",
    "tapbp": "loading_complex", "calr": "loading_complex",
    "canx": "loading_complex", "pdia3": "loading_complex",
    "ganab": "loading_complex", "sppl3": "loading_complex",
    "b2m": "mhc_null",
    "hla_dm": "class_ii_loading", "hla_do": "class_ii_loading",
    "cd74": "class_ii_loading", "ciita": "class_ii_loading",
    "rfx": "class_ii_loading", "bls": "class_ii_loading",
    "cathepsin": "class_ii_loading", "cathepsin_inhibitor": "class_ii_loading",
    "irf2": "other", "nlrc5": "other",
}

# hitlist condition_category -> stimulus. Cytokine treatment is an induction
# state, not an APM lesion, so it lives on its own axis.
#: hitlist ``condition_category`` -> stimulus token.
#:
#: Every category hitlist emits is listed, including those that map to `none`.
#: That is deliberate: an *absent* key means "hitlist grew a category we have
#: not reviewed", which `is_unmapped_condition` reports and the hitlist adapter
#: counts as `unmapped_condition_categories`. If categories that legitimately
#: mean "no stimulus" were left out, that signal would be buried in noise and a
#: genuinely new treatment would go unnoticed.
#:
#: That is not hypothetical: `SPPL3_perturbation` and `IRF2_perturbation` were
#: missing from this table while carrying 20,220 rows between them. The
#: predicate was already wired and the count was already in the returned stats
#: -- but nothing read that key, so a detector with no alarm attached let both
#: through. Keep the *reporting* path live, not just the check.
CONDITION_TO_STIMULUS: Dict[str, str] = {
    # --- direct cytokine treatment -------------------------------------
    "IFN_gamma_treatment": "ifn_gamma",
    "IFN_alpha_treatment": "ifn_type1",
    "IFN_beta_treatment": "ifn_type1",
    "TNF_alpha_treatment": "tnf_alpha",
    "TLR_stimulation": "tlr",
    # --- infection: endogenous induction, same processing consequence ---
    #
    # Viral infection is sensed by RIG-I/MDA5 (RNA) and cGAS-STING (DNA),
    # driving autocrine/paracrine type I interferon and the ISG program --
    # immunoproteasome subunits PSMB8/9/10, TAP1/2 and MHC-I upregulation.
    # That is the same processing remodelling recombinant IFN-beta produces,
    # which is what this axis encodes, so `ifn_type1` is the right bucket.
    #
    # It is induced rather than administered, and many of these viruses encode
    # interferon antagonists (influenza NS1, HIV Vpu, HCMV, EBV), so the
    # magnitude varies. The direction does not. hitlist's own categorization
    # supports reading this as the stimulated arm: it buckets UV/heat-
    # inactivated virus separately as `virus_inactivated_control`, which is the
    # paired comparator and maps to `none` below.
    "infection_viral": "ifn_type1",
    # Listeria, Salmonella, Chlamydia, Pseudomonas, Mycobacterium, Borrelia,
    # Theileria, Toxoplasma, Leishmania, Plasmodium -- the organisms hitlist
    # matches here. All engage TLRs (LPS/TLR4 for the gram-negatives,
    # lipoproteins/TLR2 for Borrelia and Mtb, flagellin/TLR5). Several
    # intracellular ones also induce type I IFN via cytosolic sensing, but TLR
    # engagement is what the set shares.
    "infection_bacterial_or_parasite": "tlr",
    # PMA/ionomycin, CD3/CD28, in vitro activation, restimulation.
    "cell_activation": "cell_activation",
    # --- categories that genuinely mean "no stimulus applied" ----------
    "unperturbed": "none",
    # The control arm of an infection study. UV/heat-inactivated virions still
    # carry some PAMPs, so this is not perfectly inert, but it is the
    # experimenters' intended comparator for `infection_viral` above -- and
    # pairing the two is what gives this axis a real contrast.
    "virus_inactivated_control": "none",
    # Vector-based gene delivery and plasmid transfection. Immunologically not
    # an infection, and the manipulation is the experimental variable rather
    # than a processing stimulus.
    "transduction": "none",
    "transfection": "none",
    "CIITA_transduction": "none",
    "transplant": "none",
    "biomaterial_contact": "none",
    "drug_exposure": "none",
    "metabolic_stress": "none",
    "labeling_control": "none",
    "other_perturbation": "none",
    # --- known stimulus, unknown identity --------------------------------
    # 6,159 rows. These cells WERE treated with a cytokine; the deposit just
    # does not say which. Mapping them to `none` would be the one case where
    # that token states something we know to be false -- `none` means "not
    # known to be stimulated", and here we do know. Given its own row rather
    # than guessed at: folding it into `ifn_gamma` (the most common named
    # treatment) would fabricate a specificity the source does not support.
    "cytokine_treatment_generic": "cytokine_unspecified",
    # --- APM gene perturbations -----------------------------------------
    # These name a knockout, not a stimulus. They are carried on the separate
    # `apm_perturbation` axis (sourced from `apm_genes_perturbed`), verified
    # populated -- e.g. ERAP1_perturbation rows arrive as `n_term_trimming`.
    # Listed here only so they do not register as unreviewed categories.
    "MHC-I_loss_B2M": "none",
    "TAP_perturbation": "none",
    "tapasin_perturbation": "none",
    "ERAP1_perturbation": "none",
    "HLA-DM_perturbation": "none",  # arrives as apm=class_ii_loading
    "ERAP2_perturbation": "none",
    "ERAP_inhibitor": "none",
    "PLC_chaperone_perturbation": "none",
    "immunoproteasome_perturbation": "none",
    "proteasome_inhibitor": "none",
    # Both carry real corpus volume (SPPL3 14,906 rows, IRF2 5,314) and both
    # were missing here until `is_unmapped_condition` was wired into ingest.
    # SPPL3 is an intramembrane protease whose loss shifts MHC-II peptide
    # loading via glycosphingolipid metabolism; IRF2 is a transcriptional
    # repressor of the interferon-stimulated program. Both are standing
    # lesions in the machinery, not treatments applied to the cells, so they
    # belong on `apm_perturbation` and read as `none` on this axis.
    "SPPL3_perturbation": "none",
    "IRF2_perturbation": "none",
}


def peptide_source_index(name: Optional[str]) -> int:
    return PEPTIDE_SOURCE_TO_IDX.get(
        str(name or "").strip().lower(), PEPTIDE_SOURCE_TO_IDX["unknown"]
    )


def enzymatic_digest_index(name: Optional[str]) -> int:
    return ENZYMATIC_DIGEST_TO_IDX.get(
        str(name or "").strip().lower(), ENZYMATIC_DIGEST_TO_IDX["none"]
    )


def processing_stimulus_index(name: Optional[str]) -> int:
    """Index for a stimulus token, resolving superseded spellings.

    Legacy names are translated rather than silently defaulted: without this,
    a saved record carrying `ifn_ab` would land on the `none` row and be
    scored as an unstimulated sample.
    """
    token = str(name or "").strip().lower()
    token = LEGACY_STIMULUS_ALIASES.get(token, token)
    return PROCESSING_STIMULUS_TO_IDX.get(
        token, PROCESSING_STIMULUS_TO_IDX["none"]
    )


def apm_perturbation_index(name: Optional[str]) -> int:
    return APM_PERTURBATION_TO_IDX.get(
        str(name or "").strip().lower(), APM_PERTURBATION_TO_IDX["none"]
    )


def apm_group_for_genes(genes: Optional[Any]) -> str:
    """Mechanism group for a set of perturbed APM genes.

    Ordered by severity when several are perturbed together: a B2M-null line
    has no class I presentation at all, so that dominates whatever else was
    knocked out; peptide supply dominates trimming, and so on.
    """
    if not genes:
        return "none"
    if isinstance(genes, str):
        tokens = [g.strip().lower() for g in genes.replace(",", ";").split(";") if g.strip()]
    else:
        tokens = [str(g).strip().lower() for g in genes if str(g).strip()]
    groups = {APM_GENE_TO_GROUP.get(token) for token in tokens}
    groups.discard(None)
    if not groups:
        return "none"
    for candidate in (
        "mhc_null", "peptide_supply", "n_term_trimming",
        "loading_complex", "class_ii_loading", "other",
    ):
        if candidate in groups:
            return candidate
    return "other"


def stimulus_for_condition(condition: Optional[str]) -> str:
    """Map a hitlist ``condition_category`` to a stimulus token.

    Unmapped and missing conditions both fall back to ``none``. That is the
    accepted conflation described on PROCESSING_STIMULI -- but an unmapped
    *non-empty* category is different from a missing one: it means hitlist grew
    a condition this table does not know about, and silently folding it into
    ``none`` would hide a real treatment. Callers that care can detect this
    with :func:`is_unmapped_condition`.
    """
    return CONDITION_TO_STIMULUS.get(str(condition or "").strip(), "none")


def is_unmapped_condition(condition: Optional[str]) -> bool:
    """True when a condition was recorded but this table has no entry for it.

    Distinguishes "nobody wrote anything down" from "hitlist added a treatment
    category we have not mapped yet". The second is a maintenance signal: it
    means real stimulated samples are being scored as unstimulated.
    """
    text = str(condition or "").strip()
    return bool(text) and text not in CONDITION_TO_STIMULUS
