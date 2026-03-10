# MHC Sequence Processing & Representation: Implementation Plan

Complete specification for how MHC sequences should be processed from raw source data through to model input tensors, including handling of novel user-specified alleles at inference time.

**Goal:** A unified, class-agnostic groove representation where:
- MHC-I: `[groove_half_1 (~91aa)] [SEP] [groove_half_2 (~93aa)]`
- MHC-II: `[alpha1_groove (~80aa)] [SEP] [beta1_groove (~90aa)]`

Both classes use the same two-segment structure. The two halves are structurally cognate (alpha1 = floor+helix of one groove wall; alpha2/beta1 = floor+helix of the other).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Index Build Pipeline](#2-index-build-pipeline)
3. [Groove Extraction Algorithm](#3-groove-extraction-algorithm)
4. [Augmented MHC Index Schema](#4-augmented-mhc-index-schema)
5. [Inference-Time Processing (New Alleles)](#5-inference-time-processing-new-alleles)
6. [Model Input Representation](#6-model-input-representation)
7. [Training-Time Augmentation](#7-training-time-augmentation)
8. [Auxiliary Tasks](#8-auxiliary-tasks)
9. [Error Taxonomy & Handling](#9-error-taxonomy--handling)
10. [Implementation Phases](#10-implementation-phases)
11. [File Manifest](#11-file-manifest)
12. [Validation Criteria](#12-validation-criteria)

---

## 1. Architecture Overview

### Current State

```
User allele string
    → allele_resolver.normalize_allele_name()
    → mhc_index.resolve_alleles()         # lookup in CSV
    → full raw sequence (variable: 70-400 aa)
    → collate.py tokenizes into mhc_a_tok, mhc_b_tok
    → model sees: [SEG_MHC_A: full alpha chain] [SEG_MHC_B: full beta/B2M]
    → groove_bias (sigmoid decay) softly downweights non-groove positions
```

**Problems with current state:**
1. Signal peptide presence is inconsistent (some seqs have it, some don't)
2. MHC-I and MHC-II have fundamentally different groove architectures but use the same segment layout
3. B2M occupies SEG_MHC_B for Class I but contributes nothing to peptide binding — it's wasted capacity
4. Groove bias uses fixed cutoffs that can't adapt to variable-length inputs
5. Positional encoding is plain sequential — position 1 in alpha1 and position 91 in alpha2 have no structural relationship despite being cognate groove floors

### Target State

```
User allele string (or raw sequence)
    → groove_extractor.parse() dispatches by class:
        Class I:  full alpha → [alpha1_groove, alpha2_groove] via Cys-anchor
        Class II: alpha chain → alpha1_groove via Cys-anchor
                  beta chain  → beta1_groove via Cys-anchor
    → model sees: [SEG_GROOVE_1: ~91aa] [SEP] [SEG_GROOVE_2: ~93aa]
    → both classes use identical segment structure
    → groove bias is unnecessary (non-groove residues already removed)
    → positional encoding within each half directly maps to structural position
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Groove vs. full sequence | Extract groove only | Removes SP ambiguity, removes irrelevant Ig/TM/cyto domains, equalizes Class I and II |
| B2M handling | Drop B2M from binding input | B2M doesn't contact peptide in Class I; provides no signal. Can keep as optional context. |
| Boundary detection | Cys-anchor heuristic | Works across all jawed vertebrates (450M years conserved); no MSA/alignment needed |
| Fallback for no-Cys sequences | Alpha3 Cys anchor → fixed-offset | 97.4% primary success, alpha3 fallback covers another 0.2%, remaining 2.6% are non-functional |
| MHC-II groove construction | alpha1 from alpha chain + beta1 from beta chain | Mirrors experimental scMHC-II constructs |

---

## 2. Index Build Pipeline

### 2.1 Existing Build (no changes needed)

`data/mhc_index.py::build_mhc_index()` already:
- Loads IMGT + IPD-MHC FASTA files
- Filters nucleotide sequences
- Normalizes allele names via mhcgnomes
- Deduplicates by normalized key
- Outputs 8-column CSV: `allele_raw, normalized, gene, mhc_class, species, source, seq_len, sequence`

### 2.2 New: Post-Build Groove Augmentation

After `build_mhc_index()`, run `augment_mhc_index()` to add groove columns.

```python
def augment_mhc_index(index_csv: str, output_csv: str) -> AugmentStats:
    """
    Read the MHC index, run groove extraction on every entry,
    write augmented index with new columns.
    """
```

**New columns added** (see Section 4 for full schema):
- `mature_start` — inferred signal peptide cleavage position
- `groove_half_1` — first groove domain sequence (alpha1 for Class I; alpha1 from alpha chain for Class II)
- `groove_half_2` — second groove domain sequence (alpha2 for Class I; beta1 from beta chain for Class II)
- `groove_status` — parse result: `ok`, `alpha3_fallback`, `no_cys_pairs`, `no_alpha2_pair`, `groove_absent`, `too_short`
- `is_null` — True if allele suffix is N (null / frameshifted)
- `is_questionable` — True if allele suffix is Q
- `is_pseudogene` — True if allele suffix is Ps

### 2.3 Also Fix During Build

These are independent cleanups that happen in `build_mhc_index()`:
- Strip trailing `X` from IMGT sequences: `seq = seq.rstrip("X")` (DQ5)
- Fix dog B2M sequence in `b2m_sequences.csv` (DQ1)
- Map remaining IPD species prefix codes (DQ6)

---

## 3. Groove Extraction Algorithm

This is the core algorithm. It must work on arbitrary sequences from any jawed vertebrate, with or without signal peptide, and degrade gracefully on fragments, null alleles, and non-classical molecules.

### 3.1 Shared Primitives

#### 3.1.1 `find_cys_pairs(seq, min_sep=48, max_sep=72) → List[(c1, c2, sep)]`

Find all Cys-Cys pairs with Ig-fold-compatible separation. The conserved intrachain disulfide bond in all Ig-fold domains has a separation of 48-72 residues (typical: 56-65). Returns pairs sorted by c1 position.

#### 3.1.2 `classify_cys_pair(c1, seq) → "alpha2" | "alpha3" | "ig_generic"`

Distinguish alpha2 groove Cys from alpha3 Ig Cys using local sequence context:
- **Alpha3 signature:** `CW[x]LGF` motif immediately following the Cys (the Trp is position c1+1). This "CWALGFY" motif is the hallmark of MHC Ig-fold domains.
- **Alpha2 groove:** No such motif. Local context is variable.

This is a disambiguation helper, not the primary detection method.

### 3.2 Class I: `parse_class_i(seq, allele, gene) → GrooveResult`

**Input:** Full MHC-I alpha chain sequence (70-400 aa), with or without signal peptide.
**Output:** `GrooveResult` with `groove_half_1` (alpha1), `groove_half_2` (alpha2), `mature_start`, `status`.

```
Algorithm:

1. Find all Cys pairs with Ig-fold separation [48, 72]

2. PRIMARY PATH — Alpha2 anchor:
   a. Filter pairs to those with c1 in [60, 180]
      (alpha2 Cys1 is at mature position ~101; with SP up to ~24, raw range is ~85-125.
       Allow wider [60,180] for fragments starting mid-sequence.)
   b. Skip pairs where classify_cys_pair identifies alpha3 (CW motif)
   c. First qualifying pair = alpha2 disulfide
   d. Infer mature_start = c1 - 101 (clamped to ≥0)
   e. alpha2_start = c1 - 10  (Cys is ~10 residues into alpha2)
   f. alpha2_end = min(c2 + 20, len(seq))
   g. alpha1 = seq[mature_start : alpha2_start]
   h. alpha2 = seq[alpha2_start : alpha2_end]
   i. status = "ok"

3. FALLBACK PATH — Alpha3 anchor (if no alpha2 pair found):
   a. Look for pairs with c1 ≥ 180 (alpha3 region)
   b. Infer mature_start = c1 - 203 (alpha3 Cys1 mature position ~203)
   c. Use fixed domain sizes: alpha1 = seq[mature_start : mature_start+91]
   d. alpha2 = seq[mature_start+91 : c1-20]  (end before alpha3 Ig domain)
   e. status = "alpha3_fallback"

4. FAILURE PATHS:
   a. No Cys pairs at all → status = "no_cys_pairs"
   b. All Cys pairs are alpha3-like (CW motif) and c1 < 180 → status = "groove_absent"
      (The protein encodes only Ig-like domains, no groove — e.g., rat Rano-S2, Rano-T24-1)
   c. Sequence too short (<70 aa) → status = "too_short"
```

**Validated performance:**
| Status | Count | % | Description |
|--------|-------|---|-------------|
| ok | 35,532 | 97.2% | Alpha2 Cys anchor found |
| alpha3_fallback | 74 | 0.2% | Alpha3 anchor used |
| no_cys_pairs | 956 | 2.6% | 90% null alleles, rest are truncated fragments |
| no_alpha2_pair | 10 | <0.1% | Pseudogenes with early Cys |

**Alpha1 length distribution (successful):** median=91, range 51-91 (tight peak at 89-91)
**Alpha2 length distribution (successful):** median=93, range 70-102 (tight peak at 92-93)

### 3.3 Class II Alpha Chain: `parse_class_ii_alpha(seq, allele, gene) → GrooveResult`

**Input:** Full MHC-II alpha chain (e.g., HLA-DRA, HLA-DQA1).
**Output:** `GrooveResult` with `groove_half_1` = alpha1 domain.

```
Algorithm:

1. Find all Cys pairs with Ig-fold separation [48, 72]

2. The alpha2 Ig-fold disulfide (alpha chain's Ig domain, NOT the groove)
   has Cys1 at mature position ~94 with sep ~56.

3. Look for the FIRST Cys pair with c1 in [60, 150] (allowing for SP).
   This is the alpha2 Ig-fold Cys, marking the END of the groove domain.

4. Infer mature_start = c1 - 94  (alpha2 Ig Cys1 mature pos)
5. groove_alpha1 = seq[mature_start : c1 - 6]
   (alpha1 groove ends ~6 residues before the Ig domain Cys)
6. Expected length: ~76-80 aa

7. If no Cys pair found: same failure taxonomy as Class I.
```

**Note:** Class II alpha1 is shorter than Class I alpha1 because MHC-II alpha only contributes one wall of the groove (floor + one helix), whereas MHC-I alpha1 contributes a full wall.

### 3.4 Class II Beta Chain: `parse_class_ii_beta(seq, allele, gene) → GrooveResult`

**Input:** Full MHC-II beta chain (e.g., HLA-DRB1, HLA-DQB1).
**Output:** `GrooveResult` with `groove_half_2` = beta1 domain.

```
Algorithm:

1. Find all Cys pairs with Ig-fold separation [48, 72]

2. The beta1 groove domain has a conserved disulfide with Cys1 at
   mature position ~15 and sep ~64.
   The beta2 Ig-fold disulfide has Cys1 at mature position ~117 with sep ~56.

3. IMPORTANT: Signal peptide can contain spurious Cys pairs.
   Example: DRB1 SP = "MVCLKLPGGSCM..." has Cys at positions 2 and 10 (sep=8,
   not Ig-fold range, but other SPs may have wider-spaced Cys).

4. Look for Cys pairs with c1 in [20, 85] (mature beta1 Cys1 is at ~15;
   with SP up to ~30, raw range is ~45-50. Use [20, 85] for safety).
   The first qualifying pair = beta1 groove disulfide.

5. Infer mature_start from beta2 Ig Cys pair (c1 at mature ~117):
   Look for second pair with c1 > first_pair.c2 + 10, c1 in [80, 170].
   mature_start = beta2_cys1 - 117.

6. beta1_groove = seq[mature_start : beta2_cys1 - 20]
   (end ~20 residues before the Ig domain Cys)
7. Expected length: ~90-95 aa

8. If only one pair found and it looks like beta2 (CW motif), use it to
   infer mature_start and cut beta1 at fixed offset.
```

### 3.5 Result Data Structure

```python
@dataclass
class GrooveResult:
    allele: str
    gene: str
    mhc_class: str           # "I" or "II"
    chain: str                # "alpha" (Class I or II alpha) or "beta" (Class II beta)
    seq_len: int              # full input sequence length
    mature_start: int         # inferred signal peptide cleavage position
    groove_seq: str           # extracted groove domain(s)
    groove_half_1: str        # alpha1 (Class I) or alpha1 from alpha chain (Class II)
    groove_half_2: str        # alpha2 (Class I) or beta1 from beta chain (Class II)
    groove_h1_len: int
    groove_h2_len: int
    anchor_type: str          # "alpha2_cys", "alpha3_cys", "beta1_cys", "beta2_cys", "fixed_offset"
    anchor_cys1: Optional[int]
    anchor_cys2: Optional[int]
    anchor_sep: Optional[int]
    status: str               # "ok", "alpha3_fallback", "no_cys_pairs", "groove_absent", "too_short"
    flags: List[str]          # warnings, e.g. "long_sp(48)", "alpha1_short(55)"
```

### 3.6 Unified Dispatch

```python
def extract_groove(
    seq: str,
    mhc_class: str,
    chain: str = "alpha",
    allele: str = "",
    gene: str = "",
) -> GrooveResult:
    """
    Top-level dispatcher. Routes to class/chain-specific parser.

    Args:
        seq: amino acid sequence (full or mature, any length)
        mhc_class: "I" or "II"
        chain: "alpha" (default) or "beta" (only for Class II)
        allele: for metadata/debugging
        gene: for metadata/debugging

    Returns:
        GrooveResult with extracted groove domains and diagnostics
    """
    if len(seq) < 70:
        return GrooveResult(status="too_short", ...)
    if mhc_class == "I":
        return parse_class_i(seq, allele, gene)
    elif mhc_class == "II" and chain == "alpha":
        return parse_class_ii_alpha(seq, allele, gene)
    elif mhc_class == "II" and chain == "beta":
        return parse_class_ii_beta(seq, allele, gene)
```

---

## 4. Augmented MHC Index Schema

### Current columns (keep all):
`allele_raw, normalized, gene, mhc_class, species, source, seq_len, sequence`

### New columns:

| Column | Type | Description |
|--------|------|-------------|
| `mature_start` | int | Inferred SP cleavage position (0 if no SP or already mature) |
| `groove_half_1` | str | First groove domain (alpha1). Empty if extraction failed. |
| `groove_half_2` | str | Second groove domain (alpha2 or beta1). Empty if extraction failed. For Class II, this is only filled on the beta chain row; alpha chain row has groove_half_1 only. |
| `groove_status` | str | `ok`, `alpha3_fallback`, `no_cys_pairs`, `groove_absent`, `too_short` |
| `groove_flags` | str | Comma-separated warning flags |
| `is_null` | bool | Allele has N suffix (frameshifted, never expressed) |
| `is_questionable` | bool | Allele has Q suffix (expression uncertain) |
| `is_pseudogene` | bool | Allele has Ps suffix |
| `is_functional` | bool | Derived: `not (is_null or is_pseudogene) and groove_status in ("ok", "alpha3_fallback")` |

### Class II Groove Assembly

Class II groove requires two separate index rows (alpha chain + beta chain). The groove is assembled at data-loading time:
- `groove_half_1` = alpha chain's `groove_half_1` (alpha1 domain)
- `groove_half_2` = beta chain's `groove_half_2` (beta1 domain)

This means `resolve_alleles()` for a Class II sample must resolve BOTH chains and combine their groove halves.

---

## 5. Inference-Time Processing (New Alleles)

When a user provides a novel MHC at inference time, the system must handle three input formats:

### 5.1 Input Formats

```python
# Format A: Allele name (most common)
predict(peptide="SIINFEKL", mhc_allele="HLA-A*02:01")

# Format B: Raw sequence (novel MHC, no name)
predict(peptide="SIINFEKL", mhc_sequence="MAVMAPRTLVL...")

# Format C: Allele name + explicit class
predict(peptide="SIINFEKL", mhc_allele="HLA-DRB1*04:01", mhc_class="II")
```

### 5.2 Resolution Flow

```
Input allele or sequence
    │
    ├─ Allele name provided?
    │   ├─ YES → Look up in augmented MHC index
    │   │   ├─ Found with groove_status="ok"?
    │   │   │   └─ Use pre-computed groove_half_1, groove_half_2 ✓
    │   │   ├─ Found but groove_status is failure?
    │   │   │   └─ Fall back to raw sequence → run extract_groove() live
    │   │   └─ Not found?
    │   │       ├─ Try progressive truncation (A*02:01:01:01 → A*02:01:01 → A*02:01)
    │   │       └─ Still not found → error: "Unknown allele"
    │   └─ NO (raw sequence provided)
    │       └─ Infer mhc_class from user input or from sequence length heuristic
    │           └─ Run extract_groove(seq, mhc_class) live
    │
    ├─ Assemble groove representation:
    │   ├─ Class I:  groove = [groove_half_1] [SEP] [groove_half_2]
    │   └─ Class II: groove = [alpha1_groove] [SEP] [beta1_groove]
    │       (requires both alpha and beta chain; beta defaults if missing:
    │        for HLA-DRB1 → use HLA-DRA*01:01 as default alpha)
    │
    └─ Tokenize groove → model input
```

### 5.3 Class II Default Pairing

For Class II, the user may specify only one chain (typically the beta chain, since it's polymorphic). Default pairings:

| User provides | Default partner | Rationale |
|---------------|-----------------|-----------|
| HLA-DRB1*xx | HLA-DRA*01:01 | DRA is virtually monomorphic |
| HLA-DQB1*xx | Error: must specify DQA1 | DQA1 is polymorphic |
| HLA-DPB1*xx | Error: must specify DPA1 | DPA1 is polymorphic |
| DRA only | Error: need beta chain | Beta is the polymorphic partner |

This pairing logic already exists in `allele_resolver.py` for B2M; extend the same pattern for Class II alpha chains.

### 5.4 Live Groove Extraction

When the allele isn't in the index (or the user provides a raw sequence), groove extraction runs at inference time. This is the same `extract_groove()` function used during index build. It must be:
- **Fast:** No file I/O, no alignment. Just Cys scanning + arithmetic. O(L) where L = sequence length.
- **Deterministic:** Same input always gives same output (no random augmentation at inference).
- **Graceful:** If extraction fails, fall back to truncated raw sequence with a warning.

```python
def prepare_mhc_input(
    allele: Optional[str] = None,
    sequence: Optional[str] = None,
    mhc_class: Optional[str] = None,
    index: Optional[MHCIndex] = None,
) -> MHCInput:
    """
    Top-level entry point for preparing MHC model input.
    Used by both training pipeline and inference API.

    Returns MHCInput with:
        groove_half_1: str
        groove_half_2: str
        mhc_class: str ("I" or "II")
        method: str ("index_lookup", "live_extraction", "fallback_truncation")
        warnings: List[str]
    """
```

### 5.5 Fallback: Raw Sequence Truncation

If groove extraction fails entirely (no Cys pairs, non-functional allele), fall back to:
- **Class I:** `groove_half_1 = seq[:91]`, `groove_half_2 = seq[91:184]` (fixed offset from sequence start)
- **Class II alpha:** `groove_half_1 = seq[:80]`
- **Class II beta:** `groove_half_2 = seq[:90]`

This is wrong for sequences with signal peptides but acceptable as a last resort. Log a warning.

---

## 6. Model Input Representation

### 6.1 New Segment Structure

Replace the current 5-segment scheme:

```python
# CURRENT (remove):
SEG_NFLANK = 0
SEG_PEPTIDE = 1
SEG_CFLANK = 2
SEG_MHC_A = 3    # full alpha chain
SEG_MHC_B = 4    # full beta chain / B2M

# NEW:
SEG_NFLANK = 0
SEG_PEPTIDE = 1
SEG_CFLANK = 2
SEG_GROOVE_1 = 3  # alpha1 (Class I) or alpha1 from alpha chain (Class II)
SEG_GROOVE_2 = 4  # alpha2 (Class I) or beta1 from beta chain (Class II)
```

**B2M is dropped from binding input.** It contributes no peptide-contact residues. If B2M information is needed for other tasks (e.g., cell-surface stability), it can be added as a separate context signal or auxiliary input, but it should not occupy groove attention bandwidth.

### 6.2 Positional Encoding

Each groove half gets its own positional embedding:

```python
self.groove_1_pos = nn.Embedding(120, d_model)  # alpha1: up to ~100 aa
self.groove_2_pos = nn.Embedding(120, d_model)  # alpha2/beta1: up to ~100 aa
```

**Key insight:** Because both halves now start at position 0, the positional encoding naturally captures the structural correspondence. Position 1 in groove_half_1 is the first residue of the groove floor in both Class I and Class II. Position 50 in groove_half_1 is approximately the start of the groove helix in both classes.

### 6.3 Groove Bias: No Longer Needed

The current sigmoid-decay groove bias (`groove_bias_a`, `groove_bias_b`) exists to downweight non-groove residues (Ig domains, transmembrane, cytoplasmic tail). With groove extraction, these residues are already removed. **Delete the groove bias computation entirely.**

This simplifies the model and removes 4 learned parameters that were trying to approximate what the preprocessing now handles exactly.

### 6.4 Max Sequence Lengths

```python
# CURRENT:
max_mhc_len = 400  # accommodates full precursor + SP + TM + cyto

# NEW:
max_groove_len = 120  # generous ceiling for a single groove half
                       # (actual max observed: 102 aa for alpha2)
```

This reduces MHC token count from ~800 (two chains × 400) to ~240 (two halves × 120), a **3.3× reduction** in MHC sequence length. This directly improves attention efficiency and memory usage.

### 6.5 Collation Changes

In `data/collate.py`:

```python
# PrestoSample changes:
@dataclass
class PrestoSample:
    peptide: str = ""
    groove_half_1: str = ""    # was mhc_a
    groove_half_2: str = ""    # was mhc_b
    mhc_class: Optional[str] = None
    # ... rest unchanged

# PrestoBatch changes:
@dataclass
class PrestoBatch:
    groove_1_tok: torch.Tensor   # was mhc_a_tok
    groove_2_tok: torch.Tensor   # was mhc_b_tok
    mhc_class: List[Optional[str]]
    # ... rest unchanged
```

### 6.6 Latent Segment Access

Update LATENT_SEGMENTS:

```python
LATENT_SEGMENTS = {
    "processing": ["nflank", "peptide", "cflank"],
    "ms_detectability": ["peptide"],
    "species_of_origin": ["peptide"],
    "pmhc_interaction": ["peptide", "groove_1", "groove_2"],  # was mhc_a, mhc_b
    "recognition": ["peptide"],
}
```

### 6.7 MHC Class Inference

The compositional class inference (per-chain type classification → class probability) needs adjustment. Currently it classifies mhc_a and mhc_b independently then composes. With groove halves, the class signal comes from:
- Groove half lengths (alpha1 ~91aa for Class I vs ~80aa for Class II alpha)
- Sequence content (the two halves come from different chains in Class II)

Options (in order of preference):
1. **Pass mhc_class as explicit input** during both training and inference. The allele resolver already knows the class. Don't ask the model to infer it from sequence alone.
2. **Lightweight classifier** on groove_half_1 length + first few residues (if explicit class unavailable).
3. Keep compositional inference but train on groove halves instead of full chains.

**Recommendation:** Option 1. Always pass mhc_class explicitly. It's known for every allele in the index and must be specified by the user for novel sequences. Remove the compositional inference complexity.

---

## 7. Training-Time Augmentation

### 7.1 Signal Peptide Augmentation (REMOVED)

The original plan (DQ8) proposed randomly prepending/removing the signal peptide during training to teach invariance. **With groove extraction, this is unnecessary** — the SP is always stripped. Remove this TODO item.

### 7.2 Groove Truncation Augmentation

To make the model robust to slightly different extraction boundaries (e.g., user provides a manually trimmed sequence):

```python
# During training, with probability p=0.1:
#   Randomly trim 0-3 residues from the N-terminus of groove_half_1
#   Randomly trim 0-3 residues from the C-terminus of groove_half_2
# This teaches the model that exact boundary positions don't matter.
```

### 7.3 Training Data Filtering

Exclude from binding/presentation training:
- `is_null = True` (frameshifted proteins)
- `is_pseudogene = True`
- `groove_status not in ("ok", "alpha3_fallback")`

Keep in index for completeness (user might query these alleles), but don't train on them.

Optionally exclude `is_questionable = True` (Q-suffix). These may or may not reach the cell surface. Could include with a reduced sample weight (0.5×).

---

## 8. Auxiliary Tasks

These are optional model heads that provide structural priors. They improve MHC representation learning but are not required for the groove extraction to work.

### 8.1 Domain Classification (MR1 from audit TODO)

Per-residue prediction of structural domain. Supervised when ground truth available (known gene family), unsupervised for novel sequences.

```python
self.groove_domain_head = nn.Linear(d_model, 3)
# Classes: GROOVE_FLOOR, GROOVE_HELIX, CONNECTOR
# With groove extraction, we no longer need IG_DOMAIN, SIGNAL_PEPTIDE, OTHER
# since those residues are removed.
```

Ground truth labels for groove halves (from IMGT G-DOMAIN numbering):
```
Groove half 1 (alpha1): floor=residues 1-49, helix=residues 50-90
Groove half 2 (alpha2): floor=residues 1-49, helix=residues 50-92
```
These positions are the same for all Class I alleles and for Class II alpha1/beta1 domains.

### 8.2 Contact Importance (MR2 from audit TODO)

Per-residue prediction of peptide contact probability. Supervised with NetMHCpan pseudosequence positions when available.

```python
self.groove_contact_head = nn.Linear(d_model, 1)  # sigmoid → contact prob
# Loss: BCE(pred, label) when labels available for known gene families
# Labels: 1 at ~34 NetMHCpan pseudosequence positions (in groove coordinates)
```

The contact predictions can modulate attention weights in the binding cross-attention:
```python
contact_weights = sigmoid(self.groove_contact_head(h_groove))
h_groove_weighted = h_groove * (1.0 + contact_weights)
```

### 8.3 Groove Bias: DELETED

The sigmoid-decay groove bias is no longer needed (Section 6.3). Remove `groove_bias_a`, `groove_bias_b` parameters and all associated computation.

---

## 9. Error Taxonomy & Handling

Every failure mode has an explicit status code, a severity level, and a prescribed action.

### 9.1 Index Build Errors

| Status | Severity | Count in current index | Action |
|--------|----------|----------------------|--------|
| `ok` | None | 35,532 (97.2%) | Use groove_half_1, groove_half_2 |
| `alpha3_fallback` | Warning | 74 (0.2%) | Use groove (slightly less precise boundaries) |
| `no_cys_pairs` | Expected | 956 (2.6%) | 90% null alleles. Mark is_functional=False. Keep in index. |
| `no_alpha2_pair` | Expected | 10 (<0.1%) | Pseudogenes/fragments. Mark is_functional=False. |
| `groove_absent` | Info | ~5 (est.) | Ig-only genes (e.g., Rano-S2). Detected by CW motif near Cys. Mark is_functional=False. |
| `too_short` | Expected | 0 (filtered by MIN_MHC_SEQUENCE_LEN=70) | Already filtered during index build |

### 9.2 Inference-Time Errors

| Situation | Behavior | User-visible |
|-----------|----------|-------------|
| Known allele, groove pre-computed | Return cached groove | Silent |
| Known allele, groove failed at build time | Return raw sequence truncation + warning | Warning: "Allele X is flagged as non-functional. Groove extraction failed. Using raw sequence fallback." |
| Unknown allele, sequence provided, groove extraction succeeds | Return extracted groove | Silent |
| Unknown allele, sequence provided, groove extraction fails | Return raw sequence truncation + warning | Warning: "Could not detect groove boundaries in provided sequence. Using positional heuristic." |
| Unknown allele, no sequence provided | Error | Error: "Allele X not found in MHC index. Provide a sequence or use a known allele name." |
| Class II, only one chain provided | Error or default | For DRB1: auto-pair with DRA*01:01. For DQB1/DPB1: Error: "Must specify alpha chain." |

### 9.3 Logging

All groove extraction events should be logged at appropriate levels:
- `DEBUG`: Successful extraction with details (Cys positions, domain sizes)
- `WARNING`: Alpha3 fallback used, unusual domain sizes, long SP
- `ERROR`: Extraction failed entirely on a sequence expected to be functional

---

## 10. Implementation Phases

### Phase 1: Core Groove Extractor (new module)

**Create:** `data/groove.py`

Contents:
- `find_cys_pairs()` — shared Cys pair finder
- `classify_cys_pair()` — alpha3 CW-motif detector
- `parse_class_i()` — Class I groove extraction
- `parse_class_ii_alpha()` — Class II alpha chain groove extraction
- `parse_class_ii_beta()` — Class II beta chain groove extraction
- `extract_groove()` — unified dispatcher
- `GrooveResult` dataclass
- `prepare_mhc_input()` — top-level entry point for training + inference

**Test:** `tests/test_groove.py`
- Test each parser on reference sequences (HLA-A*02:01, HLA-DRB1*01:01, etc.)
- Test with/without signal peptide
- Test failure modes (null alleles, fragments, groove-absent Ig-only genes)
- Test cross-species (human, mouse, chicken, salmon)
- Test that prepare_mhc_input works for allele names, raw sequences, and missing inputs

**Validation:** Run `augment_mhc_index()` over entire index, verify:
- ≥97% success rate for Class I
- groove_half_1 lengths cluster at 89-91 (Class I) or 76-80 (Class II alpha)
- groove_half_2 lengths cluster at 92-93 (Class I) or 88-92 (Class II beta)
- All null/pseudogene alleles correctly flagged

### Phase 2: Augmented Index

**Modify:** `data/mhc_index.py`
- Add `augment_mhc_index()` function
- Add null/Q/Ps suffix detection
- Add `is_functional` derived column
- Update `resolve_alleles()` to return groove columns when available

**Modify:** `cli/data.py`
- Add CLI command: `presto data augment-index`

**Test:** Verify augmented CSV has correct schema, no regressions on existing resolve_alleles() tests.

### Phase 3: Data Pipeline Integration

**Modify:** `data/collate.py`
- Replace `mhc_a` / `mhc_b` with `groove_half_1` / `groove_half_2` in PrestoSample
- Update tokenization to use `max_groove_len=120` instead of `max_mhc_len=400`
- Update segment IDs: `SEG_GROOVE_1=3, SEG_GROOVE_2=4`

**Modify:** `data/loaders.py`
- Replace `_get_mhc_sequence()` → `_get_groove_halves()` using `prepare_mhc_input()`
- Replace `_resolve_mhc_b_sequence()` with groove assembly logic
- For Class I: both halves come from one `extract_groove()` call
- For Class II: half_1 from alpha chain, half_2 from beta chain
- Filter non-functional alleles from training

**Modify:** `scripts/train_iedb.py` (or `train_unified.py` after rename)
- Update sample construction to use groove halves
- Add groove extraction stats to training logs

**Test:** Verify training samples have correct groove dimensions, no None values, correct class labels.

### Phase 4: Model Architecture Updates

**Modify:** `models/presto.py`
- Replace `mhc_a_pos`, `mhc_b_pos` with `groove_1_pos`, `groove_2_pos` (max 120 instead of 400)
- Delete `groove_bias_a`, `groove_bias_b` parameters and computation
- Update segment constants and LATENT_SEGMENTS
- Simplify mhc_class handling: use explicit input, remove compositional inference (or keep as fallback)
- Update `_compute_groove_vec()` to work with new segments
- Update context_token_proj: remove B2M species since B2M is no longer a separate input

**Test:** Run model forward pass with new groove inputs, verify output shapes match. Run existing test suite (test_presto.py) with updated fixtures.

### Phase 5: Auxiliary Heads (optional, can be deferred)

**Add to** `models/presto.py`:
- `groove_domain_head` — per-residue domain classification (floor/helix/connector)
- `groove_contact_head` — per-residue contact importance

**Create:** `data/groove_labels.py`
- Generate domain labels from IMGT G-DOMAIN boundaries
- Load NetMHCpan pseudosequence positions for contact labels

**Test:** Verify auxiliary losses decrease during training, per-residue predictions are reasonable.

---

## 11. File Manifest

### New files:
| File | Purpose |
|------|---------|
| `data/groove.py` | Core groove extraction algorithm |
| `tests/test_groove.py` | Tests for groove extraction |
| `data/groove_labels.py` | Ground truth labels for auxiliary tasks (Phase 5) |

### Modified files:
| File | Changes |
|------|---------|
| `data/mhc_index.py` | Add `augment_mhc_index()`, null/Q/Ps flags, strip trailing X |
| `data/collate.py` | Replace mhc_a/mhc_b with groove_half_1/groove_half_2, reduce max_len |
| `data/loaders.py` | Use `prepare_mhc_input()`, filter non-functional alleles |
| `data/vocab.py` | Update segment constants (SEG_GROOVE_1, SEG_GROOVE_2) |
| `models/presto.py` | New pos embeddings, delete groove bias, update segments, simplify class inference |
| `data/allele_resolver.py` | Add Class II default alpha pairing logic |
| `inference/predictor.py` | Use `prepare_mhc_input()` for user-facing API |
| `cli/data.py` | Add `augment-index` command |

### Deleted code:
| What | Where | Why |
|------|-------|-----|
| `groove_bias_a`, `groove_bias_b` | models/presto.py:461-474 | Non-groove residues already removed |
| `_resolve_mhc_b_sequence()` (B2M logic) | data/loaders.py | B2M no longer a separate model input |
| Compositional class inference | models/presto.py:1658-1707 | Class always provided explicitly |

### Unchanged (but benefit from changes):
| File | Benefit |
|------|---------|
| `scripts/train_iedb.py` | Smaller MHC tokens → faster training |
| `training/checkpointing.py` | Fewer parameters to checkpoint (groove bias removed) |

---

## 12. Validation Criteria

### Phase 1 (Groove Extractor):
- [ ] `parse_class_i()` achieves ≥97% success on all Class I index entries
- [ ] Alpha1 length median = 91 ± 2; Alpha2 length median = 93 ± 2
- [ ] Cross-species: human, mouse, chicken, salmon all parse correctly
- [ ] Null alleles (N suffix) correctly identified as non-functional
- [ ] `extract_groove(HLA-A*02:01 full precursor)` == `extract_groove(HLA-A*02:01 mature)` (SP invariance)
- [ ] `prepare_mhc_input("HLA-A*02:01")` returns groove halves matching known reference

### Phase 2 (Augmented Index):
- [ ] Augmented CSV has all new columns, no schema errors
- [ ] `is_functional` count matches expected (subtract null + pseudogene + failed)
- [ ] `resolve_alleles()` returns groove columns for known alleles

### Phase 3 (Data Pipeline):
- [ ] Training samples have `groove_half_1` and `groove_half_2` with lengths in expected ranges
- [ ] No `None` or empty-string groove values for functional alleles
- [ ] Class II samples correctly combine alpha1 from alpha chain + beta1 from beta chain
- [ ] Existing test suite passes (test_collate.py, test_loaders.py)

### Phase 4 (Model):
- [ ] Model forward pass produces same-shape outputs with new inputs
- [ ] Total MHC token count reduced ~3× (from ~800 to ~240 max)
- [ ] groove_bias parameters are gone; model has fewer total parameters
- [ ] `test_presto.py` passes with updated fixtures
- [ ] Training loss converges (smoke test on synthetic data)

### Phase 5 (Auxiliary Heads):
- [ ] Domain classification loss decreases during training
- [ ] Contact importance head produces high weights at known pseudosequence positions
- [ ] Auxiliary losses don't hurt primary binding prediction

---

## Appendix A: Biological Reference

### Conserved Cys-Cys Disulfide Separations

| Domain | Cys1 mature pos | Separation | Role |
|--------|----------------|------------|------|
| MHC-I alpha2 | ~101 | ~63 | Groove domain anchor |
| MHC-I alpha3 | ~203 | ~56 | Ig-fold (not groove) |
| MHC-II beta1 | ~15 | ~64 | Groove domain |
| MHC-II beta2 | ~117 | ~56 | Ig-fold (not groove) |
| MHC-II alpha2 | ~94 | ~56 | Ig-fold (not groove) |

### Alpha3 Ig-Domain Signature Motif

The `CW[x]LGFY` motif (Trp immediately after the first Cys of the Ig-fold disulfide) is diagnostic for alpha3/Ig domains and can be used to distinguish alpha3 Cys from alpha2 Cys when both are within the search window.

### Expected Groove Domain Lengths

| Type | Half 1 | Half 2 |
|------|--------|--------|
| Class I alpha1 | 89-91 aa | — |
| Class I alpha2 | — | 92-93 aa |
| Class II alpha1 (from alpha chain) | 76-80 aa | — |
| Class II beta1 (from beta chain) | — | 88-92 aa |

### Signal Peptide Lengths

| Gene family | Typical SP length |
|-------------|-------------------|
| HLA-A/B/C | 23-24 aa |
| H2-K/D/L | 20-22 aa |
| HLA-DRA/DQA/DPA | 23-26 aa |
| HLA-DRB1/DQB1/DPB1 | 29-32 aa |
| Non-human primates | 20-24 aa |
| Fish (salmon, trout) | 10-18 aa |

---

## Appendix B: Superseded Items from data_and_scripts_audit.md

The groove extraction plan supersedes or modifies several items from the earlier audit:

| Item | Status | Reason |
|------|--------|--------|
| DQ8 (SP parsing + training augmentation) | **Superseded** | Groove extraction handles SP removal. No need for SP/mature augmentation. |
| MR3 (class-conditional groove bias) | **Superseded** | Groove bias deleted entirely. Non-groove residues removed by extraction. |
| MR1 (domain-aware positional encoding) | **Simplified** | With groove extraction, domain labels are simpler (3 classes instead of 5). |
| MR2 (pseudosequence extraction) | **Unchanged** | Still valuable as learned attention prior on groove residues. |
| MR4 (ESM-2 init) | **Unchanged** | Still deferred. |
| DQ2 (null allele filtering) | **Incorporated** | `is_null` column added during augmentation. |
| DQ3 (Q-suffix filtering) | **Incorporated** | `is_questionable` column added during augmentation. |
| DQ7 (partial fragment flag) | **Incorporated** | Handled by `groove_status` — fragments that parse get short but valid grooves; those that don't get failure status. |
