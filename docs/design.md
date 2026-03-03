# Presto Architecture Specification

---

# Table of Contents

1. [Overview](#1-overview)
2. [Inputs](#2-inputs)
3. [Tokenization and Embedding](#3-tokenization-and-embedding)
4. [Base Encoder](#4-base-encoder)
5. [MHC Chain Inference Module](#5-mhc-chain-inference-module)
6. [Core Identification Module](#6-core-identification-module)
7. [Latent Variable DAG](#7-latent-variable-dag)
8. [Multi-Allele / MIL Aggregation](#8-multi-allele--mil-aggregation)
9. [Output Heads](#9-output-heads)
10. [T-Cell Assay Output System](#10-t-cell-assay-output-system)
11. [Missing Input Handling](#11-missing-input-handling)
12. [Dimensions and Sizing](#12-dimensions-and-sizing)

---

# 1. Overview

Presto predicts antigen processing, MHC binding, surface presentation,
T-cell recognition, and immunogenicity from sequence inputs. It handles
MHC class I and class II in a unified architecture, reserves an optional
future TCR pathway (currently disabled in canonical training/inference),
and aggregates across a patient's full allele complement via
multiple-instance learning (MIL).

## Document Structure

| Document | Scope |
|----------|-------|
| **This document** (`design.md`) | Model architecture: inputs, tokenization, encoder, latent DAG, output heads, MIL, T-cell assay system |
| `training_spec.md` | Training strategy: data sources, losses, mini-batch construction, synthetic negatives, priors, scheduling |
| `tcr_spec.md` | Future TCR pathway design (not active in canonical training/inference) |
| `cli.md` | CLI usage |
| `../TODO.md` | Implementation status audit |

## Design Principles

- **Segment-blocked base attention with latent cross-attention.**
  Base encoder tokens attend only within their own segment. Cross-segment
  integration is deferred to the latent query DAG, which enforces
  biological causal structure via access masks and dependency ordering.
- **Latent DAG encodes biological priors as architectural constraints.**
  Processing does not see MHC. Binding does not see flanks.
  Recognition sees peptide only. Presentation has no token access
  (pure bottleneck from processing + binding vectors). These are
  structural constraints, not learned behaviors.
- **Graceful degradation with missing inputs.** Every optional input
  has a defined fallback strategy. The model produces calibrated
  predictions regardless of which inputs are available.
- **Multi-allele MIL is a first-class operation**, not a post-hoc
  wrapper. Allele competition and attribution are learned within the model.
- **Compositional T-cell assay modeling.** T-cell outputs are generated
  for all attested assay configurations in parallel, with shared
  compositional context embeddings that capture method/readout/culture
  biases as learnable vectors.

## Naming Convention

- `*_vec`: vector representation.
- `*_logit`: logit-space scalar.
- `*_logits`: logit-space vector (multi-class).
- `*_prob`: probability-space scalar.
- `*_probs`: probability-space vector (multi-class).

Canonical vector keys:

| Key | What it represents | How computed | Defined in |
|-----|-------------------|--------------|------------|
| `mhc_a_vec` | MHC alpha chain representation | Masked mean of alpha-chain token hidden states from base encoder | S4.3 |
| `mhc_b_vec` | MHC beta chain representation | Masked mean of beta-chain token hidden states from base encoder | S4.3 |
| `core_context_vec` | Soft-weighted summary of peptide hidden states over the predicted binding core | Attention-pooled using core-pointer probabilities over peptide tokens | S6.3 |
| `tcr_vec` | TCR representation (future) | Planned TCR encoder [TCR_CLS] output (currently not active) | `tcr_spec.md` |
| `pmhc_vec` | Overall pMHC complex embedding for retrieval / similarity | Learned projection from presentation latent vectors | S9.7 |
| `latent_vecs` | Dict mapping latent name to its computed vector | Each produced by the latent DAG cross-attention mechanism | S7.3 |

There is no pooled `peptide_vec` or combined `mhc_vec`. Peptide tokens are
consumed as token sequences by latent cross-attention — each latent attends
to exactly the peptide tokens it needs. The two MHC chains serve different
biological roles (alpha carries the binding groove; beta is beta2m for class I
or the DRB/DQB/DPB chain for class II) and are kept separate for per-chain
type/species inference (S5).

---

# 2. Inputs

## 2.1 API-Level Inputs

These are the inputs to the Presto code interface. The code translates them
into neural network inputs (token sequences and conditioning embeddings).

| Input | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `peptide` | AA string | **Yes** | -- | Presented peptide sequence (8-15 residues) |
| `nflank` | AA string | No | Empty (sentinel token) | Source protein residues upstream of peptide N-terminus. Up to 20 residues. |
| `cflank` | AA string | No | Empty (sentinel token) | Source protein residues downstream of peptide C-terminus. Up to 20 residues. |
| `mhc_a` | AA string or resolvable allele name | No | Sentinel token | MHC alpha chain amino acid sequence. If an allele label is provided (e.g., "HLA-A*02:01"), it must resolve to sequence before tokenization. Unresolved labels are treated as errors in canonical training. |
| `mhc_b` | AA string or resolvable allele name | No | Canonical beta2m for species (class I) or sentinel (class II) | MHC beta chain amino acid sequence. Class I defaults to species beta2m; class II expects DRB/DQB/DPB-like sequence when available. |
| `mhc_class` | Enum: {I, II} | No | inferred from MHC chains | Optional hard override for class-specific downstream heads. |
| `species` | Enum: {human, nhp, murine, cattle, other_mammal, bird, fish} | No | `human` | See "Conditioning metadata" below. |
| `tcr_alpha` | AA string | No | None | TCR alpha chain (reserved for future TCR pathway). |
| `tcr_beta` | AA string | No | None | TCR beta chain (reserved for future TCR pathway). |

## 2.2 How API Inputs Map to Neural Network Inputs

The API inputs are translated into two categories of neural network input:

1. **Token sequences** — `peptide`, `nflank`, `cflank`, `mhc_a`, `mhc_b`
   become amino acid token sequences organized into segments (S3), processed
   by the base encoder. `tcr_alpha` and `tcr_beta` are tokenized separately
   for the TCR encoder (`tcr_spec.md`).

2. **Conditioning metadata** — `species` and chain completeness flags are
   NOT token sequences. They are encoded as the
   **global conditioning embedding** (S3.2.4), a single learned vector
   that is summed into every token representation. This provides the model
   with metadata about the input context:
   - `species` tells the model which organism's proteasomal and MHC
     biology to apply. Defaults to `human`.

   MHC class is inferred from MHC chain sequences (S5) as internal
   `pI/pII`, then used only in class-specific downstream mixing/gating.
   User `mhc_class` input is an optional hard override for that mixing path,
   not a token-level conditioning embedding.

## 2.3 Multi-Allele Mode

For patient-level prediction, provide a list of alleles instead of a single
MHC:

| Input | Type | Description |
|-------|------|-------------|
| `alleles` | List of {mhc_a, mhc_b, mhc_class, locus} | Up to 6 class I + 10 class II molecules (see S8 for DQ/DP combinatorics) |

**How this works internally:** The code runs the neural network once per
allele in the list, sharing the peptide+flank base encoding across alleles
(S4.4 — segment-blocked attention makes this free). Each allele produces
its own set of latent vectors and logits. The per-allele outputs are then
aggregated via multi-instance learning (S8) to produce patient-level
predictions.

## 2.4 Input Constraints

- **Peptide length**: 8-15 residues.
- **Flank length**: 0-20 residues each side. Longer flanks truncated to 20 residues adjacent to cleavage site.
- **MHC sequences**: Full-length amino acid sequences. Allele names are resolved to full sequences via the allele resolver (`data/allele_resolver.py`). Maximum chain length ~400 residues.
- **No allele-token fallback**: Raw allele strings are not valid sequence inputs. If an allele cannot be resolved to amino-acid sequence, canonical training fails fast.
- **TCR**: Reserved for a future pathway. Canonical production training/inference currently ignores TCR inputs. See `tcr_spec.md`.

---

# 3. Tokenization and Embedding

## 3.1 Token Sequence Layout

```
[NFLANK] nf_1 nf_2 ... nf_n [CLEAVE_N]
p_1 p_2 ... p_L
[CLEAVE_C] cf_1 cf_2 ... cf_m [CFLANK]
[MHC_A] a_1 a_2 ... a_j
[MHC_B] b_1 b_2 ... b_k
```

Special tokens:
- `[NFLANK]`: N-flank segment start marker
- `[CLEAVE_N]`: N-terminal cleavage boundary (between N-flank and peptide)
- `[CLEAVE_C]`: C-terminal cleavage boundary (between peptide and C-flank)
- `[CFLANK]`: C-flank segment end marker
- `[MHC_A]`: MHC alpha chain segment start marker
- `[MHC_B]`: MHC beta chain segment start marker
- `[PAD]`: Padding (masked from attention)
- `<MISSING>`: dedicated missing-value token for absent optional sequence segments/chains

Each segment is delimited by its own special tokens. There are no `[CLS]`
or `[SEP]` tokens — with segment-blocked attention (S4.2), tokens only
attend within their own segment, so global pooling tokens like `[CLS]`
would have no useful information to aggregate. Segment boundaries are
already defined by the segment ID assignments (S3.2.2).

TCR tokens are NOT in this sequence — they are encoded separately (see `tcr_spec.md`).

## 3.2 Per-Token Embedding

Every token receives a sum of four embedding components:

```python
token_repr[i] = aa_embed[i] + segment_embed[i] + position_embed[i] + global_cond_embed
```

### 3.2.1 Amino Acid Embedding (`aa_embed`)

- Shared lookup table across all chains.
- Dimension: `d_model`.
- Vocabulary: 20 standard amino acids + X (unknown) + U (selenocysteine) + all special tokens, including a dedicated `<MISSING>` token.
- Canonical tokenization is strict: unfamiliar sequence characters raise an error instead of silently mapping to `<UNK>`.
- `X` is allowed as explicit ambiguous amino acid and uses a fixed zero embedding vector.
- `<UNK>` remains only as a compatibility token for non-canonical/legacy parsing paths.
- **Initialization**: From ESM-2 token embeddings (project from ESM dim to `d_model` if needed). Trainable.

### 3.2.2 Segment Embedding (`segment_embed`)

Learned embedding per segment type, dimension `d_model`.

| Segment ID | Value | Tokens covered |
|------------|-------|---------------|
| `SEG_NFLANK` | 0 | `[NFLANK]`, N-flank residues, `[CLEAVE_N]` |
| `SEG_PEPTIDE` | 1 | Peptide residues p_1 ... p_L |
| `SEG_CFLANK` | 2 | `[CLEAVE_C]`, C-flank residues, `[CFLANK]` |
| `SEG_MHC_A` | 3 | `[MHC_A]`, MHC alpha chain residues |
| `SEG_MHC_B` | 4 | `[MHC_B]`, MHC beta chain residues |

### 3.2.3 Positional Encoding (`position_embed`) -- Segment-Specific

Each segment type uses a biologically motivated positional encoding scheme.

#### Peptide Residues: Triple-Frame Encoding

For residue at index `i` in a peptide of length `L`:

```python
peptide_pos[i] = (
    learned_nterm_embed[i]           +   # distance from N-terminus (table: 50 entries)
    learned_cterm_embed[L - 1 - i]   +   # distance from C-terminus (table: 50 entries)
    MLP_frac(i / (L - 1))               # fractional position in [0, 1] -> d_model
)
```

**Rationale**: "P2 anchor" means the same thing regardless of peptide length.
N-terminal and C-terminal tables encode this directly. The fractional MLP
captures the MHCflurry "center padding" intuition -- where a residue sits
proportionally. The model sees all three reference frames simultaneously.

In deeper layers, after core identification (S6), peptide residues receive
an **additional core-relative positional encoding** summed with the base
encoding. See S6 for specification.

#### N-Flank Residues: Distance-from-Cleavage Encoding

```python
nflank_pos[j] = learned_nflank_dist_embed[j]   # table: 25 entries
# j = distance from cleavage site, counting backward
```

**Rationale**: Proteasomal/endosomal cleavage preferences are defined relative
to the cleavage site. Position 1 (immediately upstream) has the strongest
signal.

#### C-Flank Residues: Distance-from-Cleavage Encoding

```python
cflank_pos[j] = learned_cflank_dist_embed[j]   # table: 25 entries (separate from N-flank)
# j = distance from cleavage site, counting forward
```

**Rationale**: Class I processing is primarily C-terminal (proteasome).
Class II uses both termini (cathepsins). Separate N/C tables let the model
learn this distinction.

#### MHC Alpha/Beta Chain: Sequential Positional Encoding

```python
mhc_a_pos[i] = learned_mhc_a_pos_embed[i]   # table: max_mhc_a_len entries
mhc_b_pos[i] = learned_mhc_b_pos_embed[i]   # table: max_mhc_b_len entries (separate)
```

Positions are sequential indices into the full-length MHC chain sequence.
Separate learned tables for alpha and beta chains (they have different
lengths and structural roles).

### 3.2.4 Global Conditioning Embedding (`global_cond_embed`)

Broadcast to ALL tokens. Encodes metadata about the input:

```python
global_cond = (
    species_embed[species_id]                 +   # 7 species categories
    chain_completeness_embed[completeness_bits]    # bitfield (see below)
)
```

**Chain completeness bitfield:**

| Bit | Flag | Meaning |
|-----|------|---------|
| 0 | `has_nflank` | N-terminal flanking region provided |
| 1 | `has_cflank` | C-terminal flanking region provided |
| 2 | `has_mhc_a` | MHC alpha chain explicitly provided |
| 3 | `has_mhc_b_explicit` | MHC beta chain explicitly provided (vs. defaulted to canonical beta2m) |
| 4 | `has_tcr` | TCR provided (any chain) |
| 5 | `has_tcr_paired` | Both TCR alpha and beta provided |

Total table size: 2^6 = 64 entries. Each is a learned `d_model` vector.

---

# 4. Base Encoder

## 4.1 Architecture

Standard pre-norm transformer encoder.

- **Layers**: `N_base` = 6 (recommended; 4-8 range)
- **Attention heads**: `N_heads` = 8
- **FFN hidden dimension**: `4 * d_model`
- **Activation**: GELU
- **Normalization**: Pre-LayerNorm
- **Dropout**: 0.1

## 4.2 Attention: Segment-Blocked Self-Attention

**Critical design choice**: tokens attend only within their own segment
in the base encoder. Cross-segment integration is deferred entirely to the
latent cross-attention DAG (S7).

```python
# Attention mask: True = allowed to attend
base_attn_mask[i, j] = (seg_id[i] == seg_id[j]) and not is_pad[j]
```

This means:
- Peptide residues attend to other peptide residues (learning internal peptide features)
- MHC alpha residues attend to other MHC alpha residues (learning chain-internal features)
- N-flank residues attend to other N-flank residues
- **No** peptide <-> MHC cross-attention at base level
- **No** flank <-> peptide cross-attention at base level

**Rationale**: Segment-blocked base encoding guarantees that per-segment
representations are "clean" -- a peptide token's representation reflects
peptide-internal context only, not MHC groove context. This is important
because:

1. **Processing latents** must not see MHC. If the base encoder allows
   peptide <-> MHC attention, the peptide representations consumed by
   processing would already contain MHC information, violating the
   biological prior.
2. **Different latents need different cross-segment views.** Binding needs
   peptide+MHC but not flanks. Processing needs peptide+flanks but not MHC.
   Clean per-segment representations let each latent compose exactly the
   cross-segment view it needs.
3. The latent cross-attention layers (2 per latent) provide sufficient
   capacity for cross-segment integration where needed.

## 4.3 Outputs

After `N_base` layers:
```
H in R^{n_tokens x d_model}
```

Per-segment pooled vectors are extracted for downstream modules:
```python
mhc_a_vec = masked_mean(H[mhc_a_tokens])   # (d_model,)
mhc_b_vec = masked_mean(H[mhc_b_tokens])   # (d_model,)
peptide_H = H[peptide_tokens]               # (L_pep, d_model)
```

## 4.4 Computational Sharing for Multi-Allele Mode

For MIL over multiple alleles, the base encoder naturally supports sharing
because segment-blocked attention means peptide/flank encoding is independent
of MHC. The peptide/flank tokens are encoded ONCE and reused across all alleles.

```python
# Phase A: encode peptide + flanks (once, shared)
pep_flank_H = base_encoder(pep_flank_tokens)  # all N_base layers

# Phase B: encode each allele's MHC tokens (per-allele, also N_base layers)
for allele in patient_alleles:
    mhc_H = base_encoder(mhc_tokens[allele])  # separate forward pass
```

No need for a split-phase encoder -- segment blocking makes this automatic.

---

# 5. MHC Chain Inference Module

Operates on pooled MHC segment vectors from the base encoder. Infers
per-chain properties and produces a context vector used by all downstream
latents.

## 5.1 Per-Chain Type Inference

Each MHC chain is independently classified:

```python
# Per-chain type probabilities (alpha chain)
mhc_a_type_logits = Linear_a_type(mhc_a_vec)  # (n_chain_types,)
mhc_a_type_probs = softmax(mhc_a_type_logits)
# Categories: {class_I_alpha, class_II_alpha, unknown}

# Per-chain type probabilities (beta chain)
mhc_b_type_logits = Linear_b_type(mhc_b_vec)  # (n_chain_types,)
mhc_b_type_probs = softmax(mhc_b_type_logits)
# Categories: {class_I_beta, class_II_beta, unknown}
```

**Compositional class probabilities** derived from per-chain types:

```python
# P(class I) = P(a is class_I_alpha) * P(b is class_I_beta)
# P(class II) = P(a is class_II_alpha) * P(b is class_II_beta)
class1_prob = mhc_a_type_probs[CLASS_I_ALPHA] * mhc_b_type_probs[CLASS_I_BETA]
class2_prob = mhc_a_type_probs[CLASS_II_ALPHA] * mhc_b_type_probs[CLASS_II_BETA]
class_probs = normalize([class1_prob, class2_prob])  # (2,)
```

User-provided `mhc_class` can override inferred probabilities (set to
one-hot if class I or II is specified explicitly).

## 5.2 Per-Chain Species Inference

```python
mhc_a_species_logits = Linear_a_species(mhc_a_vec)  # (n_species,)
mhc_a_species_probs = softmax(mhc_a_species_logits)

mhc_b_species_logits = Linear_b_species(mhc_b_vec)  # (n_species,)
mhc_b_species_probs = softmax(mhc_b_species_logits)
# Categories: {human, murine, nhp, other}
```

## 5.3 Chain Compatibility Score

Lightweight head that scores whether the two chains form a functional heterodimer:

```python
compat_logit = Linear_compat(concat(mhc_a_vec, mhc_b_vec,
                                     mhc_a_type_probs, mhc_b_type_probs,
                                     mhc_a_species_probs, mhc_b_species_probs))
chain_compat_prob = sigmoid(compat_logit)
```

Used as an auxiliary training target (supervised from known valid/invalid
pairings) and as a soft gate on MIL instance weights for DQ/DP trans dimers.

## 5.4 Context Vector

A context vector token assembled from class/species probabilities, appended
to the latent queries that explicitly receive it (processing, binding,
presentation; not recognition/immunogenicity):

```python
context_vec = MLP_context(concat(class_probs,
                                  mhc_a_species_probs, mhc_b_species_probs,
                                  chain_compat_prob))  # -> (d_model,)
```

**What information does context_vec carry that isn't in the MHC token sequences?**
The raw MHC tokens carry amino acid identity at each position, but the latent
queries that see MHC tokens (binding_affinity, binding_stability) would need
to independently re-derive "is this class I or class II?" and "what species
is this?" from those tokens. The context_vec provides these higher-level
summaries pre-computed, saving the latent cross-attention from spending
capacity on a classification task it doesn't need to solve.

More importantly, `context_vec` is the **only channel of MHC-class information
for latents that cannot see MHC tokens**. Processing latents see peptide+flanks
but not MHC — yet they need to know whether to apply proteasomal (class I)
or endosomal (class II) processing patterns. The context_vec provides this.

## 5.5 Source Protein Species Inference

From the processing latent vectors (after Level 1 computation):

```python
# Use class-prob-weighted combination of processing latent vectors
processing_mixed_vec = (class_probs[0] * processing_class1_vec
                      + class_probs[1] * processing_class2_vec)
source_species_logits = Linear_source_species(processing_mixed_vec)  # (n_species,)
source_species_probs = softmax(source_species_logits)
# Categories: {human, murine, nhp, other}
```

**Rationale**: The source protein species can differ from the MHC species
(e.g., viral peptide in human MHC). Inferred from flank+peptide context
via the processing latent. Auxiliary training target where labels exist.

---

# 6. Core Identification Module

Operates on base encoder outputs. Identifies the MHC-II binding core within
the peptide. For class I, learns that core = full peptide.

## 6.1 Core Pointer Head

```python
peptide_H = H[peptide_start : peptide_end]            # (L_pep, d_model)
mhc_pool = mean_pool(mhc_a_vec, mhc_b_vec)            # (d_model,)

# Core start position logits
start_logits = Linear_start(
    concat(peptide_H, mhc_pool.expand(L_pep, -1))
)                                                       # (L_pep, 1) -> (L_pep,)

# Mask invalid positions
min_core_width = 7
start_logits[L_pep - min_core_width + 1 :] = -inf

core_start_probs = softmax(start_logits)

# Differentiable core width
core_width_logit = Linear_width(mhc_pool)               # scalar
core_width = 7.0 + 5.0 * sigmoid(core_width_logit)      # in [7.0, 12.0]
# Initialize bias so sigmoid ~ 0.4 -> core_width ~ 9.0

# Expected start
positions = arange(len(core_start_probs)).float()
expected_start = (core_start_probs * positions).sum()
```

## 6.2 Soft Core Membership

Per-residue probability of being inside the binding core:

```python
def compute_soft_membership(core_start_probs, core_width, L_pep):
    """Vectorized soft rectangular window over all start positions."""
    membership = zeros(L_pep)
    for s in range(len(core_start_probs)):
        for i in range(L_pep):
            dist_from_start = i - s
            dist_from_end = (s + core_width - 1) - i
            inclusion = sigmoid(5 * dist_from_start) * sigmoid(5 * dist_from_end)
            membership[i] += core_start_probs[s] * inclusion
    return clamp(membership, 0, 1)
```

(In practice: vectorized, not loops.)

## 6.3 Core Context Vector (`core_context_vec`)

Soft-attention-weighted average of peptide hidden states, where weights
come from the MHC-informed core-start predictor:

```python
# Attention weights from core membership (soft window over peptide)
core_weights = compute_soft_membership(core_start_probs, core_width, L_pep)
core_weights = core_weights / core_weights.sum().clamp(min=1e-8)

# Weighted average of peptide representations
core_context_vec = (core_weights.unsqueeze(-1) * peptide_H).sum(dim=0)  # (d_model,)
```

This vector captures "what the core region of the peptide looks like"
as a differentiable summary. It is consumed by:
- `presentation_class2` latent (PFR/core context for DM editing)
- `recognition_cd4` latent (core vs PFR distinction matters for CD4 TCR contact)

## 6.4 Core-Relative Positional Encoding

After core identification, peptide residues receive additional positional
encoding injected into binding and recognition latent queries:

```python
for each peptide residue i:
    rel_to_core_start = i - expected_start           # fractional
    rel_to_core_end = i - (expected_start + core_width - 1)

    # Core position encoding (interpolated between integer embeddings)
    core_pos_floor = floor(rel_to_core_start).int()
    core_pos_frac = rel_to_core_start - core_pos_floor
    core_pos_embed = (
        (1 - core_pos_frac) * learned_core_pos[clamp(core_pos_floor, 0, 14)] +
        core_pos_frac * learned_core_pos[clamp(core_pos_floor + 1, 0, 14)]
    )
    # Table size 15: accommodates cores up to 12 residues

    # PFR distance encoding
    if rel_to_core_start < 0:
        pfr_embed = learned_npfr_dist[clamp(round(-rel_to_core_start), 0, 19)]
    else:
        pfr_embed = learned_cpfr_dist[clamp(round(rel_to_core_end), 0, 19)]
    # Tables size 20 each

    # PFR length embedding
    npfr_len_embed = learned_npfr_len[clamp(round(expected_start), 0, 20)]
    cpfr_len_embed = learned_cpfr_len[clamp(round(L_pep - expected_start - core_width), 0, 20)]

    # Interpolate using soft core membership
    core_rel_pos[i] = (
        core_membership[i] * core_pos_embed +
        (1 - core_membership[i]) * pfr_embed +
        npfr_len_embed + cpfr_len_embed
    )
```

Core-relative encoding is summed with the base triple-frame encoding in
latents that use it (binding_affinity, binding_stability, recognition_cd4).

## 6.5 Supervision

- **Class II**: Supervise `core_start_probs` with known core assignments
  (NetMHCIIpan, IEDB annotated cores, crystal structures). Loss: CE on
  start position.
- **Class I**: Supervise with core = full peptide (start = 0, width = L).
- **Core width**: Optionally supervise with known core length = 9 for
  class II, or leave unsupervised.

---

# 7. Latent Variable DAG

## 7.1 Latent Definitions

11 latent query tokens, each a learned embedding of dimension `d_model`.

| # | Name | Biological Meaning |
|---|------|--------------------|
| 1 | `processing_class1` | Proteasomal cleavage + TAP transport + ERAP trimming |
| 2 | `processing_class2` | Endosomal/lysosomal cathepsin processing |
| 3 | `binding_affinity` | pMHC binding affinity (KD, IC50) -- class-symmetric |
| 4 | `binding_stability` | pMHC complex half-life (koff, t1/2) -- class-symmetric |
| 5 | `presentation_class1` | Surface presentation on MHC-I |
| 6 | `presentation_class2` | Surface presentation on MHC-II |
| 7 | `recognition_cd8` | Intrinsic recognizability by CD8+ TCR repertoire |
| 8 | `recognition_cd4` | Intrinsic recognizability by CD4+ TCR repertoire |
| 9 | `immunogenicity_cd8` | Net CD8+ T-cell response likelihood |
| 10 | `immunogenicity_cd4` | Net CD4+ T-cell response likelihood |
| 11 | `ms_detectability` | Peptide-intrinsic MS detectability ("flyability") |

### Design Rationale

1. **Binding is class-symmetric**: One `binding_affinity` and one
   `binding_stability` latent serve both class I and class II, with
   class-conditioned readout heads. Shared cross-attention, two projection
   paths. The fundamental thermodynamics are the same; class differences
   are in groove topology and anchor positions, handled by conditioned heads.
2. **Presentation has NO token access**: Pure bottleneck from upstream
   latent vectors only. This forces all information through the
   processing/binding bottlenecks, preventing shortcut learning that
   bypasses the causal biology.
3. **Recognition sees peptide only**: No MHC tokens, no upstream latent
   dependencies. Recognition captures peptide-intrinsic features
   (foreignness, unusual residues at solvent-exposed positions).
4. **Immunogenicity depends on {affinity, stability, recognition}**:
   Drops presentation from dependencies. Presentation is already a
   function of processing + binding, so including it would create
   redundant paths. T-cell assay outputs (S10) handle the full biology
   via soft gates.
5. **MS detectability is a dedicated latent**: Peptide-only latent capturing
   ionization efficiency, charge state propensity, hydrophobicity --
   properties that affect MS detection independent of biological
   presentation.

## 7.2 Computational DAG Structure

```
                    Input Tokens (from base encoder H)
                    +---------------+----------------+----------------+
                    | Peptide +     | Peptide +      | Peptide        |
                    |   Flanks      |   MHC          |   only         |
                    +-------+-------+--------+-------+--------+-------+
                            |                |                |
                    +-------+-------+ +------+------+  +------+------+
    Level 0         | processing_   | | processing_ |  | ms_          |
   (parallel,       |   class1      | |   class2    |  | detectability|
    no deps)        | pep+flanks    | | pep+flanks  |  | pep only     |
                    +-------+-------+ +------+------+  +-------------+
                            |                |
                            |       +--------+--------+
    Level 1                 |       |                  |
   (parallel,               |  binding_affinity   binding_stability
    no deps)                |  pep+mhc_a+mhc_b   pep+mhc_a+mhc_b
                            |       |                  |
                    +-------+-------+------------------+-------+
                    |                                          |
    Level 2         | presentation_class1    presentation_class2
   (parallel)       | f(processing_class1,   f(processing_class2,
                    |   binding_affinity,       binding_affinity,
                    |   binding_stability)      binding_stability)
                    | NO token access        + core_context_vec
                    |                                          |
                    +-------+                          +-------+
                            |                          |
    Level 2.5       +-------+--------+       +---------+------+
   (parallel,       | recognition_   |       | recognition_   |
    no latent deps) |   cd8          |       |   cd4          |
                    | peptide only   |       | peptide only   |
                    | (+-TCR)        |       | + core_ctx_vec |
                    +-------+--------+       | (+-TCR)        |
                            |                +--------+-------+
                            |                         |
    Level 3         +-------+--------+       +--------+-------+
                    | immunogenicity_|       | immunogenicity_|
                    |   cd8          |       |   cd4          |
                    | f(binding_     |       | f(binding_     |
                    |   affinity,    |       |   affinity,    |
                    |   binding_     |       |   binding_     |
                    |   stability,   |       |   stability,   |
                    |   recognition_ |       |   recognition_ |
                    |   cd8)         |       |   cd4)         |
                    | MLP, no tokens |       | MLP, no tokens |
                    +----------------+       +----------------+
```

**Note**: Recognition latents (recognition_cd8/recognition_cd4) are at
Level 2.5 because they have no upstream latent dependencies -- they are
computed from tokens only. They could be computed in parallel with
Level 0/1/2 latents, but are shown here to indicate their logical
position in the biological cascade.

## 7.3 Latent Computation Mechanism

Each latent is computed by `N_latent` = 2 layers of cross-attention:

```python
def compute_latent(query_token, key_value_tokens, n_layers=2):
    """
    query_token: (1, d_model) -- the learnable latent embedding
    key_value_tokens: (n_kv, d_model) -- allowed inputs for this latent
    Returns: (d_model,) -- the computed latent vector
    """
    x = query_token
    for layer in range(n_layers):
        # Cross-attention
        residual = x
        x = layer_norm(x)
        x = multi_head_cross_attention(query=x, key=key_value_tokens,
                                        value=key_value_tokens)
        x = residual + x
        # FFN
        residual = x
        x = layer_norm(x)
        x = ffn(x)
        x = residual + x
    return x.squeeze(0)  # (d_model,)
```

## 7.4 Per-Latent Specifications

### Level 0: Processing and MS Detectability (parallel, no upstream deps)

---

#### `processing_class1` -- Class I Processing

**Cross-attends to:**

| Source | Tokens | Positional Encoding |
|--------|--------|---------------------|
| Peptide | p_1 ... p_L | Triple-frame (N-term, C-term, fractional) |
| N-flank | nf_1 ... nf_n | Distance-from-cleavage (N) |
| C-flank | cf_1 ... cf_m | Distance-from-cleavage (C) |
| Delimiters | `[CLEAVE_N]`, `[CLEAVE_C]` | Fixed learned |
| Context | context_vec (from S5.4) | -- |

**Does NOT attend to:** MHC alpha, MHC beta, any latent tokens.

**Biological rationale:** Proteasomal cleavage and TAP transport depend on
peptide and flanking sequences, not on which MHC allele will present the
peptide. C-terminal cleavage is primary for class I (proteasome); N-terminal
trimming by ERAP1/2 is secondary.

The context_vec provides soft class/species information, allowing the
processing pathway to learn species-specific proteasomal preferences
(e.g., immunoproteasome subunit composition differs between species)
without directly seeing MHC tokens.

---

#### `processing_class2` -- Class II Processing

**Cross-attends to:** Same as processing_class1 (peptide + flanks + context_vec).

**Does NOT attend to:** MHC alpha, MHC beta.

**Biological rationale:** Endosomal processing by cathepsins S, L depends
on local sequence context. DM editing is MHC-dependent but modeled in
the presentation latent. processing_class2 uses the same input tokens as
processing_class1 but has its **own query vector and attention parameters**,
learning different cleavage patterns (endosomal cathepsins vs proteasome).

---

#### `ms_detectability` -- MS Detectability

**Cross-attends to:**

| Source | Tokens | Positional Encoding |
|--------|--------|---------------------|
| Peptide | p_1 ... p_L | Triple-frame |

**Does NOT attend to:** MHC, flanks, any latent tokens.

**Biological rationale:** MS detection is affected by peptide-intrinsic
physicochemical properties independent of biological presentation:
- Ionization efficiency (charge state, proton affinity)
- Chromatographic behavior (hydrophobicity, retention time)
- Fragmentation patterns (proline effects, charge localization)
- Peptide length bias (shorter peptides may fragment poorly; longer
  peptides have more charge states)

These biases are systematic across different MS platforms (Orbitrap vs TOF,
DDA vs DIA) and should be modeled separately from biological presentation.

**Readout heads on ms_detectability:**
```python
ms_detectability_logit = Linear_ms_detect(ms_detectability_vec)   # base detectability

# Platform-specific bias terms (optional, if platform metadata available):
orbitrap_bias = Linear_orbitrap(ms_detectability_vec)
tof_bias = Linear_tof(ms_detectability_vec)
```

---

### Level 1: Binding (parallel, no upstream deps)

---

#### `binding_affinity` -- Binding Affinity (Class-Symmetric)

**Cross-attends to:**

| Source | Tokens | Positional Encoding |
|--------|--------|---------------------|
| Peptide | p_1 ... p_L | Triple-frame + core-relative (S6.4) |
| MHC alpha | a_1 ... a_j | Sequential |
| MHC beta | b_1 ... b_k | Sequential |
| Context | context_vec | -- |

**Does NOT attend to:** Flanks.

**Biological rationale:** Binding affinity is determined by peptide-groove
complementarity. For class I, the entire peptide is in the groove. For
class II, core residues dominate (P1, P4, P6, P9 anchors) but PFRs
contribute modestly at P-1 and P+1. The core-relative positional encoding
provides this inductive bias differentiably.

**Class-symmetric design:** One set of cross-attention parameters serves
both class I and class II. The context_vec carries class probability
information, and the class-conditioned readout heads (S9) handle
class-specific calibration:

```python
binding_affinity_vec = compute_latent(binding_affinity_query, kv=pep_mhc_kv_with_core)

# Class-specific readout (in output heads):
kd_class1 = KD_head_class1(binding_affinity_vec)   # calibrated for class I scale
kd_class2 = KD_head_class2(binding_affinity_vec)   # calibrated for class II scale
kd_mixed  = class_probs[0] * kd_class1 + class_probs[1] * kd_class2
```

**Rationale for shared binding**: Class I and II share the same
fundamental thermodynamics (peptide side chains in MHC pockets, hydrogen
bond networks). The difference is groove topology (closed vs open ends)
and anchor positions. A shared latent learns the common binding physics;
class-conditioned heads learn the calibration differences.

---

#### `binding_stability` -- Binding Stability (Class-Symmetric)

**Cross-attends to:** Same as binding_affinity.

**Does NOT attend to:** Flanks.

**Biological rationale:** Stability (koff, half-life) and affinity (KD, IC50)
are correlated but distinct. A peptide can bind tightly with fast off-rate,
or bind weakly but persist. Separate latents allow the model to learn this.

**Orthogonality regularization:**
```python
L_ortho = |cosine_similarity(binding_affinity_vec, binding_stability_vec)|
```

---

### Level 2: Presentation (computed after Level 0 + Level 1)

---

#### `presentation_class1` -- Class I Presentation

**Cross-attends to (LATENT VECTORS ONLY):**

| Source | Type | Purpose |
|--------|------|---------|
| `processing_class1_vec` | (1, d_model) | Processing likelihood |
| `binding_affinity_vec` | (1, d_model) | Binding affinity |
| `binding_stability_vec` | (1, d_model) | Complex stability |
| Context | context_vec | Class/species info |

**Does NOT attend to:** Peptide tokens, MHC tokens, or any raw sequence.

**Biological rationale:** Presentation is a BOTTLENECK function of upstream
biology. A peptide is presented on MHC-I if and only if:
1. It is generated by proteasomal cleavage (processing_class1)
2. It binds the MHC groove with sufficient affinity (binding_affinity)
3. The complex is stable enough to reach the surface (binding_stability)

By restricting presentation_class1 to see only upstream latent vectors
(no raw tokens), we force the model to route ALL peptide/MHC information
through the processing and binding bottlenecks. This prevents the
presentation latent from learning shortcuts that bypass the causal biology.
If there are features important for presentation not captured by
processing/binding, the correct fix is to improve those upstream latents,
not to give the presentation latent a bypass path.

---

#### `presentation_class2` -- Class II Presentation

**Cross-attends to (LATENT VECTORS ONLY):**

| Source | Type | Purpose |
|--------|------|---------|
| `processing_class2_vec` | (1, d_model) | Processing likelihood |
| `binding_affinity_vec` | (1, d_model) | Binding affinity |
| `binding_stability_vec` | (1, d_model) | Complex stability |
| `core_context_vec` | (1, d_model) | Core/PFR summary for DM editing |
| Context | context_vec | Class/species info |

**Does NOT attend to:** Peptide tokens, MHC tokens.

**Additional input vs presentation_class1:** `core_context_vec` is included because
HLA-DM editing preferentially removes kinetically unstable pMHC-II
complexes, and this process depends on the peptide's core register. The
core_context_vec provides a differentiable summary of which peptide positions
are in the core vs PFR, which is relevant to DM susceptibility. Class I
presentation (presentation_class1) does not need this because the core IS
the full peptide for class I.

---

### Level 2.5: Recognition (parallel, no upstream latent deps)

---

#### `recognition_cd8` -- CD8 Recognition

**Cross-attends to:**

| Source | Tokens | Positional Encoding |
|--------|--------|---------------------|
| Peptide | p_1 ... p_L | Triple-frame |

**Does NOT attend to:** MHC alpha, MHC beta, flanks, any upstream latent.

**Planned future extension (currently disabled):** gated cross-attention
from recognition to TCR tokens (see `tcr_spec.md`):

| Source | Tokens | Encoding |
|--------|--------|----------|
| TCR token representations | (n_tcr, d_model) | CDR-annotated (see `tcr_spec.md`) |

**Biological rationale:** Recognition = "how likely is SOME TCR in a
typical repertoire to recognize this peptide?" This depends on:
- **Foreignness**: Peptides similar to self-peptidome have low recognition
  (thymic tolerance deletes reactive T cells).
- **Physicochemical properties**: Unusual residues at solvent-exposed
  positions (for class I: P4, P5, P6, P7, P8 in a 9mer) increase
  recognition probability.
- **Peptide-intrinsic features**: Aromatic/charged residues at central
  positions are more immunogenic (Calis et al., PLoS Comp Bio 2013).

**Why no MHC tokens for recognition?** The TCR contacts primarily the
peptide residues at solvent-exposed positions plus the MHC alpha-helices
flanking the groove. In this canonical design we intentionally keep the
recognition branch peptide-only and do NOT explicitly model allele-specific
solvent accessibility yet. Allele-specific effects are routed through the
binding/presentation pathway and can be added later as a dedicated extension.

Restricting recognition to peptide-only forces the model to learn
peptide-intrinsic features of immunogenicity (foreignness, unusual residues)
rather than confounding them with binding/presentation.

---

#### `recognition_cd4` -- CD4 Recognition

**Cross-attends to:**

| Source | Tokens/Vectors | Positional Encoding |
|--------|---------------|---------------------|
| Peptide | p_1 ... p_L | Triple-frame |
| `core_context_vec` | (1, d_model) | -- |

**Does NOT attend to:** MHC, flanks, upstream latents.

**Planned future extension (currently disabled):** gated TCR cross-attention
(same mechanism as recognition_cd8; see `tcr_spec.md`).

**Why core_context_vec for CD4 but not CD8?** For MHC-II, the peptide
extends beyond both ends of the groove. The core residues (P1-P9) are
groove-bound, while PFRs are solvent-exposed. CD4 TCRs contact both
core-exposed positions (P2, P5, P8) AND N-terminal PFR residues. The
core_context_vec tells the recognition latent WHERE the core is, so it
can weight PFR vs core contributions appropriately.

Arnold et al. (J Immunol 2002) showed 78% of MHC-II epitopes have
PFR-dependent T cells. The core_context_vec is essential for modeling this.

For class I (CD8), the core IS the full peptide, so core_context_vec
adds no information.

---

### Level 3: Immunogenicity (computed after Level 1 + Level 2.5)

---

#### `immunogenicity_cd8` -- CD8 Immunogenicity

**Computed as MLP (not cross-attention):**

```python
immunogenicity_cd8_input = concat(binding_affinity_vec, binding_stability_vec,
                                   recognition_cd8_vec)
immunogenicity_cd8_vec = MLP_immunogenicity_cd8(immunogenicity_cd8_input)
# 2-layer MLP, 3*d_model -> d_model
```

**Does NOT attend to:** Any tokens. No cross-attention.

**Dependencies:** binding_affinity_vec, binding_stability_vec, recognition_cd8_vec.

**Biological rationale:** Immunogenicity = "will a T-cell response happen?"
This requires:
1. Sufficient binding to generate surface pMHC density (binding_affinity)
2. Sufficient stability for sustained TCR engagement (binding_stability)
3. Recognition by the TCR repertoire (recognition_cd8)

**Why no presentation dependency?** Presentation is the conjunction of
processing + binding. Immunogenicity already depends on binding. Adding
presentation would create redundant paths (binding_affinity ->
immunogenicity_cd8 directly, AND binding_affinity -> presentation_class1
-> immunogenicity_cd8). Processing does not directly affect immunogenicity
-- it affects whether the peptide is AVAILABLE, which is a question for
presentation/elution prediction, not for "given this peptide is presented,
will it be immunogenic?"

T-cell assay outputs (S10) handle the full biology including processing,
by routing through presentation when the assay format requires it.

**Why MLP instead of cross-attention?** The immunogenicity latent's inputs
are all fixed-size vectors (not token sequences). Cross-attention over 3
vectors is equivalent to a learned weighted sum. An MLP is simpler and
equally expressive for this case.

---

#### `immunogenicity_cd4` -- CD4 Immunogenicity

```python
immunogenicity_cd4_input = concat(binding_affinity_vec, binding_stability_vec,
                                   recognition_cd4_vec)
immunogenicity_cd4_vec = MLP_immunogenicity_cd4(immunogenicity_cd4_input)
# 2-layer MLP, 3*d_model -> d_model
```

Same structure as immunogenicity_cd8, using recognition_cd4_vec.

---

## 7.5 Complete Segment Access Table

| Latent | nflank | peptide | cflank | mhc_a | mhc_b | Upstream latent deps | Extra tokens |
|--------|--------|---------|--------|-------|-------|---------------------|--------------|
| processing_class1 | YES | YES | YES | -- | -- | -- | context_vec |
| processing_class2 | YES | YES | YES | -- | -- | -- | context_vec |
| binding_affinity | -- | YES | -- | YES | YES | -- | context_vec, core_rel_pos |
| binding_stability | -- | YES | -- | YES | YES | -- | context_vec, core_rel_pos |
| presentation_class1 | -- | -- | -- | -- | -- | processing_class1, binding_affinity, binding_stability | context_vec |
| presentation_class2 | -- | -- | -- | -- | -- | processing_class2, binding_affinity, binding_stability | context_vec, core_context_vec |
| recognition_cd8 | -- | YES | -- | -- | -- | -- | (+-TCR) |
| recognition_cd4 | -- | YES | -- | -- | -- | -- | core_context_vec, (+-TCR) |
| immunogenicity_cd8 | -- | -- | -- | -- | -- | binding_affinity, binding_stability, recognition_cd8 | (MLP, no cross-attn) |
| immunogenicity_cd4 | -- | -- | -- | -- | -- | binding_affinity, binding_stability, recognition_cd4 | (MLP, no cross-attn) |
| ms_detectability | -- | YES | -- | -- | -- | -- | -- |

**`context_vec`** (S5.4) is a learned projection of inferred MHC class
probabilities, per-chain species probabilities, and chain compatibility.
It carries higher-level biological identity derived from MHC sequences.
For processing latents, it is the **sole channel** of MHC-class information
(they cannot see MHC tokens). For binding latents (which can see MHC tokens),
it provides pre-computed class/species summaries so the cross-attention
layers can focus on peptide-groove complementarity rather than re-deriving
chain identity.

**`core_context_vec`** (S6.3) is a soft-attention-weighted average of peptide
hidden states, where weights come from the core-start predictor. It
summarizes "what the binding core looks like" as a single vector, telling
downstream latents where the MHC-II core is relative to the peptide
flanking regions. Only consumed by latents where the core/PFR distinction
matters: `presentation_class2` (DM editing depends on core register) and
`recognition_cd4` (CD4 TCRs contact both core-exposed and PFR positions).

---

# 8. Multi-Allele / MIL Aggregation

## 8.1 Patient Allele Set

Maximum per patient:
- **Class I**: 2 HLA-A + 2 HLA-B + 2 HLA-C = 6 molecules
- **Class II DR**: 2 DRB1 (x monomorphic DRA) = 2 molecules
- **Class II DQ**: Up to 4 molecules (2 DQA1 x 2 DQB1, cis + trans)
- **Class II DP**: Up to 4 molecules (2 DPA1 x 2 DPB1, cis + trans)

Maximum: 6 class I + 10 class II = 16 molecules.

## 8.2 Per-Allele Forward Pass

```python
def per_allele_forward(peptide, flanks, allele, species, tcr=None):
    # Shared computation (once)
    pep_flank_H = base_encoder(pep_flank_tokens)
    mhc_H = base_encoder(mhc_tokens)

    # MHC inference
    mhc_a_vec = masked_mean(mhc_H, mhc_a_mask)
    mhc_b_vec = masked_mean(mhc_H, mhc_b_mask)
    class_probs, context_vec = mhc_chain_inference(mhc_a_vec, mhc_b_vec)

    # Core identification
    core_info = core_pointer(pep_flank_H, mhc_a_vec, mhc_b_vec)

    # Level 0: Processing (shared across alleles) + MS detectability
    processing_class1_vec = compute_latent(
        processing_class1_query, kv=[pep_flank_H[pep+flank], context_vec])
    processing_class2_vec = compute_latent(
        processing_class2_query, kv=[pep_flank_H[pep+flank], context_vec])
    ms_detectability_vec = compute_latent(
        ms_detectability_query, kv=[pep_flank_H[pep_only]])

    # Level 1: Binding
    pep_mhc_kv = concat(pep_H_with_core_rel_pos, mhc_H, context_vec)
    binding_affinity_vec = compute_latent(binding_affinity_query, kv=pep_mhc_kv)
    binding_stability_vec = compute_latent(binding_stability_query, kv=pep_mhc_kv)

    # Level 2: Presentation (NO token access)
    presentation_class1_vec = compute_latent(
        presentation_class1_query,
        kv=[processing_class1_vec, binding_affinity_vec,
            binding_stability_vec, context_vec])
    presentation_class2_vec = compute_latent(
        presentation_class2_query,
        kv=[processing_class2_vec, binding_affinity_vec,
            binding_stability_vec, core_context_vec, context_vec])

    # Level 2.5: Recognition (peptide only in canonical path)
    # Planned future extension: gated TCR integration (disabled for now).
    recognition_cd8_vec = compute_latent(recognition_cd8_query, kv=pep_H)
    recognition_cd4_vec = compute_latent(recognition_cd4_query, kv=[pep_H, core_context_vec])

    # Level 3: Immunogenicity (MLP)
    immunogenicity_cd8_vec = MLP_immunogenicity_cd8(
        concat(binding_affinity_vec, binding_stability_vec, recognition_cd8_vec))
    immunogenicity_cd4_vec = MLP_immunogenicity_cd4(
        concat(binding_affinity_vec, binding_stability_vec, recognition_cd4_vec))

    return AlleleResult(
        processing_class1_vec, processing_class2_vec,
        binding_affinity_vec, binding_stability_vec,
        presentation_class1_vec, presentation_class2_vec,
        recognition_cd8_vec, recognition_cd4_vec,
        immunogenicity_cd8_vec, immunogenicity_cd4_vec,
        ms_detectability_vec, core_info, class_probs)
```

## 8.3 MIL Aggregation

### 8.3.1 Allele Competition Transformer

```python
def allele_competition(allele_pres_vectors):
    """1-2 layer transformer over the allele dimension."""
    x = allele_pres_vectors  # (n_alleles, d_model)
    for layer in competition_transformer_layers:
        residual = x
        x = layer_norm(x)
        x = self_attention(x)     # alleles attend to each other
        x = residual + x
        residual = x
        x = layer_norm(x)
        x = ffn(x)
        x = residual + x
    return x
```

**Rationale:** Alleles compete for peptides. High-affinity binding by one
allele sequesters copies, reducing presentation by others. Tiny cost:
operates over at most 16 tokens with 1-2 layers.

### 8.3.2 Attention-Based MIL Pooling

```python
def mil_pool(allele_vectors, competition=True):
    if competition:
        allele_vectors = allele_competition(allele_vectors)

    # Ilse et al. (ICML 2018) attention pooling
    attn_hidden = tanh(Linear1(allele_vectors))
    attn_logits = Linear2(attn_hidden)
    attn_weights = softmax(attn_logits, dim=0)
    patient_vector = (attn_weights * allele_vectors).sum(dim=0)
    return patient_vector, attn_weights.squeeze(-1)
```

### 8.3.3 Aggregation at Two Levels

```python
# Presentation-level MIL (with allele competition)
patient_presentation_class1_vec, pres_c1_weights = mil_pool(
    class1_presentation_stack, competition=True)
patient_presentation_class2_vec, pres_c2_weights = mil_pool(
    class2_presentation_stack, competition=True)

# Immunogenicity-level MIL (after per-allele recognition)
patient_immunogenicity_cd8_vec, imm_cd8_weights = mil_pool(
    class1_immunogenicity_stack, competition=False)
patient_immunogenicity_cd4_vec, imm_cd4_weights = mil_pool(
    class2_immunogenicity_stack, competition=False)
```

**Why aggregate immunogenicity per-allele:** Presentation is allele-specific,
so the peptide-level recognition signal is gated by allele-specific upstream
presentation strength before patient-level pooling.

## 8.4 DQ/DP Trans-Heterodimer Handling

Include all 4 possible pairings (2 cis + 2 trans) as MIL instances.
Non-functional pairings get low MIL attention weights because they
produce low binding/presentation scores. Optionally gate by
chain_compat_prob (S5.3).

---

# 9. Output Heads

## 9.1 Binding/Stability Heads

Class-conditioned readout from shared binding latent vectors:

```python
# Binding affinity
binding_base = BindingModule(binding_affinity_vec, binding_stability_vec)
# Outputs: {log_koff, log_kon_intrinsic, log_kon_chaperone}
# Derives: KD, t_half

# Class-specific calibration
binding_class1_logit = binding_base_logit + delta_class1(class_probs)
binding_class2_logit = binding_base_logit - delta_class2(class_probs)
binding_logit = class_probs[0] * binding_class1_logit + class_probs[1] * binding_class2_logit

# Assay-specific heads
kd_nM = KDHead(binding_affinity_vec)          # log10(nM) regression
ic50_nM = IC50Head(binding_affinity_vec)      # log10(nM) regression
t_half = THalfHead(binding_stability_vec)     # hours regression
tm = TmHead(binding_stability_vec)            # Celsius regression
```

**Censor-aware regression loss** for binding data with <, =, > qualifiers.

## 9.2 Processing Heads

```python
processing_class1_logit = Linear(processing_class1_vec)   # P(processed by class I pathway)
processing_class2_logit = Linear(processing_class2_vec)   # P(processed by class II pathway)

# Mixed processing probability (weighted by class_probs)
processing_logit = (class_probs[0] * processing_class1_logit
                  + class_probs[1] * processing_class2_logit)
```

## 9.3 Presentation/Elution Heads

```python
presentation_class1_logit = PresentationBottleneck(
    processing_class1_vec, binding_affinity_vec, binding_stability_vec)
presentation_class2_logit = PresentationBottleneck(
    processing_class2_vec, binding_affinity_vec, binding_stability_vec)
presentation_logit = (class_probs[0] * presentation_class1_logit
                    + class_probs[1] * presentation_class2_logit)

# Elution/MS output includes MS detectability bias
elution_logit = presentation_logit + ms_detectability_logit
ms_logit = presentation_logit + ms_detectability_logit
```

## 9.4 Recognition Heads

```python
# Population-level
recognition_cd8_logit = Linear(recognition_cd8_vec)
recognition_cd4_logit = Linear(recognition_cd4_vec)
recognition_repertoire_logit = (class_probs[0] * recognition_cd8_logit
                              + class_probs[1] * recognition_cd4_logit)

# TCR-specific matching (planned future path; currently disabled)
match_logit = TCRpMHCMatcher(tcr_vec, pmhc_vec)
```

## 9.5 Immunogenicity Heads

```python
immunogenicity_cd8_logit = Linear(immunogenicity_cd8_vec)
immunogenicity_cd4_logit = Linear(immunogenicity_cd4_vec)
immunogenicity_logit = (class_probs[0] * immunogenicity_cd8_logit
                      + class_probs[1] * immunogenicity_cd4_logit)
```

## 9.6 Auxiliary Heads

| Head | Input | Output | Training target |
|------|-------|--------|----------------|
| mhc_a_chain_type | mhc_a_vec | softmax over {class_I_alpha, class_II_alpha, unknown} | Chain type labels |
| mhc_b_chain_type | mhc_b_vec | softmax over {class_I_beta, class_II_beta, unknown} | Chain type labels |
| mhc_a_species | mhc_a_vec | softmax over {human, murine, nhp, other} | Species labels |
| mhc_b_species | mhc_b_vec | softmax over same | Species labels |
| source_species | processing_class1_vec or processing_class2_vec | softmax over {human, murine, nhp, other} | Source protein species |
| chain_compat | mhc_a_vec + mhc_b_vec | sigmoid | Valid/invalid chain pairing |
| mhc_class | class_probs | {I, II} | MHC class label |
| core_start | core_start_probs | position | Annotated core starts |

## 9.7 pMHC Embedding (`pmhc_vec`)

A fixed-size embedding of the overall pMHC complex, used for retrieval,
similarity search, and as the pMHC-side anchor for TCR matching
(`tcr_spec.md`):

```python
pmhc_vec = Linear_pmhc(concat(binding_affinity_vec,
                                presentation_class1_vec,
                                presentation_class2_vec))  # -> (d_model,)
```

This is a learned projection from the latent vectors that best summarize
the pMHC complex identity. It is NOT a pooled-CLS-token output — with
segment-blocked base attention, there is no global pooling token. Instead,
`pmhc_vec` is derived from the latent DAG outputs, which have already
integrated cross-segment information through their cross-attention layers.

---

# 10. T-Cell Assay Output System

## 10.1 Design Philosophy

T-cell assays measure the same underlying biology through different
experimental lenses. Rather than modeling each assay type as an independent
output, we use **compositional context embeddings** that capture the
systematic biases of each experimental configuration, generating all
attested combinations as parallel outputs.

## 10.2 Context Embedding Dimensions

Seven embedding tables define the assay configuration space:

```python
method_emb   = nn.Embedding(n_methods, d_ctx)      # ELISpot, ICS, multimer, ...
readout_emb  = nn.Embedding(n_readouts, d_ctx)      # IFNg, IL-2, TNFa, ...
apc_emb      = nn.Embedding(n_apcs, d_ctx)           # DC, PBMC, B-LCL, ...
culture_emb  = nn.Embedding(n_cultures, d_ctx)       # ex_vivo, short_restim, IVS, ...
stim_emb     = nn.Embedding(n_stims, d_ctx)           # ex_vivo, in_vitro_stim, ...
pepfmt_emb   = nn.Embedding(n_pepfmts, d_ctx)        # minimal, long, pool, TMG, ...
duration_emb = nn.Linear(1, d_ctx)                    # log(culture_duration_hours)
```

### 10.2.1 Peptide Format Categories

| Category | Meaning | Processing relevant? | Class ambiguity? |
|----------|---------|---------------------|------------------|
| `MINIMAL_EPITOPE` | 8-15mer exact peptide pulsed | No -- loaded directly onto MHC | No -- experimenter knows restriction |
| `LONG_PEPTIDE` | 16-30mer single peptide pulsed | Yes -- APC must cleave | Yes -- could be class I or II |
| `PEPTIDE_POOL` | Overlapping long peptides covering region/protein | Yes | Yes, and multi-epitope |
| `WHOLE_PROTEIN` | Full protein antigen | Yes, fully | Yes, fully |
| `TMG` | Tandem minigene construct (concatenated epitopes in expression vector) | Yes -- APC processes the construct | Yes -- multiple epitopes presented |
| `PEPTIDE_MIX` | Defined mixture of synthetic peptides (e.g., CEF pool) | Depends on peptide lengths in mix | Partially -- could be minimal or long |
| `unknown` | Not annotated | Assume worst case | Assume ambiguous |

**Detection logic from IEDB data:**
- Peptide length <= 15 and epitope type "exact" -> MINIMAL_EPITOPE
- Peptide length >= 16 and single peptide -> LONG_PEPTIDE
- Assay description mentions "pool" or "overlapping" -> PEPTIDE_POOL
- Epitope annotated as protein/whole antigen -> WHOLE_PROTEIN
- Immunogen description contains "TMG" pattern (e.g., "3998-TMG1-AMER1-mut") -> TMG
- Defined peptide mixture (e.g., CEF pool) -> PEPTIDE_MIX
- Otherwise -> unknown

**TMG in the data**: IEDB contains 1,231 T-cell records with TMG constructs
(tandem minigenes encoding concatenated neoepitopes). When a TMG is the
immunogen, the APC is transduced with the construct and processes/presents
the epitopes naturally. The tested peptide is a specific epitope within the
TMG. TMGs are biologically similar to WHOLE_PROTEIN in that full processing
is required, but the "protein" is synthetic and encodes only selected
epitopes.

### 10.2.2 Culture Duration

IEDB's controlled vocabulary for culture conditions (e.g., "Short Term
Restimulated", "Direct Ex Vivo") does not include specific duration
information. However, stimulation duration matters biologically:

- **12h restim**: Barely any expansion. Primarily detects pre-existing
  memory/effector cells. Low amplification bias.
- **5-7 day restim**: Moderate expansion. Standard "short-term" IVS.
  10-100x amplification with affinity selection.
- **14-21 day culture**: Full IVS. Massive expansion. Strong selection
  for high-affinity clones. Detection sensitivity is high but specificity
  drops (bystander activation).

**Approach**: Add continuous `log_culture_duration_hours` as an additional
context feature, with a learned default prior per culture category:

```python
# For each culture category, learn a default log-duration
default_log_duration = nn.Embedding(n_cultures, 1)  # per-category defaults

if culture_duration_hours is not None:
    log_dur = log(culture_duration_hours + 1.0)
else:
    log_dur = default_log_duration[culture_idx]  # learned prior

duration_feature = duration_emb(log_dur)  # (d_ctx,)
```

When duration is unknown (most records), the model falls back to the
learned per-category default. When duration is known, it provides a
finer-grained signal.

### 10.2.3 Full Context Vector

```python
ctx_vec = (method_emb(config.method) + readout_emb(config.readout)
         + apc_emb(config.apc) + culture_emb(config.culture)
         + stim_emb(config.stim) + pepfmt_emb(config.pepfmt)
         + duration_feature)
# ctx_vec: (d_ctx,)
```

## 10.3 T-Cell Output Formula

For each attested assay configuration:

```python
def tcell_output(self, config,
                 presentation_class1_logit, presentation_class2_logit,
                 immunogenicity_cd8_vec, immunogenicity_cd4_vec,
                 binding_class1_logit, binding_class2_logit,
                 class_probs):

    ctx_vec = compute_ctx_vec(config)

    # Bias: scalar sensitivity/threshold shift
    bias = self.bias_proj(ctx_vec)                                  # scalar

    # Feature gate: which immunogenicity features this readout detects
    gate = sigmoid(self.gate_proj(ctx_vec))                         # (d_model,)

    # Signal from gated immunogenicity vectors
    cd8_signal = self.signal_proj(gate * immunogenicity_cd8_vec)    # scalar
    cd4_signal = self.signal_proj(gate * immunogenicity_cd4_vec)    # scalar

    # Soft processing gate: does processing matter for this config?
    proc_weight = sigmoid(self.proc_gate(ctx_vec))     # scalar in [0, 1]
    # MINIMAL_EPITOPE -> ~0, LONG_PEPTIDE/TMG -> ~1

    # Soft class ambiguity gate
    class_ambiguity = sigmoid(self.ambiguity_gate(ctx_vec))  # scalar in [0, 1]
    # MINIMAL_EPITOPE -> ~0, LONG_PEPTIDE/POOL/TMG -> ~1

    # Upstream biology with soft processing inclusion
    cd8_upstream = (proc_weight * presentation_class1_logit
                  + (1 - proc_weight) * binding_class1_logit)
    cd4_upstream = (proc_weight * presentation_class2_logit
                  + (1 - proc_weight) * binding_class2_logit)

    # Per-lineage logits
    cd8_logit = cd8_upstream + cd8_signal + bias
    cd4_logit = cd4_upstream + cd4_signal + bias

    # Class routing: known vs ambiguous
    known_class_logit = class_probs[:, 0:1] * cd8_logit + class_probs[:, 1:2] * cd4_logit
    ambiguous_logit = noisy_or_logit(cd8_logit, cd4_logit)

    output_logit = (1 - class_ambiguity) * known_class_logit + class_ambiguity * ambiguous_logit

    return output_logit
```

### 10.3.1 Concrete Example: IFNg ELISpot Variants

**Ex vivo + minimal epitope + DCs:**
```
proc_weight -> ~0 (processing bypassed, peptide loaded directly)
class_ambiguity -> ~0 (experimenter knows restriction)
output = binding_class1_logit + cd8_signal + bias
```

**Ex vivo + long peptide (16-30mer) + PBMCs:**
```
proc_weight -> ~1 (APC must process)
class_ambiguity -> ~1 (could be class I or II)
output = noisy_or(presentation_class1_logit + cd8_signal,
                  presentation_class2_logit + cd4_signal) + bias
```

**IVS (short restim) + minimal epitope + B-LCL:**
```
proc_weight -> ~0 (minimal epitope, processing bypassed)
class_ambiguity -> ~0 (known restriction)
output = binding_class1_logit + cd8_signal + bias
# bias_ivs >> bias_exvivo (IVS amplifies signal 10-100x)
```

**Ex vivo + peptide pool + PBMCs:**
```
proc_weight -> ~1 (long peptides need processing)
class_ambiguity -> ~1 (ambiguous lineage)
output = noisy_or(presentation_class1_logit + cd8_signal,
                  presentation_class2_logit + cd4_signal) + bias
# This measures protein-level immunogenicity, not peptide-level
```

**Ex vivo + TMG + autologous APCs:**
```
proc_weight -> ~1 (TMG requires full processing)
class_ambiguity -> ~1 (multiple epitopes, both classes possible)
output = noisy_or(presentation_class1_logit + cd8_signal,
                  presentation_class2_logit + cd4_signal) + bias
```

## 10.4 Attested Combination Registry

At model init, enumerate all (method, readout, APC, culture, stim, pepfmt)
tuples that appear in the training data. Store as a registry:

```python
self.attested_configs = [
    Config(method=ELISPOT, readout=IFNG, apc=DC, culture=EX_VIVO,
           stim=EX_VIVO, pepfmt=MINIMAL_EPITOPE),
    Config(method=ELISPOT, readout=IFNG, apc=PBMC, culture=SHORT_RESTIM,
           stim=IN_VITRO_STIM, pepfmt=LONG_PEPTIDE),
    Config(method=ICS, readout=IFNG, apc=PBMC, culture=EX_VIVO,
           stim=EX_VIVO, pepfmt=MINIMAL_EPITOPE),
    # ... all attested combinations
]
```

At inference, ALL attested combinations are evaluated in parallel (batched
embedding lookups + projections). The model outputs a dict of
`{config_key: logit}` for every attested assay configuration.

## 10.5 Consistency Priors

Soft constraints between T-cell outputs encoding known biology:

| Prior | Constraint | Rationale |
|-------|-----------|-----------|
| Sensitivity ordering | P(ELISpot+) >= P(ICS+) for same peptide/context | ELISpot detects 5-10x more cells than ICS (Karlsson et al., J Immunol Methods 2003) |
| Culture amplification | P(IVS+) >= P(ex_vivo+) for same peptide | In vitro stimulation expands reactive cells |
| Multimer-function gap | P(multimer+) >= P(ELISpot+) for same peptide | Not all multimer+ cells produce cytokine |
| Cross-readout consistency | P(IFNg+) and P(TNFa+) are positively correlated | Polyfunctional T cells produce multiple cytokines |
| Th1/Th2 partition | IFNg/TNFa (Th1) vs IL-4/IL-5 (Th2) are negatively correlated within CD4 | T-helper polarization |

Implemented as soft margin losses during training:
```python
# Example: ELISpot >= ICS sensitivity
elispot_logit = tcell_outputs[elispot_config]
ics_logit = tcell_outputs[ics_config]
margin = 0.2  # in logit space
L_sensitivity = relu(ics_logit - elispot_logit + margin).square().mean()
```

## 10.6 Context Embedding Visualization

The compositional embedding structure is directly interpretable:

```python
# Visualize readout embeddings
readout_vecs = readout_emb.weight.detach()  # (n_readouts, d_ctx)
pca = PCA(n_components=2).fit_transform(readout_vecs)
# Expected: IFNg near TNFa near granzyme_B (Th1/cytotoxic cluster)
#           IL-4 near IL-5 (Th2 cluster)
#           multimer_binding far from all cytokines

# Visualize full config embeddings for attested combinations
all_ctx_vecs = [compute_ctx_vec(c) for c in attested_configs]
pca_configs = PCA(n_components=2).fit_transform(stack(all_ctx_vecs))
# Expected axis 1: readout type (largest biological variance)
# Expected axis 2: culture/peptide format (sensitivity/calibration variance)
```

---

# 11. Missing Input Handling

## 11.1 Strategy Summary

| Missing Input | Strategy | Mechanism | Affected Latents |
|---------------|----------|-----------|-----------------|
| TCR (entirely) | Future feature (inactive) | Canonical path ignores TCR inputs for now | -- |
| TCR alpha only | Future feature (inactive) | Canonical path ignores TCR inputs for now | -- |
| TCR beta only | Future feature (inactive) | Canonical path ignores TCR inputs for now | -- |
| N-flank | Missing token | Shared learned `<MISSING>` token + completeness flag | processing_class1/class2 |
| C-flank | Missing token | Shared learned `<MISSING>` token + completeness flag | processing_class1/class2 |
| MHC beta (class I) | Default canonical | Species-canonical beta2m | binding_affinity/stability, presentation_class1 |
| MHC (entirely) | Missing token | Shared learned `<MISSING>` token + completeness flag | binding, presentation |
| MHC class label | Unknown enum | Inferred from sequence | Global conditioning |
| Species | Default | Default to `human` | Global conditioning |
| Culture duration | Learned prior | Per-category default | T-cell outputs |

TCR pathway details are tracked in `tcr_spec.md` and remain future work.

## 11.2 Training-Time Random Masking

| Input | Masking probability | Rationale |
|-------|-------------------|-----------|
| N-flank | 30% | Forces good flank-absent behavior |
| C-flank | 30% | Same; also regularizer |
| MHC beta (class I) | 10% | Replace with canonical beta2m |
| MHC class label | 10% | Forces inference from sequence |

TCR masking is deferred until the TCR pathway is re-enabled.

---

# 12. Dimensions and Sizing

## 12.1 Recommended Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `d_model` | 256 | Start here; scale to 512 if capacity-limited |
| `d_ctx` | 64 | Context embedding dimension for T-cell system |
| `N_base` | 6 | Base encoder layers (4-8 range) |
| `N_latent` | 2 | Cross-attention layers per latent |
| `N_tcr` | 3 | TCR encoder layers |
| `N_competition` | 1 | Allele competition transformer layers |
| `N_heads` | 8 | Attention heads everywhere |
| `d_ffn` | 4 * d_model = 1024 | Standard transformer ratio |
| Max peptide length | 50 | Covers class II (longest IEDB binder ~44mer) |
| Max flank length | 20 | Each side |
| Max MHC chain length | 400 | Full-length sequence per chain |
| Max TCR chain length | 150 | Full V-region |
| Max alleles per patient | 16 | 6 class I + 10 class II |

## 12.2 Approximate Parameter Count (d_model = 256)

| Component | Parameters | Notes |
|-----------|-----------|-------|
| Embedding tables | ~1.0M | All tables (incl. full-length MHC positional) |
| Base encoder (6 layers) | ~4.8M | Segment-blocked transformer |
| Core pointer head | ~0.3M | Linear layers + MLP |
| Latent DAG (11 latents x 2 layers) | ~5.8M | Cross-attention + FFN (9 CA + 2 MLP) |
| TCR encoder (3 layers) | ~2.4M | Separate transformer |
| MHC inference module | ~0.2M | Per-chain heads + context MLP |
| Allele competition (1 layer) | ~0.4M | Tiny transformer |
| MIL attention heads | ~0.1M | 2 separate MIL pools |
| Output heads (12+ latents) | ~1.2M | Shared + assay-conditioned |
| T-cell context system | ~0.3M | Embedding tables + gates |
| **Total** | **~17M** | |

At d_model = 512: approximately 55M parameters.

---

# Appendix A: Context Vocabulary

## A.1 T-Cell Assay Methods
```python
TCELL_ASSAY_METHODS = [
    "unknown", "ELISPOT", "ICS", "MULTIMER", "CYTOTOXICITY",
    "PROLIFERATION", "ELISA", "LUMINEX", "CYTOKINE_CAPTURE",
    "DEGRANULATION", "ACTIVATION_MARKER", "OTHER",
]
```

## A.2 T-Cell Readouts
```python
TCELL_ASSAY_READOUTS = [
    "unknown", "IFNg", "IL2", "TNFa", "IL4", "IL5", "IL10",
    "IL17", "GMCSF", "GRANZYME_B", "PERFORIN", "CD107A",
    "MULTIMER_BINDING", "PROLIFERATION", "ACTIVATION",
    "OTHER",
]
```

## A.3 APC Types
```python
TCELL_APC_TYPES = [
    "unknown", "DC", "PBMC", "BLCL", "T2", "K562",
    "AUTOLOGOUS", "MONOCYTE", "MACROPHAGE", "OTHER",
]
```

## A.4 Culture Contexts
```python
TCELL_CULTURE_CONTEXTS = [
    "unknown", "DIRECT_EX_VIVO", "SHORT_RESTIM", "IN_VITRO",
    "IN_VIVO", "ENGINEERED", "CELL_LINE_CLONE",
    "NON_SPECIFIC_ACTIVATION", "OTHER",
]
```

## A.5 Stimulation Contexts
```python
TCELL_STIM_CONTEXTS = [
    "unknown", "EX_VIVO", "IN_VITRO_STIM", "IN_VIVO",
    "ENGINEERED", "OTHER",
]
```

## A.6 Peptide Formats
```python
TCELL_PEPTIDE_FORMATS = [
    "unknown",           # not annotated
    "MINIMAL_EPITOPE",   # exact 8-15mer pulsed
    "LONG_PEPTIDE",      # single 16-30mer pulsed
    "PEPTIDE_POOL",      # overlapping peptides covering a region
    "WHOLE_PROTEIN",     # full protein antigen
    "TMG",               # tandem minigene construct
    "PEPTIDE_MIX",       # defined mixture of synthetic peptides
]
```

---

# Appendix B: Key References

| Topic | Reference |
|-------|-----------|
| MHC-II 9mer core structure | Stern et al., *Nature* 1994 |
| PFR-dependent T cell recognition (78%) | Arnold et al., *J Immunol* 2002 |
| Peptide immunogenicity features | Calis et al., *PLoS Comp Bio* 2013 |
| ELISpot vs ICS sensitivity | Karlsson et al., *J Immunol Methods* 2003 |
| Attention-based MIL pooling | Ilse et al., ICML 2018 |
| Flamingo late cross-attention | Alayrac et al., NeurIPS 2022 |
| STAPLER pMHC transformer | Shapiro et al., bioRxiv 2024 |
| TULIP TCR-epitope modeling | Meynard-Piganeau et al., *PNAS* 2024 |
| MHCflurry binding prediction | O'Donnell et al., *Cell Systems* 2020 |
| NetMHCpan pseudosequences | Reynisson et al., *NAR* 2020 |
| Neoantigen TMG screening | Linnemann et al., *Nat Med* 2015 |
