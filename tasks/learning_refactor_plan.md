# Learning Refactor: Implementation Plan

This plan describes how to implement the fixes catalogued in `tasks/learning_refactor.md` to resolve the allele-invariant prediction problem analyzed in `tasks/training_analysis.md`.

The implementing agent should read both of those documents first. This plan provides implementation ordering, code locations, inter-item dependencies, and enough implicit context to apply all changes together.

---

## How This Codebase Works (Critical Context)

### Model Architecture Overview

Presto is a multi-task pMHC (peptide-MHC) binding prediction model. It predicts:
- **Processing probability**: will the proteasome cleave this peptide from the source protein?
- **Binding affinity/stability**: will this peptide bind to a specific MHC allele?
- **Presentation probability**: will this peptide-MHC complex appear on the cell surface?
- **Recognition/Immunogenicity**: will a T-cell recognize and respond to this complex?

The model processes amino acid sequences through a shared transformer encoder, then uses a **latent variable DAG** where each biological concept is a learned query that cross-attends to relevant encoder tokens.

### Token Streams

The encoder receives concatenated amino acid sequences as a single stream:

```
[nflank] [peptide] [cflank] [mhc_a] [mhc_b]
```

- `nflank`/`cflank`: Source protein flanking regions (cleavage context for processing)
- `peptide`: The peptide sequence (8-50 amino acids)
- `mhc_a`: MHC alpha chain amino acid sequence (~300 residues)
- `mhc_b`: MHC beta chain amino acid sequence (~100-300 residues; beta-2-microglobulin for Class I, DRB/DQB/DPB for Class II)
- Each segment has its own learned segment embedding added to token embeddings

After N transformer layers, the hidden states `h` are used by the latent DAG.

### Latent DAG (Current)

12 latent variables organized in a DAG. Each latent is a single 256-dim vector produced by cross-attention of a learned query over permitted token segments:

```
LATENT_SEGMENTS = {                                    # presto.py:182-196
    "processing_class1":    ["nflank", "peptide", "cflank"],
    "processing_class2":    ["nflank", "peptide", "cflank"],
    "ms_detectability":     ["peptide"],
    "species_of_origin":    ["peptide"],
    "binding_affinity":     ["peptide", "mhc_a", "mhc_b"],  # <-- only 2 latents see MHC
    "binding_stability":    ["peptide", "mhc_a", "mhc_b"],
    "presentation_class1":  [],    # no token access, deps only
    "presentation_class2":  [],    # no token access, deps only
    "recognition_cd8":      ["peptide"],
    "recognition_cd4":      ["peptide"],
    "immunogenicity_cd8":   [],    # MLP only
    "immunogenicity_cd4":   [],    # MLP only
}
```

Each latent query goes through 2 layers of cross-attention + FFN. Dependencies inject upstream latent vectors as extra KV tokens.

### The Core Problem

Only `binding_affinity` and `binding_stability` see MHC token sequences. Each uses 1 query compressing ~500 tokens into 256 dims. The binding vector is then crushed to 3 scalars (koff, kon_intrinsic, kon_chaperone) by `BindingModule` (771 params total). Everything downstream — presentation, immunogenicity, TCR matching — sees only these scalars or small residuals from them. Allele identity is lost.

---

## Implementation Phases

### Phase 0: Quick Wins (Independent, Do First)

Items: **E3** (weight init), **E4** (LR scheduler), **D1** (mhc_class default)

These are independent of each other and of the architecture changes. They improve training dynamics immediately and can be validated with a quick smoke test.

#### E3: Fix Weight Initialization

**Files to modify:**
- `models/presto.py` — add `_init_weights()` method, call from `__init__`

**Current state:** No explicit initialization anywhere. PyTorch defaults apply:
- Latent queries: `torch.randn(d_model) * 0.02` (lines 342-355) — too small, causes near-uniform attention
- `nn.Embedding`: `N(0, 1)` — too large (L2 norm ≈ 16)
- `nn.MultiheadAttention`: Kaiming uniform — suboptimal for transformers
- `nn.Linear` in heads: Kaiming uniform — fine, don't change

**Implementation:**
```python
def _init_weights(self):
    d = self.d_model
    # (a) Latent queries: Xavier scale
    for name, param in self.latent_queries.items():
        nn.init.normal_(param.data, std=1.0 / math.sqrt(d))

    # (b) Embeddings: scaled normal
    for m in self.modules():
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=1.0 / math.sqrt(d))

    # (c) MultiheadAttention: Xavier for Q/K/V projections
    for m in self.modules():
        if isinstance(m, nn.MultiheadAttention):
            for name, param in m.named_parameters():
                if "in_proj" in name or "q_proj" in name or "k_proj" in name or "v_proj" in name:
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        nn.init.zeros_(param)

    # (d) nn.Linear in heads: keep Kaiming default (no action)
```

Call `self._init_weights()` at the end of `__init__`.

**Validation:** After init, check that latent query norms are ~1.0 (not ~0.3), embedding norms are ~1.0 (not ~16). Run a single forward pass and verify attention weights are non-uniform.

#### E4: Add LR Scheduler

**Files to modify:**
- `scripts/train_iedb.py` — add scheduler after optimizer creation
- `scripts/train_synthetic.py` — same
- `scripts/train_modal.py` — same (if it has its own training loop)

**Current state:** Fixed LR 2.8e-4, no scheduler (train_iedb.py line ~126)

**Implementation:**
```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

total_steps = num_epochs * len(dataloader)
warmup_steps = int(0.05 * total_steps)  # 5% warmup

warmup_scheduler = LinearLR(optimizer, start_factor=1e-6 / lr, end_factor=1.0, total_iters=warmup_steps)
cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=lr * 0.1)
scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

# In training loop, after optimizer.step():
scheduler.step()
```

**Validation:** Log LR per step, verify warmup ramp + cosine decay shape.

#### D1: Fix mhc_class Default

**Files to modify:**
- `data/loaders.py` — all Record dataclasses (lines ~64-167)

**Current state:** Every record type defaults `mhc_class: str = "I"`. This silently mislabels records where class is genuinely unknown.

**Implementation:**
1. Change all record dataclasses: `mhc_class: Optional[str] = None`
2. In every loader that creates records, if the source data provides MHC class, set it explicitly. If not, leave as None.
3. Add a utility that infers class from allele name using `mhcgnomes`:
   ```python
   import mhcgnomes
   def infer_mhc_class(allele_name: str) -> Optional[str]:
       try:
           result = mhcgnomes.parse(allele_name)
           if hasattr(result, 'mhc_class'):
               return result.mhc_class  # "I" or "II"
       except:
           pass
       return None
   ```
4. At collation time, if `mhc_class is None`, try inference from allele name. If still None, propagate ambiguity (used by D3 for MIL over pathways).

**IMPORTANT:** The existing `_infer_mhc_class()` at loaders.py:346 uses regex. Keep it as fallback but prefer mhcgnomes. Do NOT write new regexes for allele parsing — always use mhcgnomes.

**Validation:** Grep for `mhc_class.*=.*"I"` to make sure no hardcoded defaults remain. Check that known Class II alleles (DRB1, DQA1, etc.) get correctly classified.

---

### Phase 1: Latent DAG Redesign (Core Architecture)

Items: **A1**, **A2**, **A3**, **A4** — these are tightly coupled and should be implemented together.

This is the highest-impact change. It restructures the latent DAG from 12 class-paired scalar-bottlenecked latents to ~6 unified vector latents with scalar readout heads.

#### New Latent DAG (Target)

Replace the 12-latent class-paired architecture with:

```python
LATENT_SEGMENTS = {
    "processing":        ["nflank", "peptide", "cflank"],
    "ms_detectability":  ["peptide"],
    "species_of_origin": ["peptide"],
    "pmhc_interaction":  ["peptide", "mhc_a", "mhc_b"],   # replaces binding_affinity + binding_stability
    "presentation":      [],                                 # deps only: processing + pmhc_interaction
    "recognition":       ["peptide"],                        # single, with class-specific readout heads
    "immunogenicity":    [],                                 # MLP only
}

LATENT_DEPS = {
    "processing":        [],
    "ms_detectability":  [],
    "species_of_origin": [],
    "pmhc_interaction":  [],                                 # no latent deps, just tokens + context
    "presentation":      ["processing", "pmhc_interaction"],
    "recognition":       ["foreignness"],
    "immunogenicity":    ["pmhc_interaction", "recognition"],
}
```

#### A1 + A2: Merge binding latents, multi-query interaction

**Files to modify:**
- `models/presto.py` — LATENT_SEGMENTS, LATENT_DEPS, LATENT_ORDER, CROSS_ATTN_LATENTS, BINDING_LATENT_NAMES, latent_queries, latent_layers, __init__, forward

**Current state:**
- `binding_affinity` and `binding_stability` are separate latents, same segments, same deps (none), same architecture — two random inits of the same thing
- Multi-query is implemented but disabled (`binding_n_queries=1`) and pools back to 256-dim

**Implementation:**
1. Replace `binding_affinity` + `binding_stability` with single `pmhc_interaction` in all class constants
2. `pmhc_interaction` uses 8 query heads, each projecting to 64 dims → output is `(B, 8, 64)` tensor, which we flatten to `(B, 512)` for downstream MLPs, or reshape to `(B, 8, 64)` for downstream cross-attention
3. Do NOT pool the 8 queries back to a single vector. Downstream consumers that need full detail (presentation, immunogenicity) see the full `(B, 512)` concat. Scalar readouts (KD, koff, etc.) use their own MLP projections from this.
4. Remove `binding_fuse` (was `nn.Sequential(Linear(512, 256), GELU, Linear(256, 256))` that merged the two binding latents)
5. Remove `binding_query_pool` logic

**Key structural change to latent query loop** (presto.py:1457-1503):
- Currently iterates `CROSS_ATTN_LATENTS` and produces one 256-dim vector per latent
- After change: `pmhc_interaction` produces 512-dim (8x64 flattened). Other latents remain 256-dim.
- Downstream consumers that take `pmhc_interaction` as a dep receive the full 512-dim vector as extra KV tokens (reshaped to 8 tokens of 64-dim, projected to d_model for attention)

#### A3: Vectors flow, scalars are readouts only

**Files to modify:**
- `models/presto.py` — presentation computation (lines 1663-1697), immunogenicity MLP (lines 1518-1528), binding section (lines 1588-1649)
- `models/pmhc.py` — BindingModule, PresentationBottleneck

**Current flow (broken):**
```
binding_affinity (256d) + binding_stability (256d)
  → binding_fuse → 256d
  → BindingModule → 3 scalars (log_koff, log_kon_intrinsic, log_kon_chaperone)
  → derive_kd → scalar KD
  → binding_logit_from_kd → scalar logit
  → PresentationBottleneck(proc_scalar, bind_scalar) → scalar
```

**New flow:**
```
pmhc_interaction queries (8x64 = 512d)
  → interaction_vec (512d vector flows to presentation, immunogenicity)
  → KD readout: MLP(512d) → scalar (supervised by binding data)
  → koff readout: MLP(512d) → scalar (supervised by kinetics data)
  → kon readout: MLP(512d) → scalar (supervised by kinetics data)
  → t_half: ln2/koff (physics derivation, not learned)
  → Tm readout: MLP(512d) → scalar (supervised by stability data)
  → binding_prob: sigmoid(logit_from_KD) (derived scalar)

presentation: MLP(concat(proc_vec, interaction_vec)) → presentation_vec (256d)
  → presentation_prob readout: Linear(256d) → scalar
  → elution: f(presentation_vec, ms_detect_vec) → scalar

immunogenicity: MLP(concat(interaction_vec, recog_vec)) → immuno_vec (256d)
  → immunogenicity_prob readout: Linear(256d) → scalar
```

**Critical change:** The `PresentationBottleneck` class in pmhc.py (lines 870-913) currently has 4 scalar parameters combining two scalar inputs. Replace it with an MLP that takes the full proc_vec + interaction_vec:

```python
# OLD (pmhc.py PresentationBottleneck):
# logit = w_proc * proc_logit + w_bind * bind_logit + bias

# NEW:
self.presentation_mlp = nn.Sequential(
    nn.Linear(256 + 512, 256),  # proc_vec(256) + interaction_vec(512)
    nn.GELU(),
    nn.Linear(256, 256),
)
```

**Physics constraints are preserved as readouts:**
- KD = koff/kon still supervised, but as a readout head from the interaction_vec
- t_half = ln2/koff as a derived quantity
- The constraint `log_KD ≈ log_koff - log_kon + 9` can be added as a soft regularization term

**BindingModule replacement:** Replace the current 3x `Linear(256, 1)` with readout heads from the 512-dim interaction_vec:
```python
self.koff_readout = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
self.kon_readout = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
self.tm_readout = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Linear(128, 1))
# KD derived from koff/kon: log_KD = log_koff - log_kon + 9
```

#### A4: Eliminate class-paired latents

**Files to modify:**
- `models/presto.py` — remove all `*_class1`/`*_class2`/`*_cd8`/`*_cd4` variants, remove class-weighted blending (lines 1530-1546), update output keys

**Current state:**
- Every concept is doubled: `processing_class1`/`processing_class2`, `presentation_class1`/`presentation_class2`, `recognition_cd8`/`recognition_cd4`, `immunogenicity_cd8`/`immunogenicity_cd4`
- At the end, results are blended: `output = class_probs[:, :1] * X_class1 + class_probs[:, 1:2] * X_class2` (line 1530-1546)
- This means each latent only trains on data from its class (Class I data doesn't help Class II processing latent, even though proteasomal processing is shared)

**Implementation:**
1. Single `processing` latent (was `processing_class1` + `processing_class2`)
2. Single `presentation` latent (was `presentation_class1` + `presentation_class2`)
3. Single `recognition` latent with class-specific readout heads:
   ```python
   self.recognition_cd8_head = nn.Linear(d_model, 1)
   self.recognition_cd4_head = nn.Linear(d_model, 1)
   # recognition_prob = class_probs[:,:1] * sigmoid(cd8_head(recog_vec)) + class_probs[:,1:2] * sigmoid(cd4_head(recog_vec))
   ```
4. Single `immunogenicity` latent with class-specific readout heads (same pattern)
5. MHC class information flows through `apc_cell_type_context` token (see C1), NOT through duplicated latents
6. Remove all `*_mixed` output keys — there's only one version of each now

**Class conditioning:** The `apc_cell_type_context` token already carries class_probs. After A4, this becomes the sole mechanism for class conditioning. The latent learns to use this context to produce class-appropriate representations.

**Output key migration:** Many downstream consumers (loss computation, inference) reference keys like `processing_class1_logit`, `binding_class1_prob`, etc. These all collapse to single keys:
- `processing_logit` (was: class-weighted blend of class1/class2)
- `presentation_logit` (was: class-weighted blend)
- `recognition_logit` (was: class-weighted blend; keep cd8/cd4 readouts for when class is known)
- `immunogenicity_logit` (was: class-weighted blend; keep cd8/cd4 readouts)

**Files that consume these output keys** (must be updated):
- `scripts/train_iedb.py` — loss computation
- `scripts/train_synthetic.py` — loss computation
- `scripts/train_modal.py` — if used
- `inference/predictor.py` — prediction extraction
- `tests/test_presto.py` — output shape assertions

---

### Phase 2: Core-Binding Coupling and PFR

Item: **B1** — depends on Phase 1 (interaction latent must exist)

This replaces the sequential "predict core → inject positions → predict binding" with joint enumeration of all candidate binding cores.

#### B1: Coupled Core-Binding with PFR

**Files to modify:**
- `models/presto.py` — core prediction section (lines 1320-1393), latent query section (binding path)
- New module or significant expansion of existing binding path

**Current state (presto.py:1320-1393):**
1. `core_start_head` predicts a probability distribution over peptide positions using mean-pooled `mhc_pair_vec` (losing groove info)
2. `core_width_head` predicts width (8-10) from `mhc_pair_vec`
3. Soft core membership computed, core-relative positions injected into `h_coreaware`
4. Binding latent then cross-attends to `h_coreaware` — sees core-relative positions but the core is already fixed

**Problem:** Core IS the binding event. Different placements put different residues in different groove pockets. The model can't explore alternative cores.

**New design:**

For peptide of length L, core width W=9, candidate core starts k=0..L-W:

```
peptide:  [----N-PFR----][---CORE---][----C-PFR----]
positions: 0..........k  k......k+W  k+W..........L

For each candidate k:
  core_repr_k  = binding_cross_attn(h_peptide[k:k+W], h_mhc, with core-relative positions)
  npfr_repr_k  = mean_pool(h_peptide[0:k]) if k > 0 else zero_vec
  cpfr_repr_k  = mean_pool(h_peptide[k+W:L]) if k+W < L else zero_vec
  npfr_len_k   = pfr_length_embed(k)
  cpfr_len_k   = pfr_length_embed(L - k - W)

  interaction_k = MLP(concat(core_repr_k, npfr_repr_k, npfr_len_k, cpfr_repr_k, cpfr_len_k))
  score_k       = readout(interaction_k) -> scalar

Marginalize: binding_score = logsumexp(score_k + log_prior_k) over all k
```

**Implementation details:**

1. **Encoder runs once.** All candidates reuse the same encoder hidden states `h`.

2. **Core width W=9 fixed.** Both Class I (8-11mer) and Class II (12-50mer) use W=9. For 8-mers, pad to 9 with a special token or handle as special case (1 candidate with 1 empty PFR position).

3. **Candidate enumeration** operates along a new dimension:
   ```python
   # h_pep: (B, L, D) — peptide hidden states
   # For each candidate k in 0..L-W:
   #   core_tokens = h_pep[:, k:k+W, :]  — (B, W, D)
   #   npfr_tokens = h_pep[:, :k, :]     — (B, k, D)
   #   cpfr_tokens = h_pep[:, k+W:, :]   — (B, L-k-W, D)

   # Batch all candidates: reshape to (B*n_candidates, W, D) for core cross-attention
   # This is efficient: dominant Class I case has 1-3 candidates
   ```

4. **PFR representation:**
   - `npfr_repr`: mean-pool of N-terminal PFR residues (zero vector if empty)
   - `cpfr_repr`: mean-pool of C-terminal PFR residues (zero vector if empty)
   - `npfr_len_embed`: learned embedding from PFR length (0 to 49), dim 32
   - `cpfr_len_embed`: learned embedding from PFR length (0 to 49), dim 32
   - PFR residues drape over Class II groove edges — explicit representation lets model learn PFR-MHC contacts

5. **Core-relative position encoding:**
   - Within each candidate's core, positions 0..W-1 get core-relative position embeddings
   - These are added to the core tokens before cross-attention with MHC
   - Replaces the current global core-relative position injection into `h_coreaware`

6. **Marginalization:**
   ```python
   # log_prior_k: uniform or learned prior over start positions
   # score_k: binding score for candidate k
   # binding_score = logsumexp(score_k + log_prior_k, dim=candidates)
   ```

7. **Interaction vector for downstream:**
   - The marginalized interaction vector (soft mixture over candidates) flows to presentation and immunogenicity
   - Alternative: use the best candidate's vector (argmax). Soft mixture is differentiable.

8. **Compute budget:**
   - 8-mer: 1 candidate (if W=8) or special handling
   - 9-mer Class I: 1 candidate
   - 11-mer Class I: 3 candidates
   - 15-mer Class II: 7 candidates
   - 25-mer Class II: 17 candidates
   - 50-mer: 42 candidates (max)
   - Each candidate: 2 lightweight cross-attention layers (interaction latent queries × core+MHC tokens)
   - Worst case: batch 512 × 42 candidates = 21K instances through 2 attn layers — feasible

9. **nflank/cflank vs PFR distinction:**
   - `nflank`/`cflank` = source protein flanking regions (context for proteasomal cleavage). These are SEPARATE tokens in the encoder stream, used by `processing` latent.
   - PFR (N-PFR, C-PFR) = portions of the PEPTIDE itself that flank the binding core. These are peptide residues outside the core.
   - Both concepts are preserved. They serve different biological functions.

10. **Remove old core prediction:** Delete `core_start_head`, `core_width_head`, `core_rel_pos`, the `h_coreaware` construction, and the soft core membership computation. The core is now implicit in the candidate enumeration.

**Validation:** For a known binder (e.g., SIINFEKL/H-2Kb), check that the model's posterior over core positions concentrates on the correct core. Compare per-candidate binding scores.

---

### Phase 3: Context and Groove Vectors

Items: **C1**, **C2**, **C3** — depend on Phase 1 (new latent names)

#### C1: Rename context_token to apc_cell_type_context

**Files to modify:**
- `models/presto.py` — `context_token_proj` (line 448), `_gets_context` set (line 1435), variable names

**Current state:** `context_token_proj` takes `[class_probs(2), mhc_a_species(~5), mhc_b_species(~5), chain_compat(1)]` → 256-dim. Goes to processing, binding, presentation latents.

**Implementation:** Rename only. Contents stay the same. This token tells latents "Class I human" vs "Class II mouse" but NOT which allele — by design, since processing doesn't depend on allele.

The token already uses ground truth class/species when provided via `mhc_class` parameter (presto.py:1282-1296) and falls back to inferred probs only when `mhc_class=None`.

#### C2: Add groove_vec context token

**Files to modify:**
- `models/presto.py` — new module definition, new usage in latent query loop

**Purpose:** Give the binding query an allele-specific fingerprint hint BEFORE it cross-attends to the full MHC sequence. Currently all same-class same-species alleles produce identical context tokens.

**Implementation:**
1. Define a learned cross-attention module that computes a groove summary:
   ```python
   self.groove_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.1, batch_first=True)
   self.groove_query = nn.Parameter(torch.randn(1, d_model) * (1.0 / math.sqrt(d_model)))
   ```

2. Class-conditional masking:
   - Class I groove = alpha chain positions ~1-180 (alpha1 + alpha2 domains). Ignore beta-2-microglobulin.
   - Class II groove = alpha1 (~1-90) + beta1 (~1-90) from both chains.
   - Use positional masking based on segment offsets and known domain boundaries.
   - Implementation: `groove_mask = seg_masks["mhc_a"]` with position filtering for groove region. For Class II, also include early `mhc_b` positions.

3. Compute: `groove_vec = groove_attn(groove_query, h_mhc_groove, h_mhc_groove)` → (B, 1, D)

4. Add `groove_vec` as extra KV token to `pmhc_interaction` and `presentation` latents only (NOT processing — processing doesn't depend on allele)

5. The `_gets_context` and `_gets_groove` sets control which latents receive which context tokens:
   ```python
   _gets_apc_context = {"processing", "pmhc_interaction", "presentation"}
   _gets_groove = {"pmhc_interaction", "presentation"}
   ```

#### C3: Include MHC vecs in pmhc_vec

**Files to modify:**
- `models/presto.py` — `pmhc_vec_proj` definition and usage (lines 1562-1567)

**Current state:**
```python
pmhc_vec = self.pmhc_vec_proj(torch.cat([
    latent_vals["binding_affinity"],
    latent_vals["presentation_class1"],
    latent_vals["presentation_class2"],
], dim=-1))
```
- Excludes `mhc_a_vec`/`mhc_b_vec` (rich 256-dim allele representations)
- `pep_vec` is computed (line 1220) but never used — dead code

**New implementation:**
```python
pmhc_vec = self.pmhc_vec_proj(torch.cat([
    interaction_vec,         # 512-dim interaction latent (was binding_affinity)
    presentation_vec,        # 256-dim (single presentation latent)
    mhc_a_vec,              # 256-dim mean-pooled MHC alpha encoder states
    mhc_b_vec,              # 256-dim mean-pooled MHC beta encoder states
], dim=-1))
# pmhc_vec_proj input: 512 + 256 + 256 + 256 = 1280 → 256
```

This gives TCR matching (which uses `pmhc_vec`) direct access to allele identity. Also serves as a skip connection that shortens gradient paths from downstream losses to MHC encoder states (4 steps vs 20+).

**Also:** Remove dead `pep_vec` computation (line 1220) or use it in `pmhc_vec` if useful for TCR matching. Decision: include it — TCR matching benefits from seeing the peptide representation.

---

### Phase 4: Data Pipeline Fixes

Items: **D2**, **D3** — D1 was in Phase 0

#### D2: Split MIL Bags by MHC Class

**Files to modify:**
- `data/loaders.py` — elution record loading (~lines 1517-1561)
- `data/collate.py` — MIL bag construction (~lines 1042-1099)

**Current state:** All alleles from an ElutionRecord go in one MIL bag regardless of class. If an APC has both Class I and Class II alleles, they share one Noisy-OR — biologically wrong (peptides enter one pathway).

**Implementation:**
1. When creating MIL bags from elution records, partition alleles by class using mhcgnomes
2. If all alleles are same class: one bag (no change)
3. If mixed classes: create separate bags, one per class. Each bag gets only alleles of that class.
4. Both bags share the same peptide and label

#### D3: MIL Over Pathways for Ambiguous T-Cell Assays

**Files to modify:**
- `data/loaders.py` — T-cell record loading
- `data/collate.py` — bag construction

**When to apply:** All three conditions must hold:
1. `mhc_class is None` (unknown after D1 fix + mhcgnomes inference)
2. Peptide length is ambiguous (11-15mer — could be Class I or Class II)
3. Assay doesn't distinguish T-cell subsets (ELISpot/proliferation on bulk PBMCs, NOT ICS with CD4/CD8 gating, NOT multimer)

**Implementation:**
- Create a MIL bag with pathway instances:
  - (peptide, Class I alleles, apc_context="I") for cross-presentation pathway
  - (peptide, Class II alleles, apc_context="II") for direct presentation pathway
- Bag label = T-cell response. Noisy-OR: "at least one pathway produced this response."
- When class IS known, use it directly — no bag needed.

---

### Phase 5: Loss and Training Fixes

Items: **E1**, **E2** — depend on Phase 1 (new presentation architecture)

#### E1: Contrastive MIL Loss

**Files to modify:**
- `scripts/train_iedb.py` (or wherever loss is computed for elution data)
- New utility for contrastive allele selection

**Purpose:** Fix MIL gradient dilution. For each positive elution bag, create a contrastive negative by substituting a different cell's MHC genotype.

**Implementation:**
1. **Contrastive negative creation:**
   - For each positive MIL bag (peptide + allele set from cell A), find a dissimilar allele set from cell B
   - Replace MHC amino acid TOKEN SEQUENCES (not allele names) with those from cell B
   - Same peptide, different MHC sequences
   - Run forward pass on both original and contrastive bags

2. **Contrastive allele selection** (use `mhcgnomes` for all parsing):
   ```python
   import mhcgnomes

   def is_sufficiently_different(original_alleles, candidate_alleles):
       """Check that candidate alleles are sufficiently different from originals."""
       for orig in original_alleles:
           orig_parsed = mhcgnomes.parse(orig)
           for cand in candidate_alleles:
               cand_parsed = mhcgnomes.parse(cand)
               # Cross-species: always valid
               if orig_parsed.species != cand_parsed.species:
                   continue  # fine
               # Same species: require different gene+group
               # e.g., A*02:01 vs A*24:02 is good (different group)
               #        A*02:01 vs A*02:05 is bad (same group)
               if (hasattr(orig_parsed, 'gene') and hasattr(cand_parsed, 'gene')
                   and orig_parsed.gene == cand_parsed.gene
                   and orig_parsed.allele_family == cand_parsed.allele_family):
                   return False
       return True
   ```

3. **Loss:**
   ```python
   # margin loss: original should score higher than contrastive
   margin = 1.0
   loss_contrastive = F.relu(contrastive_pres_score - original_pres_score + margin).mean()
   ```

4. **Optional:** Also check <90% sequence identity between original and substitute MHC-alpha chains for extra safety.

#### E2: MIL Bag Sparsity Regularizer

**Files to modify:**
- Loss computation in training scripts

**Implementation:**
```python
# After computing per-instance probabilities p_i for a bag:
bag_sum = p_instances.sum(dim=-1)  # sum of instance probs
sparsity_loss = F.softplus(bag_sum - 1.5)  # penalty when sum > 1.5
loss += lambda_sparsity * sparsity_loss.mean()
```

Biology: usually 1 allele presents a given peptide, sometimes 2, rarely 3+. This directly encodes the biological prior.

---

## Dependency Graph

```
Phase 0 (independent, do first):
  E3 (init) ─────────────────────────────── can validate immediately
  E4 (LR scheduler) ────────────────────── can validate immediately
  D1 (mhc_class default) ───────────────── can validate with data loading tests

Phase 1 (core architecture, biggest change):
  A1 + A2 + A3 + A4 ────────────────────── all tightly coupled, implement together
    depends on: nothing (but E3 init should be done first so new params get good init)

Phase 2 (core-binding coupling):
  B1 ────────────────────────────────────── depends on A1-A3 (interaction latent must exist)

Phase 3 (context vectors):
  C1 + C2 + C3 ─────────────────────────── depends on A1-A4 (new latent names, new DAG)
    C1 is just a rename (trivial)
    C2 requires groove masking logic (moderate)
    C3 requires updating pmhc_vec_proj (easy once A1-A4 done)

Phase 4 (data pipeline):
  D2 ────────────────────────────────────── independent of model changes (data layer)
  D3 ────────────────────────────────────── independent of model changes, depends on D1

Phase 5 (loss fixes):
  E1 (contrastive MIL) ─────────────────── depends on Phase 1 (new presentation scores)
  E2 (sparsity reg) ────────────────────── independent of architecture (just a loss term)
```

## Testing Strategy

### Unit Tests

Each phase should have passing tests before moving to the next:

1. **Phase 0:** Run existing test suite (`tests/test_presto.py`, `tests/test_loaders.py`). Check init norms. Check scheduler shape.

2. **Phase 1:** Update `tests/test_presto.py` for new output keys (no more `*_class1`/`*_class2` pairs). Verify:
   - Forward pass produces all expected outputs
   - `interaction_vec` is 512-dim
   - Scalar readouts (KD, koff, etc.) produce correct shapes
   - Presentation receives full vectors, not scalars
   - Gradient flows from presentation loss to MHC tokens (use `torch.autograd.grad`)

3. **Phase 2:** Test core enumeration:
   - 9-mer produces 1 candidate, 11-mer produces 3, 15-mer produces 7
   - Marginalized score is differentiable
   - PFR representations have correct shapes for each candidate

4. **Phase 3:** Test groove_vec computation, verify pmhc_vec includes MHC vecs

5. **Phase 4:** Test that elution records with mixed-class alleles produce separate bags. Test MIL pathway creation for ambiguous T-cell records.

6. **Phase 5:** Test contrastive loss computation with mock data.

### Integration / Smoke Test

After all phases:
- Full training loop for 100 steps on a small data subset
- Verify loss decreases
- Verify that binding predictions for different alleles (e.g., HLA-A*02:01 vs HLA-B*07:02) produce meaningfully different scores for the same peptide
- Compare allele discrimination metrics before/after refactor

---

## Key Files Summary

| File | Role | Major Changes |
|------|------|---------------|
| `models/presto.py` | Main model, forward pass, latent DAG | Phases 1-3: new latents, new DAG, core enumeration, groove vec, pmhc_vec |
| `models/pmhc.py` | BindingModule, PresentationBottleneck, Noisy-OR | Phase 1: replace BindingModule with readout heads, replace PresentationBottleneck with MLP |
| `models/heads.py` | AssayHead (KD/IC50/EC50 derivation) | Phase 1: update to work with new interaction_vec |
| `data/loaders.py` | Record dataclasses, data loading | Phase 0: D1 mhc_class default. Phase 4: D2/D3 MIL bags |
| `data/collate.py` | Batch construction, MIL bags | Phase 4: D2/D3 bag splitting |
| `scripts/train_iedb.py` | Training loop, loss, optimizer | Phase 0: E4 scheduler. Phase 5: E1/E2 contrastive + sparsity loss |
| `scripts/train_synthetic.py` | Synthetic training loop | Phase 0: E4 scheduler. Phase 1+: output key updates |
| `scripts/train_modal.py` | Modal cloud training | Same as train_iedb.py |
| `inference/predictor.py` | Prediction extraction | Phase 1: output key updates |
| `tests/test_presto.py` | Model tests | All phases: output shape assertions |
| `tests/test_loaders.py` | Data loading tests | Phase 0+4: mhc_class, MIL bags |
| `tests/test_collate.py` | Collation tests | Phase 4: bag splitting |

---

## Important Conventions

1. **Use mhcgnomes for ALL allele parsing.** Do not write new regexes for MHC allele name parsing. The existing `_infer_mhc_class()` regex in loaders.py can remain as fallback, but prefer mhcgnomes.

2. **nflank/cflank vs PFR are distinct concepts.** nflank/cflank = source protein flanking regions (cleavage context). PFR = peptide flanking regions (parts of the peptide outside the binding core). Never conflate them.

3. **Output key naming:** After A4, drop the `*_class1`/`*_class2`/`*_cd8`/`*_cd4` suffixes from output keys. Use `processing_logit` not `processing_class1_logit`. Keep cd8/cd4 readout keys only for recognition and immunogenicity where class-specific heads are warranted.

4. **Interaction vector dimensionality:** 8 queries × 64 dims = 512 total. This is the "interaction_vec" that flows downstream. It's flattened for MLP consumers but can be reshaped to (8, 64) for cross-attention consumers.

5. **Physics constraints are readouts, not bottlenecks.** KD = koff/kon is supervised via readout heads. The full interaction_vec flows to downstream consumers (presentation, immunogenicity) independently of the scalar KD.
