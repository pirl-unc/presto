# Training Failure Analysis (2026-03-06)

Deep analysis of why the model learns poorly and pMHC binding predictions are similar regardless of allele.

## Root Cause Ranking

| # | Issue | Impact | Status |
|---|-------|--------|--------|
| 1 | Single-query binding latent compresses all MHC info into 256 dims | Allele signal crushed at entry point | TODO |
| 2 | Presentation latents are biologically-motivated bottleneck but receive collapsed input | Allele collapse propagates to all downstream tasks | TODO |
| 3 | `pmhc_vec` excludes `mhc_a_vec`/`mhc_b_vec`; `pep_vec` computed but unused | TCR matching and downstream heads have no direct allele signal | TODO |
| 4 | Context token encodes class+species, not allele identity | All same-class alleles look identical to latent queries | TODO |
| 5 | Scalar compression (KD -> binding_logit) before presentation | 256-dim allele info reduced to 1 float too early | TODO |
| 6 | MS/elution MIL loss satisfied by "mildly positive for all alleles" | Weak allele discrimination signal, 15:1 volume vs binding data | TODO |
| 7 | No allele embeddings, no contrastive loss, no allele-stratified batching | No explicit pressure to maintain allele discrimination | TODO |
| 8 | Long gradient paths (~30 steps) from downstream loss to MHC tokens | Vanishing gradients for allele-specific signal | TODO |
| 9 | Weak latent query initialization (std=0.02) | Uniform attention early in training, slow specialization | TODO |
| 10 | No LR scheduler (fixed 2.8e-4) | Suboptimal convergence dynamics | TODO |

---

## Detailed Analysis Per Issue

### #1: Binding Latent Information Bottleneck

**Location:** `presto.py:1478-1501`, `presto.py:233` (`binding_n_queries=1`)

Only 2 of 12 latents see MHC tokens directly:
- `binding_affinity`: `["peptide", "mhc_a", "mhc_b"]`
- `binding_stability`: `["peptide", "mhc_a", "mhc_b"]`

Each uses 1 query token (256-dim) cross-attending to ~500 tokens (peptide + mhc_a + mhc_b). With 8 attention heads, each head has only 32 dims to capture allele-specific patterns.

Multi-query (Variant B) is implemented but disabled by default. Even when enabled, queries get pooled back to 256-dim.

**Fix direction:** Keep multi-query outputs as a small sequence (e.g. 4x256) and let downstream latents cross-attend to them rather than receiving a single pooled vector.

### #2: Presentation Bottleneck (Biologically Correct, Upstream Problem)

**Location:** `presto.py:190-191` (LATENT_SEGMENTS = []), `pmhc.py:870-913`

Presentation latents have NO cross-attention to tokens. They see only upstream latent deps (processing, binding_affinity, binding_stability) and the context token.

The `PresentationBottleneck` combines processing_logit + binding_logit (both scalars) with 4 learned parameters. This is biologically sound (presentation = processing AND binding).

**Root problem is upstream:** If binding latents collapse, presentation inherits the collapse.

**Fix direction:** Keep biological prior. Make presentation receive full 256-dim binding vectors, not scalar KD. "All latents are vectors, only observables are scalars." The PresentationBottleneck would become: `MLP(concat(proc_vec, bind_aff_vec, bind_stab_vec)) -> scalar logit`.

### #3: Vec Architecture (pmhc_vec vs mhc_a_vec vs mhc_b_vec)

**Locations:**
- `mhc_a_vec`, `mhc_b_vec` (line 1221-1222): mean-pooled encoder hidden states, rich allele-specific representations
- `pmhc_vec` (line 1562-1566): projection of `[binding_affinity, presentation_class1, presentation_class2]`
- `pep_vec` (line 1220): computed but never used (dead code)

`mhc_a_vec`/`mhc_b_vec` are used for auxiliary heads (chain type, species, compat, core positioning) but **never routed to binding, presentation, or pmhc_vec**.

**Fix direction:** Include `mhc_a_vec`/`mhc_b_vec` in `pmhc_vec` projection. Also consider routing them as skip connections to presentation heads.

### #4: Context Token

**Location:** `presto.py:1426-1431`, `presto.py:1435-1439`

A synthetic KV token appended to cross-attention for: processing, binding, presentation latents.

Input: `[class_probs(2), mhc_a_species(~5), mhc_b_species(~5), chain_compat(1)]` -> MLP -> 256-dim.

Tells latents "Class I human" vs "Class II mouse" but NOT which allele. All same-class same-species alleles produce identical context tokens.

Biologically appropriate for processing (doesn't depend on allele). But binding also receives this same coarse context.

**Fix direction:** Could add an "allele fingerprint" derived from mhc_a_vec/mhc_b_vec as a second context token for binding latents only.

### #5: Scalar Compression Too Early

**Location:** `pmhc.py:384-467` (BindingModule: 3x nn.Linear(256,1) = 771 params total)

The BindingModule takes a rich 256-dim binding vector and immediately produces 3 scalars (koff, kon_intrinsic, kon_chaperone). All allele-specific capacity in the vector is gone.

The KD derivation (KD = koff/kon) is an excellent inductive bias. The problem is that it's positioned as a bottleneck in the computation graph rather than a readout.

**Fix direction:** Keep kinetic derivation as a supervised readout/observable. Let the full 256-dim binding vectors flow to presentation and other downstream consumers. Scalar KD is supervised by binding data; presentation also gets the full vector.

### #6: MS/Elution MIL Gradient Dilution

MIL structure is actually correct: each instance in a bag gets a different allele from the cell's genotype. Noisy-OR correctly models "at least one allele presents this peptide."

The problem: MIL loss is satisfied when at least one instance scores mildly positive. Gradient for each instance: `dL/dp_i = (1-p_bag)^{-1} * prod_{j!=i}(1-p_j)`. If multiple instances have moderate p, each gradient is weak. Model learns "peptide is generically presentable" rather than allele-specific binding.

With 3.8M elution samples vs ~250K binding measurements (15:1 ratio), this diluted signal dominates.

**Fix direction:** Contrastive MIL loss -- for each positive bag, create a negative by swapping in a random allele set from a different cell. Forces model to discriminate alleles, not just predict generic presentability.

### #7: No Allele Discrimination Pressure

No per-allele learned embeddings (wouldn't generalize anyway). No contrastive loss. No allele-stratified batch sampling.

**Fix directions:**
- Use `mhc_a_vec` (already computed, generalizes to any sequence) as allele representation
- Contrastive loss: allele-shuffled negatives for MS/elution data
- Allele pseudo-sequences (NetMHCpan-style groove fingerprint, ~34 residues) as explicit features
- Allele-stratified batch sampling to ensure diverse alleles per batch

### #8: Gradient Path Length

Longest path (immunogenicity loss -> MHC token embeddings): ~30 steps through:
immunogenicity_head -> immunogenicity_mlp -> binding_affinity latent -> 2x cross-attention layers -> h_coreaware -> 4x transformer layers -> embeddings

Each step has LayerNorm, GELU, dropout(0.1).

**Fix direction:**
- Skip connections from mhc_a_vec/mhc_b_vec directly into pmhc_vec (4-step path)
- Presentation probe: `probe(mhc_a_vec, pep_vec) -> scalar` with direct loss (like existing binding_affinity_probe)
- Include mhc_a_vec in presentation computation (6-step path vs 20+)

### #9: Latent Query Initialization

**Location:** `presto.py:352-354`

`torch.randn(d_model) * 0.02` produces values in [-0.06, 0.06]. Q*K dot products ≈ 0, so attention weights ≈ uniform (1/seq_len for all tokens). Binding query doesn't know to attend to groove positions.

**Fix direction (simplest):** Xavier-scale initialization: `torch.randn(d_model) * (1.0 / sqrt(d_model))` ≈ 0.0625 std. Gives non-trivial attention patterns from step 1.

Other options: orthogonal init across queries, semantic init from groove residues.

### #10: No LR Scheduler

**Location:** `train_iedb.py` (no scheduler anywhere)

Fixed LR 2.8e-4 throughout training.

**Fix direction:** Cosine annealing with linear warmup.
- warmup: linear 0 -> lr_max over first 5-10% of steps
- decay: cosine lr_max -> lr_min (0.1 * lr_max) over remaining steps
- Warmup especially important because latent queries start near-zero (#9)

```python
scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=2.8e-5)
# + LinearLR warmup via SequentialLR
```

---

## MHC Access Map

### Direct MHC token access (cross-attention):
| Latent | Segments |
|--------|----------|
| binding_affinity | ["peptide", "mhc_a", "mhc_b"] |
| binding_stability | ["peptide", "mhc_a", "mhc_b"] |

### Direct MHC vec access (auxiliary heads only):
- mhc_a_type_head, mhc_b_type_head (chain type classification)
- mhc_a_species_head, mhc_b_species_head (species classification)
- chain_compat_head (compatibility)
- core_start_head, core_width_head (core positioning)
- **None of these feed into binding or presentation prediction**

### Indirect MHC access (via latent DAG deps):
| Latent | Deps (MHC info source) |
|--------|----------------------|
| presentation_class1 | [processing_class1, binding_affinity, binding_stability] |
| presentation_class2 | [processing_class2, binding_affinity, binding_stability] |
| immunogenicity_cd8 | MLP of [binding_affinity, binding_stability, recognition_cd8] |
| immunogenicity_cd4 | MLP of [binding_affinity, binding_stability, recognition_cd4] |

### No MHC access:
processing_class1/2, ms_detectability, species_of_origin, recognition_cd8/cd4
