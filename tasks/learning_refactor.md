# Learning Refactor TODO

High-level architectural changes to fix allele-invariant predictions and improve gradient flow.

- Problem analysis: tasks/training_analysis.md
- Implementation plan: tasks/learning_refactor_plan.md

## TODO Items

### Architecture: Latent DAG Redesign

- [ ] A1: Merge binding_affinity + binding_stability into single pmhc_interaction latent. Both currently see identical tokens with no deps — two random inits of the same cross-attention with no reason to specialize. One rich interaction latent replaces both.

- [ ] A2: Multi-query interaction latent: 8 query heads projecting into 64 dimensions (8x64 total). Do NOT pool back to single vector — downstream consumers see the full query set so specialization is preserved.

- [ ] A3: All latents are vectors, only observables are scalar readouts. Physics constraints (KD=koff/kon, t_half=ln2/koff) live in the readout layer, not as bottlenecks in the computation graph. Presentation receives full interaction_vec, not scalar KD.

  Canonical DAG (every latent is a vector, every observable is a scalar readout):

    processing:       [nflank, peptide, cflank] + apc_context -> proc_vec (256-dim)
                        -> processing_prob readout (scalar, sigmoid)

    pmhc_interaction: [core peptide, mhc_a, mhc_b] + apc_context + groove_vec + PFR repr -> interaction_vec (8x64 multi-query)
                        -> KD readout: MLP -> scalar (supervised by binding data)
                        -> koff readout: MLP -> scalar (supervised by kinetics data)
                        -> kon readout: MLP -> scalar (supervised by kinetics data)
                        -> t_half: derived from koff (ln2/koff, physics not learned)
                        -> Tm readout: MLP -> scalar (supervised by stability data)
                        -> binding_prob: sigmoid(logit_from_KD) (derived)
                        -> KD can also be constrained via physics: koff - kon + 9

    presentation:     MLP(proc_vec, interaction_vec) -> presentation_vec (256-dim)
                        -> presentation_prob readout (scalar)
                        -> elution/MS: f(presentation_vec, ms_detect_vec) -> scalar

    recognition:      [peptide] + foreignness_dep -> recog_vec (256-dim)
                        -> recognition_prob readout (scalar)

    immunogenicity:   MLP(interaction_vec, recog_vec) -> immuno_vec (256-dim)
                        -> immunogenicity_prob readout (scalar)

- [ ] A4: Eliminate class-paired latents. Currently every concept is doubled (processing_class1/class2, presentation_class1/class2, recognition_cd8/cd4, immunogenicity_cd8/cd4) and soft-blended by class_probs at scalar level. Replace with one latent per concept, conditioned on MHC class via apc_cell_type_context token. Benefits: half the latent parameters, each latent trains on ALL data, class info flows through vectors not scalar blend coefficients. Recognition may warrant class-specific readout heads (CD8 vs CD4 repertoires are biologically distinct) but a single recog_vec with class-specific readout heads works.

### Architecture: Core-Binding Coupling and PFR

- [ ] B1: Couple core determination with binding. Current design predicts core first (using mean-pooled MHC, losing groove info), then injects core-relative positions into binding. But core IS the binding event — different placements put different residues in different groove pockets. Replace with joint enumeration:

  For peptide of length L, core width W, candidate core start k:

    peptide:  [----N-PFR----][---CORE---][----C-PFR----]
    positions: 0..........k  k......k+W  k+W..........L

    core_repr_k  = binding_cross_attn(h_peptide[k:k+W], h_mhc, with core-relative positions)
    npfr_repr_k  = mean_pool(h_peptide[0:k]) if k > 0 else zero_vec
    cpfr_repr_k  = mean_pool(h_peptide[k+W:L]) if k+W < L else zero_vec
    npfr_len_k   = pfr_length_embed(k)          # 0 to 49
    cpfr_len_k   = pfr_length_embed(L - k - W)  # 0 to 49

    interaction_k = MLP(concat(core_repr_k, npfr_repr_k, npfr_len_k, cpfr_repr_k, cpfr_len_k))
    score_k       = readout(interaction_k) -> scalar

  Marginalize: binding_score = logsumexp(score_k + log_prior_k) over all candidates k.

  Handles 8-50mer inputs directly, no tiling needed:
    8-mer Class I, W=8:  1 candidate, empty PFRs
    9-mer Class I, W=9:  1 candidate, empty PFRs
    11-mer Class I, W=9: 3 candidates, PFRs 0-2 residues
    15-mer Class II, W=9: 7 candidates
    25-mer Class II, W=9: 17 candidates
    50-mer, W=9: 42 candidates (max)

  PFR is biologically important: PFR residues drape over Class II groove edges and stabilize/destabilize the complex. Explicit PFR representation lets the model learn PFR-MHC contacts distinctly from core-groove contacts.

  Compute: encoder runs once. Candidates batched along a new dim. Dominant Class I case (1-3 candidates) is essentially free. Worst case (50-mer, 42 candidates at batch 512) = 21K instances through 2 lightweight attention layers — feasible.

  Note: nflank/cflank segments are SOURCE PROTEIN flanks (cleavage context for processing), a separate biological concept from peptide flanking regions (PFR). Both are preserved.

### Architecture: Context and Groove Vectors

- [ ] C1: Rename context_token to apc_cell_type_context. Contents: class_probs + species_probs + chain_compat. Already uses ground truth class/species when known at training time (batch.mhc_class), inferred probs only as fallback. Goes to processing + binding + presentation latents. Does NOT carry allele-specific info — by design, since processing doesn't depend on allele.

- [ ] C2: Add groove_vec as a distinct context token for binding and presentation latents only (NOT processing). Class-conditional logic: Class I groove = alpha chain positions ~1-180 (alpha1+alpha2 domains), ignore beta2m. Class II groove = alpha1 (~1-90) + beta1 (~1-90) from both chains. Implement as learned cross-attention with class-conditional masking over MHC tokens. Gives binding query an allele fingerprint hint before it cross-attends to the full sequence.

- [ ] C3: Include mhc_a_vec/mhc_b_vec in pmhc_vec projection. Currently pmhc_vec = proj([binding_affinity, pres_class1, pres_class2]) and excludes the rich mean-pooled MHC encoder states. Also: pep_vec is computed (line 1220) but never used — dead code. pmhc_vec feeds TCR matching; without direct MHC vecs it has no allele identity signal.

### Data Pipeline Fixes

- [ ] D1: Fix mhc_class default across ALL record dataclasses. Change from `str = "I"` to `Optional[str] = None` in: BindingRecord, KineticsRecord, StabilityRecord, ProcessingRecord, ElutionRecord, TCellRecord, VDJdbRecord. Default "I" silently mislabels records where class is genuinely unknown. When None, either infer from allele name (using mhcgnomes) or propagate ambiguity.

- [ ] D2: Split MIL bags by MHC class. Currently all alleles from an ElutionRecord go in one bag regardless of class (loaders.py:1524-1536). If an APC has both Class I and Class II alleles, they share one Noisy-OR — biologically wrong since peptides enter ONE processing pathway. Fix: create separate bags per class when record has mixed-class alleles.

- [ ] D3: MIL over pathways for ambiguous T-cell assays. When mhc_class is None AND peptide length is ambiguous (11-15mer) AND assay doesn't distinguish T cell subsets (ELISpot/proliferation on bulk PBMCs, not ICS with CD4/CD8 gating, not multimer): create a MIL bag with pathway instances — (peptide, Class I allele, apc_context="I") for cross-presentation and (peptide, Class II allele, apc_context="II") for direct binding. Bag label = T-cell response. Noisy-OR: "at least one pathway produced this response." When class IS known, use it directly — no bag needed.

### Loss and Training Fixes

- [ ] E1: Contrastive MIL loss for allele discrimination. For each positive elution bag, create a contrastive negative by substituting a different cell's MHC genotype — replace MHC-α and MHC-β amino acid TOKEN SEQUENCES with sequences from dissimilar alleles. Same peptide tokens, different MHC token sequences. Run forward pass on both bags. Margin loss: presentation_score(original) > presentation_score(substitute) + margin.
  Contrastive allele selection (use mhcgnomes for parsing, NOT new regexes):
  - Cross-species always valid (maximally different)
  - Within species: require all gene+group entries to differ (A*02:01 vs A*24:02 good, A*02:01 vs A*02:05 bad)
  - Optionally require <90% sequence identity between any original and substitute MHC-α

- [ ] E2: MIL bag sparsity regularizer. Add penalty: loss += softplus(sum(p_i) - 1.5) per bag. Biology says usually 1, sometimes 2, rarely 3+ alleles present a given peptide. Directly encodes biological prior without changing Noisy-OR aggregation.

- [ ] E3: Fix weight initialization. Currently NO explicit init anywhere — all PyTorch defaults.
  (a) Latent queries: N(0, 0.02) -> N(0, 1/sqrt(d_model)) ≈ N(0, 0.0625). Current is 3-4x too small, causes near-uniform attention.
  (b) nn.Embedding: N(0,1) -> N(0, 1/sqrt(d_model)). Default gives L2 norm ≈ 16, too large vs transformer working scale.
  (c) nn.MultiheadAttention: Kaiming -> Xavier for Q/K/V projections (Transformer convention).
  (d) nn.Linear in heads: Kaiming default is fine, no change.

- [ ] E4: Add LR scheduler. Cosine annealing with linear warmup:
  - warmup: linear 0 -> lr_max over first 5-10% of steps
  - decay: cosine lr_max -> lr_min (0.1 * lr_max = 2.8e-5) over remaining steps
  - PyTorch: compose LinearLR + CosineAnnealingLR via SequentialLR
