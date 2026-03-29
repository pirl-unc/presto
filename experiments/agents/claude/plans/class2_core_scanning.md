# Class II Core Scanning via Class I Transfer

Detailed design plan for reusing the class I binding core mechanism as a scanning module for class II peptides.

## The key insight

Presto's binding core mechanism is already a scanner. For every peptide, it:

1. **Enumerates** all valid (start_position, core_length) candidates (presto.py:1590-1628)
2. **Extracts** core tokens + N/C peptide flanking regions for each candidate
3. **Scores** each candidate via learned MLP + structural prior (presto.py:1681-1783)
4. **Marginalizes** over candidates using softmax posterior: `interaction_vec = Σ posterior[i] × candidate_vec[i]`

This is fully differentiable. No hard core labels needed — the model learns core selection end-to-end from binding affinity supervision.

For class I peptides (8-11mers), this is a near-trivial scan: an 9-mer peptide has exactly one 9-mer core candidate (the whole peptide). A 10-mer has two 9-mer candidates. The posterior concentrates quickly because there are few options.

For class II peptides (13-25mers), the same mechanism becomes a real scanner: a 15-mer has seven 9-mer candidates, each with different peptide flanking regions. The posterior must learn to select the correct core — the one whose residues actually contact the groove.

**No new architecture needed.** The plan is to train this existing mechanism correctly through curriculum staging.

## What already exists

| Component | Location | Status |
|-----------|----------|--------|
| Core enumeration | presto.py:1590-1628 | Working, handles any peptide length |
| Core token extraction | presto.py:1629-1680 | Working, extracts core + N/C flanking |
| Core scoring MLP (shared) | presto.py:1754 | Working, `core_window_score` |
| Core scoring MLP (class-specific) | presto.py:498-507 | Exists, untrained for class II |
| Structural prior | presto.py:1737-1745 | Working, inputs: core_len_frac, flank_fracs, class_probs |
| Soft marginalization | presto.py:1760-1783 | Working, softmax + weighted sum |
| Class conditioning | presto_modules.py:431-443 | Working, class-dependent binding calibration |
| Groove vector (cross-attention) | presto.py:1533-1569 | Working, allele-specific |
| BindingModule (kinetics) | pmhc.py:385-437 | Working, class-agnostic |
| Two-chain input (α+β) | data/loaders.py, groove.py | Working for both class I and II |

## What class II adds

### Longer peptides with variable binding core position

Class I: 8-11 residues. The peptide is almost entirely the binding core. Anchors at P2 and PΩ are fixed relative to the peptide termini. The closed groove constrains the core.

Class II: 13-25 residues. A 9-mer binding core sits somewhere in the middle, flanked by peptide extensions (peptide flanking regions, PFRs) that protrude from both ends of the open groove. The core position varies by peptide — it depends on which 9-mer window has the right anchor residues for that allele's groove pockets.

Example: For HLA-DRB1*01:01 with peptide PKYVKQNTLKLAT (13-mer):
- Candidate cores: PKYVKQNTL, KYVKQNTLK, YVKQNTLKL, VKQNTLKLA, KQNTLKLAT
- True core: YVKQNTLKL (positions 2-10, anchors at P1=Y, P4=K, P6=T, P9=L)
- The model must learn: P1 prefers large hydrophobic/aromatic, P9 prefers aliphatic

### Different groove geometry

Class I groove: closed at both ends by conserved tyrosine residues. Constrains peptide to 8-11mers.

Class II groove: open at both ends. The α1 and β1 domains form a groove that's wider and doesn't cap the peptide. This means:
- Groove contacts are spread across 9 core residues, not concentrated at termini
- The N-flank and C-flank of the peptide extend beyond the groove and are solvent-exposed
- PFR residues do NOT contact the groove directly but may influence binding through steric/electrostatic effects

### Two polymorphic chains

Class I: Only the alpha chain is polymorphic. Beta-2-microglobulin (β2m) is invariant.

Class II: Both alpha and beta chains are polymorphic (except DRA, which is nearly monomorphic for humans). The groove is formed by α1 (from alpha chain) + β1 (from beta chain). Both contribute to peptide-allele specificity.

The model already handles this: groove_half_1 comes from the alpha chain, groove_half_2 from the beta chain, regardless of class.

## Training plan

### Phase 1: Class I core training (stages 2a-2c of curriculum)

The core scanner is first trained on class I data where the answer is constrained:

- 8-mer: 4 candidates (8,9,10,11-length cores, but only 8-mer fits → 1 candidate per length that fits)
- 9-mer: ~6 candidates across core lengths 8-11
- 10-mer: ~10 candidates
- 11-mer: ~14 candidates

For most class I peptides, the posterior should concentrate on the full-length core (the peptide IS the core). The scanner learns basic residue-groove compatibility: which amino acids are preferred at which pocket positions.

**What the core scorer learns from class I**:
- Anchor residue preferences per allele (P2 and PΩ for class I)
- Core length preferences (most class I alleles prefer 9-mers)
- That the scoring function should reward hydrophobic residues at certain core positions
- Basic groove-residue compatibility patterns

**What it does NOT learn from class I alone**:
- How to handle long PFRs (class I PFRs are 0-3 residues)
- P1 pocket preferences (class II's primary anchor, less important in class I)
- The different anchor spacing of class II (P1, P4, P6, P9 vs class I's P2, PΩ)
- That class II alleles have much more variable core-length preferences

### Phase 2: Class II core transfer (stage 4a of curriculum)

When class II data is introduced, the core scanner faces a harder problem:

- 13-mer: ~21 candidates (many windows, most wrong)
- 15-mer: ~29 candidates
- 20-mer: ~46 candidates

The class I-trained scorer provides a starting point: it already knows that binding cores should have hydrophobic residues at certain positions. But the class II groove has different pocket preferences, so the scorer must adapt.

**Transfer mechanism** (three options, not mutually exclusive):

#### Option A: Shared scorer with class conditioning (simplest)

Use the existing `core_window_score` shared MLP. It already receives class probabilities as input via the structural prior (presto.py:1737-1745). The class I training teaches it general residue-groove compatibility; when class II data arrives, the class probability input tells the scorer to adjust its preferences.

Advantages:
- No new parameters
- Maximum transfer from class I
- Works immediately when class II data is introduced

Disadvantages:
- The shared MLP has limited capacity to represent both class I and class II pocket preferences
- Class II anchor positions differ from class I (P1,P4,P6,P9 vs P2,PΩ) — a single scorer may struggle to represent both

#### Option B: Class-specific scorers (already exists)

Use `core_window_score_class1` and `core_window_score_class2` (presto.py:498-507). Initialize class2 scorer from class1 weights. The class-weighted combination `class1_weight × class1_score + class2_weight × class2_score` lets the class II scorer diverge from class I while retaining the class I scorer's knowledge.

Advantages:
- Dedicated capacity for class II preferences
- Class I scorer is protected from class II gradient interference
- Initialization from class I weights provides warm start

Disadvantages:
- More parameters
- The class II scorer starts with class I weights that may be misleading (wrong anchor positions)
- Needs enough class II data to train the class II-specific parameters

#### Option C: Shared trunk + class-specific heads (recommended)

Hybrid approach. The core candidate vector (extracted tokens + groove context) is computed identically for both classes. The scoring function has:
1. A shared feature extractor (first linear layer) — captures general residue-groove compatibility
2. Class-specific scoring heads (second linear layer) — captures class-specific anchor preferences

This is architecturally similar to the DAG assay heads: shared base + class-specific residual.

```
candidate_vec → shared_linear → ReLU → class1_head → score_I
                                      → class2_head → score_II
final_score = class_prob_I × score_I + class_prob_II × score_II
```

Initialize both heads from the current shared scorer weights. The shared layer retains general residue-groove knowledge; the class-specific heads learn pocket-specific preferences.

### Phase 3: Core refinement with gold-standard labels (stage 4b)

Some class II peptide-allele pairs have experimentally determined binding cores (from crystal structures, competition assays, or truncation studies). These provide direct supervision for core selection.

**Core supervision loss**: For peptides with known cores, add a cross-entropy loss on the core posterior:

```
L_core = -log(posterior[true_core_index])
```

This is similar to the "attention supervision" technique used in machine translation (Liu et al., 2016) where alignment labels from external tools supervise the attention distribution.

**Important**: This loss applies only to the subset of peptides with known cores. It's an auxiliary loss, not the primary objective. The primary objective remains binding affinity regression — the core selection is a means to an end.

**Data sources for gold-standard cores**:
- PDB crystal structures: ~500 unique class II peptide-allele pairs with resolved binding cores
- IEDB binding core annotations: derived from competition assays, available for ~2000 pairs
- NetMHCIIpan training data: includes binding core annotations from the IEDB reference dataset

### Phase 4: Ongoing core refinement (stages 5-6)

As multi-allelic MS and T-cell data arrive (stages 5-6), the core scanner continues to refine. Presentation and immunogenicity signals provide indirect core supervision: if the model selects the wrong core, its binding prediction will be wrong, the presentation prediction will be wrong, and the loss will be high. This is the end-to-end training signal.

The soft marginalization ensures gradients flow back through the core selection even without explicit core labels.

## Evaluation metrics for core scanning

### Per-candidate posterior entropy

For each peptide, compute the entropy of the core posterior distribution:
```
H = -Σ posterior[i] × log(posterior[i])
```

Low entropy means the model is confident about one core. High entropy means it's uncertain (spreading probability across many windows).

Expected behavior:
- Class I: very low entropy (1-3 plausible cores for short peptides)
- Class II with clear anchors: low entropy (model commits to one core)
- Class II with degenerate anchors: higher entropy (ambiguous core position)

Track entropy over training to confirm that core selection sharpens as training progresses.

### Core accuracy on gold-standard peptides

For peptides with known binding cores:
- **Exact match**: predicted core (argmax of posterior) matches true core exactly
- **±1 match**: predicted core start is within 1 position of true core start
- **Anchor match**: predicted P1 and P9 residues match true P1 and P9

Target: >70% exact match, >90% ±1 match on class II peptides with known cores.

### Binding affinity conditioned on core selection

Compare binding predictions when using:
1. The model's learned core selection (soft marginalization)
2. The known true core (forced selection)
3. A random core window

If the model's learned selection produces binding predictions as accurate as the forced true core, the scanner is working. If a random core is almost as good, the scanner is not contributing.

### Per-allele core length distribution

For each class II allele, histogram the predicted core lengths across all peptides. Compare to known preferences:
- HLA-DR alleles: predominantly 9-mer cores
- HLA-DQ alleles: some prefer 9-mers, some tolerate 10-mers
- HLA-DP alleles: 9-mer cores

If the model predicts the right length distribution per allele, it has learned allele-specific groove geometry.

## Architectural considerations

### Core window enumeration for long peptides

A 25-mer peptide with core_lengths=(8,9,10,11) generates:
- Length 8: 18 candidates
- Length 9: 17 candidates
- Length 10: 16 candidates
- Length 11: 15 candidates
- Total: 66 candidates

This is 3x more than a typical class I peptide. The softmax over 66 candidates requires:
- 66 forward passes through the scoring MLP (can be batched)
- 66 interaction vectors from the cross-attention mechanism

**Memory concern**: Each candidate's cross-attention context includes core tokens + groove tokens + dependency latents. For 66 candidates at d_model=128, this is manageable (~33K floats per sample).

**Compute concern**: The dominant cost is the cross-attention in `_binding_latent_query()`. With 66 candidates, this is 66 attention operations per sample. At batch_size=256, that's ~17K attention operations per batch. This is ~3-6x more compute than class I batches.

**Mitigation options**:
1. **Pre-filter candidates**: Before full scoring, use a cheap filter (e.g., check if P1 is hydrophobic for DR alleles) to prune obviously wrong candidates. Reduces 66 → ~20 candidates.
2. **Shared KV cache**: The MHC groove tokens and dependency latents are the same across all candidates for a given sample. Cache them and only recompute the peptide-core portion of the KV.
3. **Smaller batch size for class II**: If memory is tight, use a smaller effective batch size for class II samples. Mixed batches (class I + class II) can use different per-sample candidate counts.

### Gradient flow through core selection

The soft marginalization `interaction_vec = Σ posterior[i] × candidate_vec[i]` is fully differentiable. Gradients flow through:
- The marginalized interaction_vec (directly)
- The posterior weights (via softmax → scoring MLP → core candidate features)

This means the binding affinity loss automatically teaches core selection: if the model picks the wrong core, the interaction_vec will be wrong, the binding prediction will be wrong, and the loss gradient will push the posterior toward the correct core.

**Potential issue**: Posterior collapse. If one candidate dominates early in training (posterior ≈ 1.0 for one candidate, ≈ 0.0 for others), gradients to the non-selected candidates vanish. This could prevent the model from exploring alternative core positions.

**Mitigation**:
- **Temperature annealing**: Start with a high softmax temperature (τ=2.0, uniform-ish posterior), anneal to τ=1.0 over training. This encourages exploration early on. Similar to the temperature schedule in Gumbel-softmax training (Jang et al., 2017).
- **Entropy regularization**: Add a small bonus for posterior entropy: `L_entropy = -λ × H(posterior)`. Prevents premature collapse. λ should be small (0.01-0.1) to not prevent convergence.
- **Label smoothing on core supervision**: When using gold-standard core labels (stage 4b), use label-smoothed cross-entropy rather than hard cross-entropy. Distribute 10% of probability mass across non-target cores.

### PFR (peptide flanking region) handling

The current implementation extracts N-terminal and C-terminal peptide flanking regions for each core candidate. For class II, these PFRs are the residues extending beyond the groove:

- 15-mer with core at position 3-11: N-PFR = residues 0-2, C-PFR = residues 12-14
- 15-mer with core at position 0-8: N-PFR = empty, C-PFR = residues 9-14

The PFR representation is fused with the core interaction vector before scoring. This means the model can learn that certain PFR compositions favor or disfavor presentation (e.g., long PFRs may sterically hinder loading).

**Important for class II**: PFRs are NOT in contact with the groove — they extend beyond the open ends. But they DO influence:
- Binding kinetics (long PFRs increase on-rate via "flapping" dynamics)
- DM-mediated editing (CLIP peptide displacement depends on PFR length/composition)
- Processing (flanking residues determine protease cleavage sites)

The current PFR fusion (concatenation + projection) is adequate. No class II-specific changes needed.

## Implementation timeline

### Already done (no code changes needed):
- Core enumeration for any peptide length ✓
- Soft marginalization over candidates ✓
- Class-specific scoring MLPs ✓
- Two-chain (α+β) groove input for class II ✓
- Class-conditional binding calibration ✓

### Needed for stage 4a (class II introduction):
1. **Data pipeline**: Add class II affinity data to training. Requires:
   - Remove `--train-mhc-class-filter I` from training command
   - Verify class II groove parsing works for all alleles in mhcseqs
   - Confirm class II peptide length distribution (12-20mers) is correctly handled by collation/padding

2. **Core scorer initialization**: Copy class1 scorer weights to class2 scorer at the start of stage 4a:
   ```python
   model.core_window_score_class2.load_state_dict(
       model.core_window_score_class1.state_dict()
   )
   ```

3. **Temperature schedule**: Add softmax temperature parameter to core scoring:
   ```python
   posterior = F.softmax(logits / temperature, dim=-1)
   ```
   Start at τ=2.0, anneal to τ=1.0 over first 5 epochs of stage 4a.

4. **Entropy monitoring**: Log per-sample core posterior entropy as a training metric. No code change to the model, just add to the training loop's metric collection.

### Needed for stage 4b (core refinement):
1. **Gold-standard core labels**: Load experimentally determined binding cores from IEDB or PDB. Match to training peptides by sequence + allele.

2. **Core supervision loss**: Add auxiliary cross-entropy loss on the core posterior for labeled peptides:
   ```python
   if core_label is not None:
       core_loss = F.cross_entropy(logits, core_label, label_smoothing=0.1)
       losses.append(core_weight * core_loss)
   ```

3. **Evaluation**: Add core accuracy metrics (exact match, ±1 match) to validation.

### Nice-to-have (not required for initial class II support):
- Pre-filtering of candidate cores by anchor residue heuristics
- KV cache sharing across candidates
- Adaptive temperature per allele (some alleles have sharper core preferences)
- Visualization of core posterior as a heatmap over peptide position × core length

## Risk assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Class II gradients damage class I binding | Medium | High | Freeze class I scorer, only train class II scorer. Monitor class I metrics with decision gate. |
| Posterior collapse (one core dominates) | Medium | Medium | Temperature annealing + entropy regularization |
| Insufficient class II data for scorer | Low | Medium | Class I transfer provides strong initialization. Even small class II datasets (~10K measurements) should be enough to adapt the scorer. |
| Memory issues with long peptides (>20-mer) | Low | Low | Max peptide length is 25 in our data. 66 candidates at d=128 is ~33K floats — small relative to batch. |
| Class II groove parsing failures | Low | Medium | mhcseqs provides precomputed grooves for most class II alleles. Fallback groove parsing has been validated. |
| Core length mismatch (class II vs I) | Medium | Low | Both classes use binding_core_lengths=(8,9,10,11). Class II overwhelmingly prefers 9-mers, which is already the dominant class I core length. |

## References

- Reynisson, B. et al. (2020). NetMHCpan-4.1 / NetMHCIIpan-4.0. Covers class II binding core prediction approach.
- Jensen, K. K. et al. (2018). Improved Methods for Predicting Peptide Binding Affinity to MHC Class II Molecules. Describes the core enumeration and best-core selection approach used in NetMHCIIpan.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. Temperature-controlled discrete variable sampling.
- Liu, L. et al. (2016). Agreement on Target-Bidirectional Neural Machine Translation. Attention supervision concept.
- Stern, L. J. et al. (1994). Crystal Structure of the Human Class II MHC Protein HLA-DR1. Defines the class II groove structure and peptide binding geometry.
