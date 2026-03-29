# Curriculum Training Roadmap

Staged training plan for the full Presto multi-task model. Each stage builds on the previous checkpoint, progressively adding data types following the biological causal chain of antigen presentation: sequence structure → binding → processing → presentation → immunogenicity.

## Design principles

### 1. Causal ordering of tasks

The stages follow the real biology: a peptide must (1) bind the MHC groove, (2) survive proteasomal cleavage and TAP transport, (3) be loaded and presented on the cell surface, and (4) be recognized by a T-cell receptor. Training in this order means each new task can build on representations learned for the upstream task, rather than learning everything simultaneously from noisy mixed signal.

This is analogous to the approach in AlphaFold 2 (Jumper et al., 2021), where structure prediction is decomposed into sequential stages (MSA processing → pair representation → structure module → recycling), with each stage refining the output of the previous one. The key insight is the same: decompose a complex prediction into biologically-motivated intermediate representations.

### 2. Persistent loss inventory

Once a loss term is introduced (regression, contrastive, or synthetic), it remains in all subsequent batches at a maintained fraction. This follows the continual learning principle from experience replay (Lin, 1992) — forgetting is prevented not by regularization (EWC) but by continued exposure. We prefer replay over EWC because our tasks have separate output heads and a shared encoder; replay is simpler and empirically sufficient in this regime (Chaudhry et al., 2019, showed even tiny replay buffers prevent forgetting when task heads are distinct).

### 3. Sub-stages for augmentation layering

Each major stage has sub-stages: first train on clean real data, then add contrastive terms, then add synthetic negatives. This isolates the contribution of each augmentation type and provides fallback checkpoints if an augmentation hurts. The pattern is: establish signal → sharpen discrimination → stress-test with adversarial examples.

### 4. Adaptive duration, not fixed epochs

Each sub-stage runs until a validation metric plateaus, not for a fixed number of epochs. This follows standard practice in transfer learning (Howard & Ruder, 2018, "ULMFiT"): monitor validation loss with patience-based early stopping, then advance to the next sub-stage. The exception is stage 1 (pretraining), which uses a fixed epoch budget since its classification tasks converge quickly.

### 5. Loss balancing via uncertainty weighting

When multiple loss terms coexist in a batch, their relative weights are learned using homoscedastic uncertainty weighting (Kendall et al., 2018). Each loss term $L_i$ is weighted by a learned log-variance $\sigma_i$:

$$\mathcal{L}_{total} = \sum_i \frac{1}{2\sigma_i^2} L_i + \log \sigma_i$$

This automatically balances regression losses (continuous, unbounded) against classification losses (binary, bounded) without manual weight tuning. The $\log \sigma_i$ penalty prevents any single task from being downweighted to zero.

The alternative — fixed manual weights — requires tuning ~10 weight hyperparameters across stages. Uncertainty weighting reduces this to zero tunable weights at the cost of ~10 scalar parameters.

---

## Batch composition

Every batch from stage 2 onward is a structured mixture. The allocation follows a priority system:

| Component | Target fraction | Source |
|-----------|----------------|--------|
| Current stage primary data | 50% | New data type being introduced |
| Replay from prior stages | 30% | Reservoir-sampled from all prior real data |
| Synthetic negatives | 15% | Matched to each data type present in the batch |
| (Contrastive pairs) | Computed on-the-fly from all real + replay data in the batch | Not a separate allocation |

### Replay buffer design

We use a **reservoir sampling** buffer (Vitter, 1985) with fixed capacity per prior stage. As new stages are added, the buffer maintains a uniform random sample from each prior stage's training data. This is simpler than priority-based replay (Schaul et al., 2016) and avoids the need to maintain per-example priority scores.

Buffer size per stage: 10,000 examples (enough to cover the diversity of each data type without consuming excessive memory). At stage 6, the replay buffer contains ~50,000 examples across 5 prior stages.

### Synthetic negative lifecycle

Synthetic negatives are generated on-the-fly each epoch (not cached) so the model never memorizes specific negatives. The generation seed advances per epoch to ensure different synthetic examples at each pass. This follows the practice in MHCflurry 2.0, which regenerates presentation decoys at each training iteration.

---

## Stage 1: Pre-training on sequence metadata

**What**: Predict species category and chain type from MHC groove sequences. Predict vertebrate vs pathogen origin from peptide sequences.

**Why**: Initializes the encoder with structural literacy before any binding signal. Similar in spirit to the masked language model pretraining used in ESM (Rives et al., 2021) and ProtTrans (Elnaggar et al., 2022), but with supervised classification instead of masked prediction. We use supervised pretraining because our downstream tasks are supervised and our sequences are short (8-34 AA) — too short for meaningful masked language modeling.

**Data**:
- MHC: All groove sequences from mhcseqs → predict (species, chain_type)
- Peptides: All peptides from merged_deduped.tsv → predict vertebrate_vs_pathogen (derived from source organism annotation)

**Loss**: Cross-entropy for both classification tasks. Balanced sampling if class imbalance > 3:1.

**Synthetics**: None (natural class balance is adequate for classification).

**Contrastive**: None.

**Duration**: Fixed, 5-10 epochs or until classification accuracy > 95% on held-out validation.

**Checkpoint**: `stage1_pretrain.pt`

**Eval**: Species classification accuracy, chain type accuracy, peptide origin AUROC. These are sanity checks — if the encoder can't classify species from groove sequences, something is wrong with the tokenization or input pipeline.

**Open question**: Whether to include a peptide length prediction head (predict peptide length from MHC groove alone, as a proxy for groove geometry). This would teach the encoder that A*02:01's groove prefers 9-mers while B*44:02 tolerates 10-11mers, before seeing any binding data. Biologically motivated but may not provide enough signal to be worth the complexity.

---

## Stage 2: Binding affinity (class I)

### Stage 2a: Clean affinity regression

**What**: Train binding affinity heads (KD, IC50, EC50) on all quantitative binding assay data. Class I only. No augmentation — pure censor-aware regression.

**Why**: Establish the clean binding signal before any augmentation. This is the control condition for all subsequent comparisons. Uses the L2 recipe proven in the 7-allele bakeoff: DAG assay heads (dag_prep_readout_leaf), assay_heads_only loss mode, warmup_cosine schedule.

**Init**: Stage 1 checkpoint. Reset optimizer (cold optimizer, warm model), following the ULMFiT practice (Howard & Ruder, 2018) of resetting learning rate and momentum when switching tasks.

**Data**: merged_deduped.tsv, `numeric_no_qualitative`, all class I alleles, split_kd_proxy, mhcflurry encoding. ~250K rows across ~105 HLA alleles.

**Loss**: Censor-aware regression per assay family (handles `>` and `<` qualifiers correctly). Uncertainty-weighted across KD/IC50/EC50 families.

**Synthetics**: None.

**Contrastive**: None.

**Duration**: Until validation loss plateaus (patience 3-5 epochs). Expected: 10-30 epochs given dataset size.

**Checkpoint**: `stage2a_affinity_clean.pt`

**Eval**: Test Spearman, AUROC (≤500 nM threshold), AUPRC, RMSE (log10 nM), balanced accuracy, F1. SLLQHLIGL probe discrimination (A02 vs A24 Kd ratio and binding probability). Per-allele Spearman to check coverage across rare alleles.

### Stage 2b: Add contrastive ranking

**What**: Add two contrastive ranking losses to the affinity regression.

**Why**: Pointwise regression optimizes absolute value accuracy but doesn't explicitly reward correct *ordering*. Contrastive losses add direct gradient signal for ranking: "peptide X binds allele A tighter than allele B" (cross-allele) and "peptide X binds allele A tighter than peptide Y does" (within-allele). This is closely related to the pairwise ranking approach in LambdaMART (Burges, 2010), adapted to the binding prediction setting.

At 105 alleles, there are ~5,000 allele pairs to discriminate. The regression loss gives indirect ranking signal (get the numbers right → rankings follow), but contrastive gives direct signal for the ranking specifically.

**Init**: Stage 2a checkpoint. Carry optimizer state forward (same task, just adding a loss term).

**Data**: Same affinity data.

**Contrastive losses**:
- `binding_contrastive` (cross-allele): Pairs of (peptide, allele_A, allele_B) where the same peptide has different measured affinities for two alleles. Margin-based ranking loss: `max(0, margin - (pred_strong - pred_weak))`. Margin 0.2 log10 units, target gap clamped to [0.3, 2.0]. Max 64 pairs per batch.
- `binding_peptide_contrastive` (within-allele): Pairs of (allele, peptide_X, peptide_Y) where two peptides have different measured affinities for the same allele. Same loss form. Max 128 pairs per batch.

**Synthetics**: None yet.

**Duration**: Until validation Spearman plateaus (patience 3 epochs). Expected: 3-10 epochs.

**Checkpoint**: `stage2b_affinity_contrastive.pt`

**Eval**: Same as 2a. Compare probe discrimination vs 2a. The 2-allele synth ablation showed contrastive can hurt IC50 calibration — if probe Kd inflates above 100 nM for A02:01 or discrimination ratio drops below 500x, revert to 2a checkpoint and skip 2b.

**Decision gate**: If 2b degrades probe discrimination, skip it and proceed from 2a to 2c (or directly to stage 3). If 2b improves ranking without hurting calibration, keep it.

### Stage 2c: Add synthetic negatives

**What**: Add `peptide_scramble` synthetic non-binders at low ratio.

**Why**: Alleles with few training examples (e.g., rare HLA-C alleles with <100 measurements) may default to predicting moderate affinity for everything because they haven't seen enough non-binders. Synthetic negatives provide explicit "this doesn't bind" signal for data-sparse alleles. This is the same motivation as negative sampling in word2vec (Mikolov et al., 2013) — without negatives, the model only sees positives and learns a degenerate solution.

We use only `peptide_scramble` (shuffle peptide sequence with forced anchor changes). The synth ablation showed `mhc_scramble` produces structurally invalid groove sequences (54/128 fallback parses with non-canonical lengths), so MHC-side synthetics are excluded.

**Init**: Best of 2a or 2b checkpoint.

**Data**: Affinity data + synthetic negatives.

**Synthetics**:
- `peptide_scramble` with anchor-opposite forcing at P2 and PΩ
- Ratio: 0.10 of real data per batch (1 synthetic per 10 real)
- Target affinity: Sampled uniformly in [500, 20000] nM range (weak/non-binder)
- Regenerated each epoch with advancing seed

**Contrastive**: Carried forward from 2b (if 2b was kept), or none (if 2b was skipped).

**Duration**: Until validation loss plateaus (patience 3 epochs). Expected: 3-10 epochs.

**Checkpoint**: `stage2c_affinity_full.pt`

**Eval**: Same as 2a/2b. Critical check: A02:01 probe Kd must stay in 1-50 nM range. If synthetics inflate Kd to 1000+ nM (as seen in the 2-allele ablation), reduce ratio or revert to prior checkpoint.

---

## Stage 3: Mono-allelic mass spec (class I)

### Stage 3a: MS elution positives only

**What**: Add mono-allelic mass spec elution data. Train the processing and presentation latents/heads. Class I only. Binding data continues with all stage-2 augmentations.

**Why**: Mono-allelic MS is the cleanest source of presentation signal: each eluted peptide was definitively presented by a known single allele. This separates the processing/presentation learning from the allele-ambiguity problem of multi-allelic data (stage 5). The processing latent needs to learn: "given this peptide in this flanking context, does it survive cleavage and transport?"

This is analogous to the pretraining → fine-tuning pattern in BERT (Devlin et al., 2019), where the model first learns general representations (binding, stage 2) then fine-tunes on a downstream task (presentation, stage 3). The key difference is that we continue training on binding data (replay) rather than fine-tuning exclusively on presentation data.

**Init**: Stage 2c checkpoint. Reset optimizer for the new presentation heads; carry optimizer state for existing binding heads.

**Data**:
- Replay: Stage 2 affinity data (30% of batch, with synthetics/contrastive)
- New: Mono-allelic MS elution hits (class I, positive only)

**Loss**: Binary cross-entropy for presentation (eluted = positive). Binding losses continue via replay. Uncertainty weighting balances the two.

**Synthetics**: Affinity synthetics carried forward via replay. No MS-specific synthetics yet (we need to establish the positive signal first before adding negatives).

**Contrastive**: Affinity contrastive carried forward via replay. No MS contrastive yet.

**Trunk training mode**: Freeze trunk encoder for the first 2-3 epochs, then unfreeze with 10x lower learning rate. This follows the gradual unfreezing approach from ULMFiT — let the new presentation heads warm up on the frozen encoder before allowing presentation gradients to modify the shared representation. This prevents early, noisy presentation gradients from damaging the binding representation.

**Duration**: Until validation presentation AUPRC plateaus (patience 3 epochs). Expected: 5-15 epochs.

**Checkpoint**: `stage3a_monoallelic_positives.pt`

**Eval**: Binding metrics (must not degrade from 2c). Presentation recall and AUPRC on held-out mono-allelic peptides. Check that the model isn't trivially predicting "presented" for everything (precision must be meaningful).

### Stage 3b: Add MS negatives and processing contrastive

**What**: Add three types of negatives for the presentation task, plus a processing-specific contrastive loss.

**Why**: MS data is positive-only — you observe what was presented, never what wasn't. Without negatives, the model converges to predicting "presented" for everything (trivial 100% recall, 0% precision). The negatives teach three distinct lessons:

1. **Peptide scramble**: Wrong binding motif → not presented (even if processed correctly). This reinforces the binding → presentation dependency.
2. **Flank shuffle**: Same core peptide, wrong flanking context → may not be cleaved/transported correctly. This teaches the processing latent that upstream context matters.
3. **Proteome hard negatives**: Real human proteome peptides not observed in the MS experiment for that allele. These are the hardest negatives — they look like real peptides, have plausible length and composition, but the cell chose not to present them. This is similar to the "hard negative mining" strategy in FaceNet (Schroff et al., 2015) — easy negatives (random sequences) become uninformative quickly; hard negatives from the actual proteome keep pushing the decision boundary.

**Processing contrastive**: Pairs of peptides presented by the same allele where one has high predicted processing score and the other has low. The processing prediction should correlate with flanking context quality — this loss directly rewards that.

**Init**: Stage 3a checkpoint. Carry optimizer state.

**Data**:
- Replay: Stage 2 affinity data (25% of batch, with synthetics/contrastive)
- Real: Mono-allelic MS positives (40% of batch)
- Synthetic negatives (15% of batch):
  - `peptide_scramble` for MS (ratio 0.3 of MS positives)
  - `flank_shuffle` (ratio 0.15 of MS positives)
  - Proteome hard negatives (ratio 0.3 of MS positives, sampled from 9-11mer human proteome peptides NOT observed for that allele in that experiment)

**Contrastive**: All prior affinity contrastive + processing contrastive (pairs of presented vs non-presented peptides with similar binding scores, forcing the model to discriminate on processing features rather than binding).

**Proteome negative mining**: Use the reference human proteome (UniProt reviewed, ~20K proteins). For each allele, digest in silico to 8-14mers. Exclude any peptide observed in any MS experiment for that allele. Sample length-matched negatives. Regenerate the negative pool each epoch (different random subset of the proteome) to prevent memorization. This is the MS analog of what MHCflurry 2.0 does for its presentation decoys.

**Duration**: Until validation presentation AUPRC plateaus (patience 3 epochs). Expected: 5-15 epochs.

**Checkpoint**: `stage3b_monoallelic_full.pt`

**Eval**: Binding metrics + presentation precision/recall/AUPRC on held-out mono-allelic data. Processing-specific metric: among peptides with matched binding affinity, does processing score correlate with presentation?

---

## Stage 4: Class II binding and presentation

### Stage 4a: Class II affinity + mono-allelic MS

**What**: Extend to class II alleles for both affinity and mono-allelic MS. Introduce the binding core scanning mechanism for class II's open groove.

**Why**: Class II MHC has an open-ended groove that accommodates longer peptides (13-25mers) with a 9-mer binding core buried in the middle. The core's position within the peptide is not fixed — it must be identified. Rather than training a separate class II core predictor, we reuse the class I binding function as a scanner: evaluate binding for each candidate 9-mer window, then select or aggregate.

This is architecturally similar to the scanning approach in NetMHCIIpan (Jensen et al., 2018), which scores all possible binding cores and selects the best. The difference is that our scanner is the full Presto binding module (not a simple position-specific scoring matrix), and it was pre-trained on class I data.

**Scanning mechanism options** (in order of preference):

1. **Soft attention over cores** (differentiable): Score each candidate core, apply softmax to get attention weights, take weighted average of predictions. This is fully differentiable and trains end-to-end. Risk: the softmax may spread probability across multiple cores rather than committing to one, blurring the binding signal.
2. **Gumbel-softmax** (approximately differentiable): Sample a core from the score distribution using Gumbel-softmax (Jang et al., 2017). Straight-through estimator in the forward pass, gradient flows through the soft approximation in the backward pass. Better for learning a sharp core selection without the blurriness of soft attention.
3. **Hard argmax + REINFORCE**: Select the highest-scoring core and use REINFORCE (Williams, 1992) to train the core selection. Most biologically accurate (the real binding core is one window, not a mixture) but high-variance gradients.

Recommendation: Start with soft attention (simplest), switch to Gumbel-softmax if the model fails to commit to a single core.

**Init**: Stage 3b checkpoint. Reset optimizer for class II-specific heads; carry state for class I heads.

**Data**:
- Replay: Stage 2 affinity (class I) + stage 3 MS (class I) (30% of batch, with all synthetics/contrastive)
- New: Class II affinity + class II mono-allelic MS (50% of batch)
- Synthetics: All prior + `peptide_scramble` for class II affinity and MS (15% of batch)

**Contrastive**: All prior + class II binding contrastive (cross-allele ranking for class II allele pairs).

**Trunk training mode**: Partially frozen for first 2-3 epochs (only class II-specific parameters trainable), then full unfreeze with discriminative learning rates (trunk at 0.1x, binding heads at 0.3x, new class II heads at 1x). This follows the discriminative fine-tuning approach from ULMFiT.

**Duration**: Until class II validation Spearman plateaus (patience 3 epochs).

**Checkpoint**: `stage4a_class2.pt`

**Eval**: Class I binding metrics (must not degrade). Class II binding Spearman and AUROC. Class II MS AUPRC. Binding core accuracy on peptides with experimentally known cores (from crystal structures).

### Stage 4b: Core selection refinement

**What**: Add core-window contrastive loss for class II peptides where the binding core is experimentally known.

**Why**: Crystal structures and competition assays identify the true binding core for some class II peptide-allele pairs. These gold-standard annotations can directly supervise the core selection mechanism. The contrastive loss pairs the correct core window against off-register windows: "the model should score the true 9-mer core higher than shifted windows."

This is analogous to anchor-positive-negative triplet training in metric learning — the true core is the positive, off-register cores are negatives.

**Init**: Stage 4a checkpoint. Carry optimizer state.

**Data**: Same as 4a + core-annotated class II peptides for the contrastive loss.

**Contrastive**: All prior + core-window ranking loss (true core vs off-register cores, margin 0.5).

**Duration**: Until core accuracy plateaus (patience 3 epochs). This sub-stage may be short if gold-standard core annotations are limited.

**Checkpoint**: `stage4b_class2_refined.pt`

**Eval**: Same as 4a + core identification accuracy (% of peptides where predicted core matches known core ±1 position).

---

## Stage 5: Multi-allelic MS with MIL

### Stage 5a: MIL bag-level training

**What**: Add multi-allelic mass spec data using multiple instance learning (MIL). An eluted peptide from a cell line expressing alleles {A, B, C, D, E, F} is treated as a "bag" — the model predicts presentation probability for each allele and the bag-level prediction is `max(allele_probs)` (the peptide was presented by at least one allele).

**Why**: Multi-allelic MS is the largest source of presentation data (~millions of peptides across thousands of experiments) but has allele ambiguity. MIL handles this without requiring allele deconvolution. The `max` aggregation assumes each peptide is presented by its best-binding allele — this is biologically reasonable for class I (typically one dominant presenter per peptide).

This is the standard MIL formulation from Dietterich et al. (1997), applied to immunopeptidomics as in Gfeller et al. (2018) and MHCflurry 2.0. The `max` pooling is equivalent to the "standard MI assumption" — a bag is positive iff at least one instance is positive.

**Init**: Stage 4b checkpoint. Carry optimizer state for existing heads; initialize MIL-specific parameters.

**Data**:
- Replay: All prior stages (25% of batch)
- New: Multi-allelic MS elution data, class I and II (50% of batch)
- Synthetics: All prior + genotype-swap decoys (see 5b) not yet added (15% of batch is prior synthetics)

**Loss**: Bag-level binary cross-entropy: `BCE(max(allele_probs), 1.0)` for eluted peptides. The max is taken over the allele set for that sample. Uncertainty-weighted against binding and mono-allelic losses.

**Contrastive**: All prior carried forward. No MIL-specific contrastive yet.

**Trunk training mode**: Fully trainable. By stage 5, the trunk has been gradually unfrozen through stages 3 and 4, and the binding representation is stable enough to tolerate MIL gradients.

**Duration**: Until bag-level AUPRC plateaus (patience 3 epochs).

**Checkpoint**: `stage5a_mil_positives.pt`

**Eval**: All prior metrics + bag-level AUPRC on held-out multi-allelic experiments. Per-allele presentation AUPRC (deconvolve using the model's own allele probabilities and compare to mono-allelic ground truth where available).

### Stage 5b: MIL contrastive + genotype-swap decoys

**What**: Add MIL contrastive loss and genotype-swap decoys.

**Why**: Without explicit genotype discrimination, the model can cheat on MIL by predicting "presented" for any processed-looking peptide regardless of the allele set. The genotype-swap contrastive pairs a real bag (peptide + true genotype) against a decoy bag (same peptide + unrelated genotype from a different cell line with <90% MHC sequence identity). This forces the presentation prediction to depend on the specific alleles present.

This is related to the "contrastive predictive coding" approach (van den Oord et al., 2018) — the model must distinguish the true context (real genotype) from negative contexts (wrong genotypes).

**Init**: Stage 5a checkpoint. Carry optimizer state.

**Data**: Same as 5a + genotype-swap decoys.

**Contrastive**: All prior + `mil_contrastive`:
- Pairs: (peptide, real_genotype) vs (peptide, decoy_genotype)
- Decoy selection: Sample genotypes from different experiments with <90% MHC sequence identity to the real genotype
- Margin: 0.5 in logit space
- Max 32 pairs per batch

**Synthetics**: All prior + genotype-swap decoys as presentation-level negatives.

**Duration**: Until validation MIL AUPRC plateaus (patience 3 epochs).

**Checkpoint**: `stage5b_mil_full.pt`

**Eval**: All prior + genotype-swap discrimination accuracy (the model should predict lower presentation for the decoy genotype in >90% of pairs). This is a direct test of whether the model uses allele information for presentation prediction.

---

## Stage 6: T-cell immunogenicity

### Stage 6a: T-cell assay data

**What**: Add T-cell reactivity data. Train immunogenicity output heads.

**Why**: T-cell data is the final biological layer — presented peptides that trigger an immune response. This is downstream of everything: processing, binding, presentation, plus TCR-side recognition. T-cell datasets are naturally contrastive (each experiment tests multiple peptides and reports positive/negative results), so we have real negatives unlike the MS setting.

The challenge is data scarcity and noise. IEDB T-cell assays have ~10K-50K entries, much smaller than binding (~250K) or MS (~millions). The assays are also heterogeneous (IFN-γ ELISPOT, intracellular cytokine staining, tetramer binding, proliferation, etc.) with different sensitivity and specificity.

**Init**: Stage 5b checkpoint. Carry optimizer state for existing heads; reset for immunogenicity heads.

**Data**:
- Replay: All prior stages (25% of batch)
- New: T-cell assay data from IEDB, both positive and negative results (50% of batch)
- Hard negatives: Peptides confirmed presented (from MS data) but NOT immunogenic in matched T-cell assays. These are the hardest negatives — they passed processing, bound MHC, reached the cell surface, but didn't trigger a T-cell response. The model must learn what makes a presented peptide immunogenic vs ignored. (15% of batch)

**Loss**: Binary cross-entropy for immunogenicity (positive/negative T-cell response). Uncertainty-weighted against all prior losses.

**Contrastive**: All prior carried forward. No immunogenicity-specific contrastive yet.

**Duration**: Until immunogenicity AUROC plateaus (patience 5 epochs — longer patience because the dataset is small and noisy).

**Checkpoint**: `stage6a_tcell.pt`

**Eval**: All prior metrics (binding, presentation) + immunogenicity AUROC, AUPRC, precision/recall. Check that binding and presentation metrics haven't degraded (T-cell data is noisy and could corrupt the shared encoder if the gradients are too aggressive).

### Stage 6b: Immunogenicity contrastive

**What**: Add contrastive pairs between immunogenic and non-immunogenic peptides presented by the same allele.

**Why**: The hardest immunogenicity problem is distinguishing two peptides that are both presented on the same allele, but only one triggers a T-cell response. The difference must be in the peptide's foreignness (self vs pathogen), TCR-facing residue properties, or population-level TCR repertoire coverage. Contrastive pairs that hold the allele constant and vary the peptide force the model to learn these features.

This is related to the "hard pair mining" strategy in Hermans et al. (2017) — within each allele, find the most confusing immunogenic/non-immunogenic pair and use it for contrastive training.

**Init**: Stage 6a checkpoint. Carry optimizer state.

**Contrastive**: All prior + immunogenicity contrastive:
- Pairs: (allele, immunogenic_peptide) vs (allele, non_immunogenic_peptide), both confirmed presented
- Source: Matched MS + T-cell data (peptides observed in MS for an allele AND tested in T-cell assays for the same allele)
- Margin: 0.3 in logit space

**Duration**: Until immunogenicity AUROC plateaus (patience 5 epochs).

**Checkpoint**: `stage6b_full_model.pt` — the final multi-task model.

**Eval**: Full evaluation suite. Compare to 6a. If contrastive doesn't help (small matched MS + T-cell dataset), this sub-stage may have minimal impact.

---

## Summary tables

### Synthetic negative inventory

| Introduced | Type | Description | Ratio | Persists through |
|------------|------|-------------|-------|-----------------|
| 2c | `peptide_scramble` (affinity) | Shuffled peptide, forced anchor changes, target 500-20000 nM | 0.10 of real | All subsequent |
| 3b | `peptide_scramble` (MS) | Shuffled peptide as presentation negative | 0.30 of MS positives | All subsequent |
| 3b | `flank_shuffle` (MS) | Swapped flanking context | 0.15 of MS positives | All subsequent |
| 3b | Proteome hard negatives | Unobserved human proteome peptides for each allele | 0.30 of MS positives | All subsequent |
| 4a | `peptide_scramble` (class II) | Shuffled class II peptides for affinity and MS | Same ratios as class I | All subsequent |
| 5b | Genotype-swap decoys | Same peptide + unrelated MHC genotype | 32 per batch | All subsequent |
| 6a | Presented-not-immunogenic | MS-confirmed, T-cell-negative peptides | 0.15 of batch | 6a-6b |

### Contrastive loss inventory

| Introduced | Type | Pairs | Margin | Persists through |
|------------|------|-------|--------|-----------------|
| 2b | `binding_contrastive` | Same peptide, different alleles | 0.2 log10 | All subsequent |
| 2b | `binding_peptide_contrastive` | Same allele, different peptides | 0.2 log10 | All subsequent |
| 3b | Processing contrastive | Same allele, similar binding, different presentation | 0.3 logit | All subsequent |
| 4a | Class II binding contrastive | Same peptide, different class II alleles | 0.2 log10 | All subsequent |
| 4b | Core-window contrastive | True core vs off-register cores | 0.5 logit | All subsequent |
| 5b | `mil_contrastive` | Real genotype vs swap genotype | 0.5 logit | All subsequent |
| 6b | Immunogenicity contrastive | Same allele, immunogenic vs non-immunogenic | 0.3 logit | 6b |

### Checkpoint chain

```
stage1_pretrain.pt
  └─ stage2a_affinity_clean.pt
       └─ stage2b_affinity_contrastive.pt  (skippable if hurts calibration)
            └─ stage2c_affinity_full.pt
                 └─ stage3a_monoallelic_positives.pt
                      └─ stage3b_monoallelic_full.pt
                           └─ stage4a_class2.pt
                                └─ stage4b_class2_refined.pt
                                     └─ stage5a_mil_positives.pt
                                          └─ stage5b_mil_full.pt
                                               └─ stage6a_tcell.pt
                                                    └─ stage6b_full_model.pt
```

Every checkpoint is independently deployable for all tasks trained up to that point.

---

## Decision gates

Each sub-stage has an explicit decision: keep or revert.

| Gate | Condition to keep | Fallback |
|------|-------------------|----------|
| 2b → 2c | Probe discrimination ≥ 2a | Skip 2b, proceed from 2a |
| 2c → 3a | A02 probe Kd < 100 nM | Reduce synthetic ratio or revert to 2b/2a |
| 3a → 3b | Binding Spearman within 0.01 of 2c | Freeze trunk permanently for MS training |
| 4a → 4b | Class I binding Spearman within 0.01 of 3b | Freeze class I heads, only train class II |
| 5a → 5b | Bag AUPRC > 0.5 | More epochs at 5a before advancing |
| 6a → 6b | Immunogenicity AUROC > 0.6 | More epochs at 6a or more aggressive replay |

---

## Open design questions

1. **Reservoir buffer size**: 10K per stage is a guess. Empirically validate by comparing 5K vs 10K vs 50K on stage 3 binding regression metrics.

2. **Uncertainty weighting initialization**: Initialize all $\log \sigma_i = 0$ (equal weights) or initialize new-task $\log \sigma_i$ higher (lower initial weight for new, noisy tasks)?

3. **Multi-allelic MIL pooling**: `max` assumes one dominant presenting allele. Consider `log-sum-exp` (smooth approximation to max) or attention-weighted pooling (learn which allele is most likely the presenter). Attention pooling is used in Attention MIL (Ilse et al., 2018) and may be more appropriate when multiple alleles contribute.

4. **Proteome negative refresh frequency**: Regenerate every epoch (diversity) or every N epochs (stability)? If every epoch, the model never memorizes negatives but the negative set is non-stationary. If every N epochs, the model may memorize the current negatives.

5. **Class II core window enumeration**: Score all possible 9-mer windows, or use a preliminary filter (e.g., require at least one anchor residue match) to reduce the candidate set? NetMHCIIpan scores all windows; the preliminary filter is faster but may miss non-canonical cores.

6. **T-cell assay heterogeneity**: Treat all T-cell assay types equally, or weight by assay reliability? IFN-γ ELISPOT is more specific than proliferation assays. Consider assay-type-aware loss weighting or separate heads per assay type (similar to the DAG assay decomposition for binding).

7. **Stage 1 peptide source labels**: Requires curated organism → vertebrate/pathogen mapping. Some peptides have missing or ambiguous source organism annotations. Decide whether to exclude ambiguous entries or assign soft labels.

8. **When to stop the curriculum**: Not every application needs all 6 stages. A binding-only predictor stops at stage 2. A presentation predictor stops at stage 5. Only immunogenicity prediction needs all 6 stages. Make each checkpoint independently useful.

## References

- Bengio, Y. et al. (2009). Curriculum Learning. ICML.
- Burges, C. J. C. (2010). From RankNet to LambdaRank to LambdaMART. MSR-TR-2010-82.
- Chaudhry, A. et al. (2019). Efficient Lifelong Learning with A-GEM. ICLR.
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers. NAACL.
- Dietterich, T. G. et al. (1997). Solving the Multiple Instance Problem with Axis-Parallel Rectangles. Artif. Intell.
- Elnaggar, A. et al. (2022). ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning. IEEE TPAMI.
- Gfeller, D. et al. (2018). The Length Distribution and Multiple Specificity of Naturally Presented HLA-I Ligands. J. Immunol.
- Hermans, A. et al. (2017). In Defense of the Triplet Loss for Person Re-Identification.
- Howard, J. & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. ACL.
- Ilse, M. et al. (2018). Attention-based Deep Multiple Instance Learning. ICML.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. ICLR.
- Jensen, K. K. et al. (2018). Improved Methods for Predicting Peptide Binding Affinity to MHC Class II Molecules. Immunology.
- Jumper, J. et al. (2021). Highly Accurate Protein Structure Prediction with AlphaFold. Nature.
- Kendall, A. et al. (2018). Multi-Task Learning Using Uncertainty to Weigh Losses. CVPR.
- Khosla, P. et al. (2020). Supervised Contrastive Learning. NeurIPS.
- Kirkpatrick, J. et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. PNAS.
- Lin, L.-J. (1992). Self-Improving Reactive Agents Based on Reinforcement Learning. Machine Learning.
- Lopez-Paz, D. & Ranzato, M. (2017). Gradient Episodic Memory for Continual Learning. NeurIPS.
- Mikolov, T. et al. (2013). Distributed Representations of Words and Phrases. NeurIPS.
- O'Donnell, T. J. et al. (2020). MHCflurry 2.0: Improved Pan-Allele Prediction of MHC Class I-Presented Peptides. Cell Systems.
- Reynisson, B. et al. (2020). NetMHCpan-4.1 and NetMHCIIpan-4.0. Nucleic Acids Res.
- Rives, A. et al. (2021). Biological Structure and Function Emerge from Scaling Unsupervised Learning. PNAS.
- Schaul, T. et al. (2016). Prioritized Experience Replay. ICLR.
- Schmidt, J. et al. (2021). Prediction of Neo-Epitope Immunogenicity Reveals TCR Recognition Determinants and Provides Insight into Immunoediting. Cell Reports Medicine.
- Schroff, F. et al. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR.
- van den Oord, A. et al. (2018). Representation Learning with Contrastive Predictive Coding.
- Vitter, J. S. (1985). Random Sampling with a Reservoir. ACM Trans. Math. Softw.
- Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning. Machine Learning.
