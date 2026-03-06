# Training Specification

This is the canonical Presto training specification.

Model architecture is in `design.md`. TCR pathway details are in `tcr_spec.md`
and are currently future-facing (not active in canonical training).

**Normative choice**: Use one unified mixed-source training loop with
time-varying task/regularizer weights. Do not use hard stage boundaries.
Do not keep parallel staged-training codepaths in the repository.

---

# 1. Objective

Train one model over all active canonical supervision signals so that:
- upstream biology (processing/binding/presentation) is shared,
- downstream assay heads are condition-specific readouts,
- synthetic negatives and biologic priors are integrated directly into the same loop.

---

# 2. Data Sources

## 2.1 Canonical Data Entry

Unified training command:

```bash
python -m presto train unified --data-dir ./data --epochs <N> --checkpoint <path>
```

Default canonical input:
- merged table: `data/merged_deduped.tsv` (or `--merged-tsv` override).

Canonical behavior:
- merged input is required by default (`--require-merged-input`),
- raw-source fallback is non-canonical and only enabled with `--allow-raw-fallback`.
- MHC alleles must resolve to amino-acid sequences by default (`--strict-mhc-resolution`).
  Unresolved alleles are a hard error, with unresolved-allele reports written to
  the run directory for triage. `--allow-unresolved-mhc` is debug-only.

Canonical cross-source consolidation before training:

```bash
python -m presto data merge --datadir ./data
```

This writes:
- one merged deduplicated table (`merged_deduped.tsv`),
- optional per-assay CSVs (`merged_assays/*.csv`) when `--per-assay-csv` is used.

## 2.2 Source Inventory

Core mergeable supervision sources loaded by `presto data merge`:
- `iedb`, `cedar`, `vdjdb`, `mcpas` (plus optional `bcell` rows from IEDB/CEDAR).
- `10x`, `imgt`, and `ipd_mhc` are used for auxiliary/sequence resolution but are not assay-row contributors in `merged_deduped.tsv`.

| Source | Key files | Assay/data types extracted | Role |
|--------|-----------|---------------------------|------|
| IEDB MHC ligand | `data/iedb/mhc_ligand_full_single_file.zip` | binding_affinity, binding_kon/koff/t_half/tm, elution_ms, processing-like | Core pMHC supervision |
| IEDB T-cell | `data/iedb/tcell_full_v3.zip` | tcell_response with assay/context metadata | Core T-cell supervision |
| CEDAR MHC ligand | `data/iedb/cedar_mhc_ligand_full_single_file.zip` | Same as IEDB ligand | Additional curated pMHC |
| CEDAR T-cell | `data/iedb/cedar_tcell_full_v3.zip` | tcell_response | Additional curated T-cell |
| VDJdb | `data/vdjdb/vdjdb.zip` | tcr_pmhc positives | Future TCR supervision pool (not active in canonical loop) |
| McPAS-TCR | `data/mcpas/McPAS-TCR.csv` | tcr_pmhc positives | Future TCR supervision pool (not active in canonical loop) |
| 10x VDJ | `data/10x/10x_pbmc_10k_tcr.csv` | chain/species/phenotype auxiliary labels | Optional auxiliary labels; not required in canonical objective |
| IMGT/HLA + IPD-MHC | `data/imgt/*.fasta`, `data/ipd_mhc/*.fasta` | Full MHC sequences (human + non-human) | Allele->sequence resolution |

Observed pre-dedup load totals by source x assay (from latest merge run):
- `iedb`: elution_ms=5,080,741, bcell_response=1,432,961, tcell_response=567,252, binding_affinity=252,458, binding_t_half=11,437, binding_tm=1,250, binding_koff=75, binding_kon=40.
- `cedar`: elution_ms=4,493,447, tcell_response=151,474, bcell_response=144,675, binding_affinity=82,183, binding_t_half=6,873, binding_tm=709, binding_koff=30, binding_kon=18.
- `vdjdb`: tcr_pmhc=226,494.
- `mcpas`: tcr_pmhc=15,058.

## 2.3 Cross-Source Dedup

Dedup strategy (implemented in `data/cross_source_dedup.py`):
1. Normalize keys: peptide sequence, MHC allele normalization, record type, plus type-specific keys.
2. Normalize references: PMID normalization, DOI normalization, normalized free-text reference strings.
3. Collapse duplicates only when both sample signature AND reference match.
4. Fuzzy reference matching uses sequence similarity threshold (default 0.92) and token-overlap fallback (>=0.85 Jaccard-like overlap).
5. For duplicate groups, choose canonical row by source preference and completeness.

Current merged snapshot:
- Total input rows: 12,467,175
- Total output rows: 5,593,161
- Cross-source duplicates removed: 4,734,734
- Source retention: IEDB 72.64%, CEDAR 1.86% (expected high overlap), VDJdb 67.37%, McPAS 91.22%

## 2.4 Canonical Assay Buckets

When `--per-assay-csv` is enabled, the merge step writes one CSV per assay bucket in `merged_assays/`.

| Assay CSV | Current rows | Primary supervision target(s) |
|-----------|-------------|-------------------------------|
| `binding_affinity.csv` | 251,418 | binding/log-affinity heads |
| `binding_kon.csv` | 40 | kon assay head |
| `binding_koff.csv` | 75 | koff assay head |
| `binding_t_half.csv` | 11,375 | t_half assay head |
| `binding_tm.csv` | 1,250 | Tm assay head |
| `elution_ms.csv` | 3,867,752 | presentation/elution/MS heads |
| `tcell_response.csv` | 478,464 | tcell_logit, immunogenicity/recognition supervision |
| `tcr_pmhc.csv` | 166,319 | reserved future TCR supervision bucket |
| `bcell_response.csv` | 816,468 | optional/non-canonical B-cell objectives |

Assay content details:
- `binding_affinity.csv`: ic50=163,850, ec50=58,674, kd=21,049, plus 7,845 other affinity-like value types. Class I: 156,007, Class II: 95,411. 452 unique MHC alleles.
- `elution_ms.csv`: Class I: 2,582,215, Class II: 1,285,537. 696 unique alleles.
- `tcell_response.csv`: ELISpot=235,365, ICS=43,456, multimer/tetramer=28,664, 51Cr=10,786. IFNg=261,573, IL-2=13,330, TNFa=7,005, IL-17=4,152, granzyme B=1,667.
- `tcr_pmhc.csv`: 166,319 rows, 185 unique alleles. Class I: 153,625, Class II: 12,694.

---

# 3. Unified Mini-Batch Construction

Each epoch uses mixed-task mini-batches. Samples can carry any subset of labels.

## 3.1 Canonical Task Groups

- Binding / kinetics / stability
- Processing
- Elution / MS
- T-cell assays and context labels
- Recognition supervision from peptide/context labels (peptide+foreignness canonical branch)
- Chain attribute supervision from 10x

## 3.2 Sparse-Label Collation

Batch collation is sparse-label aware:
- Every sample contributes only to losses for labels it actually has.
- Masks are required for each supervised target.

## 3.3 Balanced Mini-Batch Sampler

Training sampler behavior (`BalancedMiniBatchSampler`, default enabled):
- per-batch task quotas enforce assay/task mixing,
- per-task label alternation enforces positive/negative exposure where labels exist,
- weighted in-batch selection actively balances:
  - assay/task group,
  - data source,
  - label bucket,
  - primary MHC allele,
  - synthetic-negative subtype.
- synthetic negatives from distinct biologic constructions are represented through dedicated source/synthetic strata.

## 3.4 Stratification

Stratification axes:
```
(mhc_class, assay_category, label, species_group, data_source)
```

Recommended strategy: `sqrt_frequency` weighting:
```python
stratum_weight[s] = sqrt(count[s]) / sum(sqrt(count[s']) for s' in all_strata)
```

---

# 4. Synthetic Negative Schedule

Synthetic negatives are woven into data construction before batching.
They serve as structural supervision for the model's biological priors.

## 4.1 Binding Negatives

| Type | Construction | Rationale |
|------|-------------|-----------|
| `peptide_scramble` | Permute amino acids of the source peptide, keep source MHC | Disrupts motif compatibility while preserving residue composition |
| `peptide_random` | Generate de novo peptide sequence, keep source MHC | Non-binder baseline with no source-peptide structure |
| `mhc_scramble` | Keep source peptide, scramble MHC alpha-chain sequence | Breaks MHC structure while preserving amino-acid composition |
| `mhc_random` | Keep source peptide, generate de novo MHC-like alpha sequence | Harder MHC corruption baseline |
| `no_mhc_alpha` | Keep source peptide/allele, blank MHC alpha chain | Invalid complex should not bind/present |
| `no_mhc_beta` | Keep source peptide/allele, blank MHC beta chain | Invalid class-I assembly (no beta2m) should not bind/present |

## 4.2 Elution Negatives

| Type | Construction | Rationale |
|------|-------------|-----------|
| `peptide_random_mhc_real` | Random peptide with source alleles | Should not appear in MS/elution data |
| `peptide_real_mhc_random` | Source peptide with mismatched/random alleles | Should not appear in MS/elution data |
| `peptide_random_mhc_random` | Random peptide and random alleles | Hard negative with no biological pairing |
| Hard pair negatives | Peptide + known non-presenting allele | Allele-specific negatives with realistic peptides |

Note:
- For sequence-corruption modes, `random` means de novo AA generation and `scramble` means permutation.
- For elution allele corruption, `mhc_random` means allele sampling from in-class allele pools.

## 4.3 Processing Negatives

| Type | Construction | Rationale |
|------|-------------|-----------|
| `flank_shuffle` | Shuffle N/C flanks while keeping peptide | Disrupts cleavage-site context without changing peptide identity |
| `peptide_scramble` | Permute peptide while keeping flanks | Disrupts peptide-local processing motifs |

## 4.4 Cascaded Negatives

| Type | Construction | Rationale |
|------|-------------|-----------|
| Binding -> Elution | Synthetic binding negatives projected as elution negatives | If it doesn't bind, it can't be eluted |
| Binding -> T-cell | Synthetic binding negatives projected as T-cell negatives | If it doesn't bind, it can't elicit T-cell response |

These enforce the biological cascade: non-binders cannot be presented, and
non-presented peptides cannot be immunogenic.

## 4.5 User-Facing Controls (Canonical)

The unified trainer now exposes a simplified synthetic-negative interface:

| Flag | Default | Effect |
|------|---------|--------|
| `--synthetic-pmhc-negative-ratio` | `1.0` | Primary pMHC non-binder synthesis rate per real binding record; also drives downstream elution and cascade negatives |
| `--synthetic-class-i-no-mhc-beta-negative-ratio` | `0.25` | Additional class-I no-beta-chain negatives per real class-I binding record |
| `--synthetic-processing-negative-ratio` | `0.5` | Processing-specific corruption negatives per real processing record |

Derived downstream rates from `--synthetic-pmhc-negative-ratio`:
- elution corruption negatives: `0.5x`;
- cascade binding->elution negatives: `0.5x`;
- cascade binding->T-cell negatives: `0.5x`.

---

# 5. Loss Functions

## 5.1 Primary Supervised Losses

| Target | Loss | Latent/Head |
|--------|------|-------------|
| KD, IC50, EC50 | Censor-aware regression | binding_affinity |
| koff, t_half, Tm | Censor-aware regression | binding_stability |
| Processing | BCE | processing_class1/class2 |
| Presentation | BCE | presentation_class1/class2 |
| Elution/MS | BCE (bag MIL Noisy-OR for multi-allele) | presentation + ms_detectability |
| T-cell outcomes | BCE (per attested config) | immunogenicity + T-cell system |
| T-cell contexts | CE for method/readout/APC/culture prediction | Context heads |
| Core start | CE | Core pointer |
| MHC class/species | CE | Auxiliary heads |
| Contrastive TCR-pMHC | InfoNCE (future) | Reserved; not active in canonical loop |

Core-start CE is applied when `core_start` labels are present in the collated
batch (`targets["core_start"]`, `target_masks["core_start"]`).

### Censor-aware regression

Binding data often carries inequality qualifiers (<, =, >). The censor-aware
loss handles these correctly:

```python
def censor_aware_loss(pred, target, qualifier):
    """
    qualifier: '<' means target is an upper bound (IC50 > threshold -> non-binder)
               '=' means exact measurement
               '>' means target is a lower bound
    """
    residual = pred - target
    if qualifier == '=':
        return residual ** 2
    elif qualifier == '<':
        return relu(residual) ** 2    # penalize only if pred > upper bound
    elif qualifier == '>':
        return relu(-residual) ** 2   # penalize only if pred < lower bound
```

### Multi-allele bag MIL loss

For elution/presentation/MS with multi-allele bags:

```python
# Collator emits instance_to_bag mapping and bag_label
instance_probs = sigmoid(instance_logits)   # per-allele presentation probs
bag_prob = 1 - scatter_prod(1 - instance_probs, instance_to_bag)  # Noisy-OR
loss = bce(bag_prob, bag_label)
```

## 5.2 Loss Aggregation

**Uncertainty-based weighting** is the canonical default:
```python
# Learnable log-variance per task
log_var = nn.Parameter(torch.zeros(n_tasks))
task_loss_weighted = task_loss / (2 * exp(log_var)) + log_var / 2
```

Low-value sequence-labeling auxiliaries should also carry a smaller fixed base
weight before aggregation. Canonical defaults:
- `mhc_class`, `mhc_species`, `mhc_a_fine_type`, `mhc_b_fine_type`: `0.1`
- `chain_species`, `chain_type`, `chain_phenotype`: `0.1`
- `binding_affinity_probe`: `0.3`

This keeps `sample_weighted` aggregation focused on assay supervision instead
of letting always-labeled bookkeeping heads dominate by raw support count.

Optional **PCGrad** for conflicting gradients (project conflicting gradient
components to reduce task interference).

---

# 6. Time-Varying Weight Schedules

Weights are implemented as smooth ramps in the unified loop.

Epoch progress `p = epoch_idx / (epochs - 1)`.

| Loss group | Ramp |
|-----------|------|
| Supervised data losses | Active from start (weight 1.0) |
| Consistency priors (cascade, affinity, presentation, no-MHC-beta) | Ramp 0->1 over first 50% |
| T-cell consistency priors | Ramp 0->1 over first 70% |
| Orthogonality regularization | Active from start |

This keeps optimization stable while preserving one continuous training process.

---

# 7. Biologic Priors (Regularization)

Weighted regularizers (soft constraints):

| Prior | Constraint | Ramp |
|-------|-----------|------|
| Cascade anti-saturation | Prevent high presentation with low upstream support | 0-50% |
| Affinity-head agreement | Weak supervision on agreement between affinity/stability predictions | 0-50% |
| Chain-assembly prior | Invalid chain setups penalized | 0-50% |
| No-MHC-beta prior | Class I without beta2m should have low presentation | 0-50% |
| T-cell context ordering | In-vitro vs ex-vivo sensitivity ordering (see design.md S10.5) | 0-70% |
| T-cell upstream prior | Strong T-cell outputs require strong upstream biology | 0-70% |
| Binding orthogonality | \|cos(binding_affinity_vec, binding_stability_vec)\| minimized | From start |

Binding orthogonality is controlled by `binding_orthogonality_weight`
(canonical default: `0.01`).

---

# 8. Multi-Instance Training

Canonical requirement:
- Multiple-MHC bags are first-class for inference and training.

Implemented multi-allele training path:
- Elution/presentation/MS supervision uses bag-level MIL Noisy-OR.
- Each elution sample carries an allele bag.
- Collator flattens allele instances and emits `instance_to_bag` and `bag_label`.
- Training computes instance logits and bag-level BCE via Noisy-OR.

Multi-TCR bag training/inference follows the same semantics but remains a
tracked item in `TODO.md`.

---

# 9. Validation and Splits

Default evaluation should include:
- per-task metrics (affinity, presentation/elution, T-cell),
- source-aware reporting (IEDB vs CEDAR primarily; VDJdb/10x for future/auxiliary tracking),
- leakage-aware split policy for peptide/allele/epitope generalization studies.

Runtime observability:
- training/eval loops use `tqdm` progress bars by default,
- progress postfix reports running loss and samples/sec (`sps`),
- per-epoch logs include `train_samples_per_sec` and `eval_samples_per_sec`.

---

# 10. Runtime Notes (Measured)

Latest local benchmark snapshot (CPU host, merged canonical dataset):
- Merged canonical dataset size: 7,371,416 training samples after synthetic augmentation.
- Full profile batches: 46,072 train + 11,518 val (batch size 128).
- Observed startup throughput: ~3.1 samples/sec (~41 sec/batch) in full profile.
- Projected full epoch wall-clock on this host: hundreds of hours (not practical without accelerator).

Observed fast-profile throughput:
- Canary profile: ~6.6-8.8 samples/sec during train, ~31-42 samples/sec during eval (host-load dependent).

Operational guidance:
- Use GPU-backed runs (Modal or equivalent) for canonical full-data training.
- Keep `tqdm` + `sps` logging enabled for live throughput monitoring.
- When iterating locally, use canary profile or explicit `--max-*` caps.
- `--max-*` values are load-time record caps (per source/modality), not "records per epoch".
- With caps set to `0`, the loader uses the full merged corpus; each epoch iterates over the loaded split.
- GPU memory scales with batch tensors/activations, not total on-disk dataset size.

Bottleneck priorities:
1. Per-allele forward cost dominates full-data runtime.
2. Data loading/tokenization overhead is secondary but non-trivial.
3. Synthetic-negative expansion materially increases sample volume.

First optimization levers:
1. Precompute/cache tokenized sequences and allele encodings.
2. Use mixed precision and `torch.compile` in accelerator runs.
3. Tune synthetic-negative ratios during early experiments, then restore canonical ratios for final training.

---

# 11. Non-Normative Paths

- Reference-model comparisons and exploratory ideas are in `docs/notes/`.
