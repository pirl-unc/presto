# Source-junction ambiguity masking comparison

## Question

Does replacing genuinely unresolved source-protein junctions with explicit
unknown flank context (`?/?`) improve held-out prediction relative to the
historical global-canonical / deterministic-protein choice, enough to justify
later candidate-junction marginalization?

This is a policy-isolation experiment. Candidate marginalization and RNA
expression are not factors in this family.

## Dataset contract

- Source: hitlist 1.55.8 curated indexes, hitlist-only Presto ingest.
- Allele filter: `HLA-A*02:01`.
- Length: production default, 7--30 aa.
- Caps before dataset construction:
  - binding: all qualifying numeric records (the 5,000-row preflight left too
    few unresolved test examples; the filtered source has 29,784 evidence rows
    before numeric routing);
  - elution/MS: 20,000 records;
  - stability: 1,000 records;
  - kinetics: 500 records;
  - no processing, T-cell, TCR, merged-TSV, or bulk-proteomics records.
- Cap policy: reservoir sampling for capped modalities, seeded with training
  seed + 17.
- Split: peptide-disjoint 80/10/10 train/validation/test, with all rows for a
  peptide confined to one partition.
- Qualifier/censor policy: retain all curated binding qualifiers; report the
  censor-aware task metrics, exact-only regression metrics (`qualifier == 0`),
  and qualifier-aware `<=500 nM` classification metrics.
- Assay families: numeric binding (IC50, direct KD, KD proxies, EC50), stability
  and kinetics when present, and positive MS/elution evidence. The primary
  decision metric is binding because the real hitlist-only MS corpus supplies
  positives but no real elution negatives.
- Frozen local artifacts to stage into the Modal data volume:
  - `observations.parquet`: SHA256
    `f51440ab229fd187d2548b4dddcd1fc04580d97d45fb4d5b8e0222aa8080f928`;
  - `binding.parquet`: SHA256
    `fbcae6f762f43edb4eb87b1a7c7f3849757859204d37f3ce82d1f83c23cece9f`;
  - `peptide_mappings.parquet`: 5,880,924 rows, SHA256
    `45580c16649daf75b11b51497d9d96dfaa987cdcf1197d6449deaa9261f6ec5c`;
  - `observations_meta.json`: SHA256
    `ac459e184fc6c54c73f7f1b4fc7dff424b2360b13f55f12bb68a3bbb743de118`;
  - `peptide_mappings_meta.json`: artifact contract v2, SHA256
    `9a93de21753029ac08f1ba05e1a667ee6d7fd1b63a885bdc8dade1e626c7cbbb`.

## Conditions

Paired across seeds 42, 43, and 44:

1. `legacy_global_canonical`: reproduce the prior global-canonical /
   arbitrary-source semantics with stable candidate ordering, and feed the
   selected junction to the model even when candidates disagree across genes.
   This is deliberately reproducible rather than a byte-for-byte replay of the
   old bulk frame-order fallback.
2. `mask_unresolved`: preserve single mappings, agreeing multi-mappings, and a
   unique canonical source only within one gene; replace cross-gene and
   residual within-gene disagreement with `?/?`.

Both conditions retain identical observation targets and row-level mapping
categories. Only flank input on unresolved categories changes.

## Training contract

- Unified Presto trainer, hitlist-only input.
- Expanded latent topology; `d_model=128`, 2 layers, 4 heads.
- 10 epochs, batch size 256, AdamW, learning rate `2.8e-4`, weight decay 0.01.
- AMP/bfloat16 on CUDA; no `torch.compile`.
- Uncertainty-weighted, task-mean supervised aggregation and the trainer's
  current consistency/loss weights, identical across conditions.
- No synthetic pMHC, no-B2M, processing, presentation, or T-cell negatives;
  no MHC-only augmentation. This prevents generated decoys from being confused
  with biological held-out negatives in the policy comparison.
- Ordinary seeded shuffle, not the balanced batch sampler.
- Final-epoch validation/test predictions are the comparison target; best
  validation loss is still saved as trainer metadata.
- Hardware: Modal `H100!` for every condition. Record requested GPU and observed
  GPU model/memory from worker logs.
- hitlist package pinned to exactly 1.55.8 in the experiment image.

## Metrics and decision rule

Report validation and test metrics for the binding task:

- all-row Spearman, Pearson, RMSE in `log10(nM)`;
- exact-only Spearman, Pearson, and RMSE;
- qualifier-aware `<=500 nM` accuracy, balanced accuracy, precision, recall,
  F1, AUROC, and AUPRC;
- each metric stratified by `single`, `flanks_agree`,
  `within_gene_canonical`, `cross_gene_unresolved`, and
  `within_gene_unresolved`; report `unmapped` separately but do not use it to
  decide the ambiguity policy.

Primary comparison: paired seed difference in test exact Spearman and RMSE for
the union of the two unresolved categories. Secondary: overall binding and
`<=500 nM` metrics. Do not interpret elution AUPRC as biological performance
because held-out real elution rows contain no real negatives.

Invest in candidate-junction marginalization only if masking has a consistent
direction across all three seeds and either improves unresolved-category exact
Spearman by at least 0.01 or lowers unresolved-category exact RMSE by at least
0.02 log10(nM), without a material overall regression. Otherwise keep explicit
unknown masking and defer marginalization.

For closure, a material overall regression means a mean test exact-Spearman
change below -0.01 or a mean test exact-RMSE change above +0.02 log10(nM).
This operational definition was frozen before reading any completed-run result.

## Expected artifacts

- One run directory per condition/seed with checkpoint, config, logs,
  `val_summary.json`, `test_summary.json`, `val_metrics.csv`,
  `test_metrics.csv`, and per-example validation/test prediction CSVs.
- Experiment-local manifest with run ids, policy, seed, requested/observed GPU,
  data hashes, git commit, and dirty status.
- Aggregated overall/category metrics CSV and paired-difference JSON; plots only
  if they clarify a nontrivial effect.
- Frozen `code/launch.py`, `reproduce/launch.sh`, `reproduce/launch.json`, and
  `reproduce/source/launch.py`.
- Closed README and canonical `experiments/experiment_log.md` entry.

## Risks / checks before launch

- Run one local data audit with the exact caps. The initial 5,000-binding audit
  produced only 12--21 cross-gene and 1--2 residual within-gene test rows per
  seed, so the final pre-launch contract was revised to uncapped binding. If
  the final audit still leaves an individual category too small for
  correlation, preserve the preregistered unresolved union and treat that
  individual category as descriptive.
- Modal must use `/data/hitlist` through `HITLIST_DATA_DIR` but an empty trainer
  `--data-dir`, otherwise the mounted merged TSV silently adds T-cell/TCR rows.
- Any condition with missing held-out artifacts, a different data hash, or a
  non-H100 worker is invalid and must be rerun before aggregation.
- The initial `20260902a` launch was stopped during the first/second epoch,
  before results were read, after review found that selected-`X` filtering ran
  after policy transformation and could change row membership between arms.
  The corrected `20260902b` launch filters the common deterministic selection
  before applying either flank-input policy; aggregation also requires exact
  paired supervision-row parity.
