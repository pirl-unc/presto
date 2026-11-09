# Unified parity retrain evaluation and report

Date: 2026-11-08
Owner: Codex / GPT-5
Status: preregistered before evaluator implementation or final fit collection

## Objective

Evaluate the frozen reference and comparator-parity retrains on the exact
unchanged Stage B development population, alongside the historical Stage B
checkpoint and already frozen NetMHCpan/NetMHCIIpan/MHCflurry predictions.
Produce deterministic per-example tables, reusable figure-generation code, and
one timestamped state-of-Presto report without accessing TEST, locked
confirmation, or T-cell outcomes.

This is a development comparison. It can choose the next experimental parent
under the already preregistered gates, but it cannot establish independent
superiority over any comparator.

## Immutable inputs and row gates

1. Require completed, locally hash-verified `fits-v3/reference` and
   `fits-v3/parity` receipts before final evaluation.
2. Require both selected-checkpoint validation dumps to contain exactly 51,066
   records and the same ordered observation-identity multiset as each other and
   the frozen historical Stage B dump.
3. Require endpoint census equality: 48,530 `elution_observation`, 1,597
   `ic50`, 69 `ec50`, 528 `kd_via_ec50`, and 342 `kd_via_ic50` rows.
4. Join affinity comparator predictions only by `observation_id`; require exact
   equality of peptide, class, endpoint, qualifier, canonical target, study,
   and assay fields before admitting a row. Redundant raw-nM values reconstructed
   from the canonical log target may differ only within `1e-12` relative error.
5. Join MHCflurry predictions only by frozen `pair_id`; require equal target,
   study, sample, and peptide/MHC identity fields before admitting a row.
6. Fail on duplicate identities, missing comparator values inside a declared
   matched cohort, non-finite scores, split drift, or any TEST/confirmation row.

The evaluator consumes local copies whose bytes and SHA-256 values are pinned in
its launch contract. Large checkpoints and prediction dumps stay outside Git;
Git receives code, launch metadata, compact tables, figures, and receipts only.

## Output semantics

- Affinity uses each row's explicit assay-aware prediction selected by the
  frozen training evaluator. Do not pool assay methods into a generic head.
- Primary comparator affinity is exact IC50, split by MHC-I and MHC-II. Report
  pooled and equal-study Spearman, Pearson, RMSE in `log10(nM)`, and the complete
  qualifier-aware `<=500 nM` classification family.
- Preserve separate summaries for exact EC50, IC50, KD-via-EC50, and
  KD-via-IC50. Censored rows get satisfaction and violation magnitude; they do
  not enter exact-value correlation.
- Presentation is the selected elution-observation logit/probability.
- Operational processing is the model's MHC-register-marginalized processing
  contribution plus the same MS-response and offset terms used by the frozen
  Stage B comparator analysis. Label it operational, not a direct proteolysis
  measurement.
- Report pooled, equal-study, and per-study AUROC/AUPRC for presentation and
  operational processing.

## Uncertainty and paired comparisons

Use deterministic seed `42` and 2,000 paired bootstrap replicates. Resample
whole studies for the primary study-aware interval. When a stable peptide-family
key is available, resample study/family clusters; otherwise report that this
secondary interval was unavailable rather than substituting row bootstrap.
Each replicate evaluates all methods on the identical sampled rows. Preserve
the replicate contract and seeds in the receipt, but commit only compact
interval tables unless the replicate file remains small.

## Figures

Implement reusable functions under `evaluation/` and keep the experiment driver
thin. Generate deterministic PNG and SVG panels for:

1. observed versus predicted exact affinity by MHC class/method, with identity
   line and marginal metrics;
2. affinity residual and reliability/calibration views in `log10(nM)`;
3. `<=500 nM` ROC and precision-recall curves;
4. paired per-study affinity deltas with pooled bootstrap intervals;
5. presentation and operational-processing ROC and precision-recall curves;
6. per-study presentation/processing comparator deltas.

Figures must consume saved per-example tables rather than discover or rescore a
checkpoint. SVG hash salts and metadata are fixed so identical inputs produce
identical outputs.

## Report and decision

Generate a timestamped report using the versioned performance-report engine.
Its manifest must include source commit/dirty state, both training receipts,
checkpoint hashes, every dataset/prediction/comparator artifact hash, training
and curation contracts, requested/observed hardware, runtime, and explicit
development-only limitations.

Apply the preregistered parent decision exactly: prefer parity only if MHC-II
exact-IC50 Spearman improves by at least 0.05 over reference, MHC-I declines by
no more than 0.01, and neither presentation nor operational-processing AUROC
declines by more than 0.01. Record comparator gaps, but do not use comparator
outcomes to change this threshold after seeing results.

If parity fails the MHC-II gate, specify the next single chain-aware class-II
mechanism experiment from the frozen reference/parity evidence. Do not begin
T-cell training until the affinity/presentation parent decision is closed.

## Verification

- Unit-test row identity rejection, qualifier-aware thresholding, equal-study
  aggregation, paired bootstrap determinism, operational-processing recovery,
  and deterministic figure bytes.
- Run Ruff and focused tests in the exact shared environment.
- Independently recompute primary metrics from the saved per-example table.
- Re-render the report twice in separate directories and require byte equality.
- Register final artifacts and conclusions in the experiment README and
  `experiments/experiment_log.md`.
