# Distributional vs Regression Output Heads for Censored BA Prediction

## Goal

Run a fixed-contract 32-condition comparison of output head encodings for broad class-I binding-affinity prediction under censor-aware supervision.

The encoder/backbone, optimizer, schedule, batch size, dataset, and number of epochs are held fixed. Only the output-head family changes.

## Fixed Contract

- Source:
  - `data/merged_deduped.tsv`
- Alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- Included assay families:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- Excluded:
  - qualitative binding
  - presentation / T-cell
- Qualifiers:
  - exact and `>` rows
  - censor-aware loss for all heads
- Split:
  - deterministic peptide-group train/val/test
  - target fractions `0.8 / 0.1 / 0.1`
- Batch size:
  - `256`
- Epochs:
  - `10`
- GPU:
  - `H100!`
- Optimizer / LR / schedule:
  - fixed across all 32 runs
  - use the broad-contract winner setup from prior experiments unless a bug blocks it
- Warm start:
  - use the same broad-contract warm-start policy across all runs

## Backbone Invariant

- Use one fixed backbone/encoder across all 32 runs.
- No architecture changes inside the encoder/trunk.
- Head-only comparison:
  - regression targets
  - distributional targets
  - assay integration mode
  - target max / binning / sigma

## Methods

### Regression

1. MHCflurry bounded regression
2. Log-space regression (`log(1 + IC50)`)

Each with:
- affine assay integration
- additive-only assay integration

### Distributional

1. Two-Hot cross-entropy
2. HL-Gauss cross-entropy

Each with:
- `K in {64, 128}`
- `MAX in {50k, 100k}`
- assay integration:
  - `D1-affine`
  - `D2-logit`
- sigma multipliers for HL-Gauss:
  - `0.5`
  - `0.75`
  - `1.5`

## Assay Context

Shared assay covariates for all methods:
- assay family
- prep
- geometry
- readout

For this experiment:
- use the same factorized assay covariates across all methods
- do not mix flat and factorized assay context in the 32-way sweep

## Logging Requirements

### Step-level

Per training step:
- `step`
- `epoch`
- `train_loss`
- `grad_norm_output_layer`

### Epoch-level validation

Per epoch:
- `val_loss`
- `val_auc_500`
- `val_srcc_5k`
- `val_mse_log_1k`
- calibration metrics for distributional runs:
  - `val_pit_ks_pvalue`
  - `val_coverage_90`
  - `val_entropy_error_srcc`

### Probe logging

Per probe, per epoch:
- `pred_ic50`
- and for distributional methods:
  - `pred_probs`
  - `pred_entropy`
  - `pred_5th_ic50`
  - `pred_95th_ic50`

### Final held-out metrics

On validation and test:
- overall loss
- Spearman
- Pearson
- RMSE in `log1p` or `log10` space
- `<=500 nM` classification metrics:
  - accuracy
  - balanced accuracy
  - precision
  - recall
  - F1
  - AUROC
  - AUPRC

## Files

New experiment package:

```text
scripts/distributional_ba/
  config.py
  train.py
  evaluate.py
  analyze.py
  heads/
    mhcflurry_head.py
    log_mse_head.py
    distributional.py
    twohot_head.py
    hlgauss_head.py
  assay_integration.py
```

Launcher:

```text
scripts/benchmark_distributional_ba_heads.py
```

Experiment family:

```text
experiments/YYYY-MM-DD_HHMM_codex_distributional-ba-heads-round1/
```

## Execution Plan

1. Implement shared dataset split / loader wrapper for the fixed broad contract.
2. Implement the head families and censored losses exactly as specified.
3. Add evaluation/logging helpers and tests.
4. Run a local smoke on one regression and one distributional condition.
5. Launch the 32-run Modal sweep.
6. Harvest metrics, generate required plots, and update the canonical log.

## Risks / Checks

- Need to keep backbone frozen/fixed across conditions.
- Need to ensure `mhcflurry` residuals are applied in encoded/logit space, not on clamped output.
- Need to ensure D1-affine censoring uses per-example adjusted edges.
- Need to save enough raw outputs to recompute metrics later.
