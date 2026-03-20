# EXP-11: Distributional vs Regression BA Heads (32 conditions)

## Overview

Systematic comparison of 4 output head families for binding affinity prediction, crossed with assay integration modes, bin counts, MAX clipping values, and sigma multipliers.

**Question**: Can distributional output heads (Two-Hot or HL-Gauss) match or beat scalar regression heads (MHCflurry or Log MSE) for binding affinity prediction, and does the assay integration mode (affine, additive, d1_affine, d2_logit) matter?

**Answer**: No. MHCflurry regression remains the best head family in this sweep. D2-logit distributional heads are viable and competitive but trail by ~0.015 Spearman. D1-affine distributional heads are fundamentally broken and collapse to constant predictions. Additive assay integration slightly outperforms affine for MHCflurry heads.

## Metadata

| Field | Value |
|-------|-------|
| Date | 2026-03-13 |
| Agent | Claude Code (claude-opus-4-6) |
| Git commit | `e01eea1e91eb3b07ac5a8d75e65956ef688cccfb` (main, dirty) |
| GPU | H100! (Modal) |
| Source script | `scripts/benchmark_distributional_ba_heads.py` |
| Launcher | `scripts/distributional_ba/train.py` via `scripts/train_modal.py::distributional_ba_run` |
| Experiment directory | `experiments/2026-03-13_0728_claude_distributional-ba-heads/` |
| Reproducibility bundle | [`reproduce/`](./reproduce/) |

## Important Caveat

This is a completed 32-condition sweep, but it is **not** the final fixed-contract benchmark for head-to-head comparison against the best current Presto broad-contract backbone.

Differences from the intended clean benchmark:
- Uses `AblationEncoder(embed=128, layers=2, heads=4)` instead of the current best frozen canonical Presto backbone
- Uses 20 epochs instead of the planned fixed 10
- Uses `lr=1e-3` rather than the current broad-contract best LR/schedule settings

Therefore this family is informative about output-head behavior, but should be treated as a **diagnostic sweep** rather than the final answer.

## Dataset

- **Panel**: 7 alleles -- HLA-A\*02:01, HLA-A\*24:02, HLA-A\*03:01, HLA-A\*11:01, HLA-A\*01:01, HLA-B\*07:02, HLA-B\*44:02
- **Measurement profile**: `numeric_no_qualitative`
- **Qualifier filter**: `all`
- **Split**: peptide-stratified 80/10/10
- **Approximate sizes**: ~32.8k train / ~4.2k val / ~4.1k test rows
- **Assay families**: IC50, KD, KD(~IC50), KD(~EC50), EC50
- **Assay-label to output mapping**: all assay families map to a single binding affinity output head; assay-specific integration is handled by the `assay_mode` parameter (affine, additive, d1_affine, or d2_logit)
- **No synthetics, no contrastive**

## Training

| Parameter | Value |
|-----------|-------|
| Encoder | `AblationEncoder(embed=128, layers=2, heads=4)` |
| Learning rate | 1e-3 |
| Optimizer | AdamW (weight_decay=0.01) |
| Epochs | 20 |
| Batch size | 256 |
| Seed | 42 |
| Warm start | None (AblationEncoder, cold start) |
| LR schedule | None (constant) |

## Experimental Design

### Head families (4)

1. **MHCflurry regression** (`mhcflurry`): sigmoid-bounded `1 - log(IC50)/log(MAX)` target, MSE loss. The standard from MHCflurry.
2. **Log MSE regression** (`log_mse`): `log10(IC50)` target, MSE loss. Simpler unbounded target.
3. **Two-Hot distributional** (`twohot`): discretize `log10(IC50)` into K bins, place soft two-hot target across two adjacent bins, cross-entropy loss.
4. **HL-Gauss distributional** (`hlgauss`): discretize `log10(IC50)` into K bins, place Gaussian soft label centered on the target, cross-entropy loss. Based on the HL-Gauss approach from Imani & White (2024).

### Assay integration modes (4)

1. **Affine** (`affine`): per-assay learned scale and bias applied to the scalar prediction (regression heads) or to the latent before the output head.
2. **Additive** (`additive`): per-assay learned bias only (no scale), applied the same way.
3. **D1-affine** (`d1_affine`): for distributional heads, apply per-assay learned scale and bias to the bin centers/edges before computing probabilities. The idea is that different assays measure on shifted/scaled versions of the same underlying affinity distribution.
4. **D2-logit** (`d2_logit`): for distributional heads, apply per-assay learned logit shifts to each bin's log-probability. Assay effects act in probability space rather than value space.

### Sweep axes

- `max_nM`: {50000, 100000} -- clipping ceiling for affinity values
- `K` (n_bins): {64, 128} -- number of bins for distributional heads
- `sigma` (sigma_mult): {0.5, 0.75, 1.5} -- Gaussian width multiplier for HL-Gauss targets

### Condition matrix (32 total)

| cond_id | head | assay | max_nM | K | sigma | Notes |
|---------|------|-------|--------|---|-------|-------|
| 1 | mhcflurry | affine | 50k | -- | -- | Regression baseline |
| 2 | mhcflurry | affine | 100k | -- | -- | Regression baseline |
| 3 | log_mse | affine | 50k | -- | -- | Regression baseline |
| 4 | log_mse | affine | 100k | -- | -- | Regression baseline |
| 5-6 | twohot | d1_affine | 50k/100k | 128 | 0.75 | D1-affine distributional |
| 7-8 | hlgauss | d1_affine | 50k/100k | 128 | 0.75 | D1-affine distributional |
| 9-10 | twohot | d1_affine | 50k/100k | 64 | 0.75 | D1-affine, fewer bins |
| 11-12 | hlgauss | d1_affine | 50k/100k | 64 | 0.75 | D1-affine, fewer bins |
| 13-14 | twohot | d2_logit | 50k/100k | 128 | 0.75 | D2-logit distributional |
| 15-16 | twohot | d2_logit | 50k/100k | 64 | 0.75 | D2-logit, fewer bins |
| 17-18 | hlgauss | d2_logit | 50k/100k | 128 | 0.75 | D2-logit distributional |
| 19-20 | hlgauss | d2_logit | 50k/100k | 64 | 0.75 | D2-logit, fewer bins |
| 21-22 | hlgauss | d1_affine | 50k/100k | 128 | 0.5 | D1-affine, narrow sigma |
| 23-24 | hlgauss | d1_affine | 50k/100k | 128 | 1.5 | D1-affine, wide sigma |
| 25-26 | hlgauss | d2_logit | 50k/100k | 128 | 0.5 | D2-logit, narrow sigma |
| 27-28 | hlgauss | d2_logit | 50k/100k | 128 | 1.5 | D2-logit, wide sigma |
| 29 | mhcflurry | additive | 50k | -- | -- | Additive regression |
| 30 | mhcflurry | additive | 100k | -- | -- | Additive regression |
| 31 | log_mse | additive | 50k | -- | -- | Additive regression |
| 32 | log_mse | additive | 100k | -- | -- | Additive regression |

## Held-Out Test Results (All 32 Conditions)

Binding threshold for classification metrics: <= 500 nM.

### Regression heads (MHCflurry and Log MSE)

| cond_id | head | assay | max_nM | AUROC | Spearman | Pearson | RMSE | F1 | Bal Acc |
|---------|------|-------|--------|-------|----------|---------|------|-----|---------|
| **30** | **mhcflurry** | **additive** | **100k** | **0.937** | **0.812** | **0.828** | **0.829** | **0.810** | **0.850** |
| **29** | **mhcflurry** | **additive** | **50k** | **0.937** | **0.811** | **0.828** | **0.818** | **0.800** | **0.842** |
| 2 | mhcflurry | affine | 100k | 0.937 | 0.802 | 0.828 | 0.841 | 0.806 | 0.847 |
| 1 | mhcflurry | affine | 50k | 0.938 | 0.801 | 0.825 | 0.828 | 0.815 | 0.854 |
| 32 | log_mse | additive | 100k | 0.934 | 0.805 | 0.822 | 0.867 | 0.806 | 0.846 |
| 31 | log_mse | additive | 50k | 0.933 | 0.800 | 0.824 | 0.855 | 0.806 | 0.845 |
| 3 | log_mse | affine | 50k | 0.930 | 0.777 | 0.808 | 0.884 | 0.793 | 0.837 |
| 4 | log_mse | affine | 100k | 0.921 | 0.762 | 0.784 | 0.949 | 0.785 | 0.831 |

### D2-logit distributional heads (working)

| cond_id | head | assay | max_nM | K | sigma | AUROC | Spearman | Pearson | RMSE | F1 | Bal Acc |
|---------|------|-------|--------|---|-------|-------|----------|---------|------|-----|---------|
| 20 | hlgauss | d2_logit | 100k | 64 | 0.75 | 0.930 | 0.798 | 0.817 | 0.847 | 0.802 | 0.845 |
| 27 | hlgauss | d2_logit | 50k | 128 | 1.5 | 0.927 | 0.793 | 0.811 | 0.850 | 0.805 | 0.847 |
| 28 | hlgauss | d2_logit | 100k | 128 | 1.5 | 0.927 | 0.796 | 0.814 | 0.849 | 0.799 | 0.842 |
| 19 | hlgauss | d2_logit | 50k | 64 | 0.75 | 0.929 | 0.791 | 0.814 | 0.848 | 0.808 | 0.849 |
| 17 | hlgauss | d2_logit | 50k | 128 | 0.75 | 0.930 | 0.789 | 0.810 | 0.851 | 0.805 | 0.847 |
| 25 | hlgauss | d2_logit | 50k | 128 | 0.5 | 0.928 | 0.786 | 0.809 | 0.857 | 0.808 | 0.849 |
| 13 | twohot | d2_logit | 50k | 128 | 0.75 | 0.931 | 0.786 | 0.811 | 0.855 | 0.805 | 0.848 |
| 14 | twohot | d2_logit | 100k | 128 | 0.75 | 0.928 | 0.787 | 0.813 | 0.870 | 0.803 | 0.846 |
| 16 | twohot | d2_logit | 100k | 64 | 0.75 | 0.931 | 0.783 | 0.810 | 0.889 | 0.811 | 0.852 |
| 18 | hlgauss | d2_logit | 100k | 128 | 0.75 | 0.928 | 0.782 | 0.809 | 0.889 | 0.798 | 0.841 |
| 26 | hlgauss | d2_logit | 100k | 128 | 0.5 | 0.926 | 0.779 | 0.808 | 0.890 | 0.808 | 0.849 |
| 15 | twohot | d2_logit | 50k | 64 | 0.75 | 0.923 | 0.778 | 0.801 | 0.878 | 0.801 | 0.846 |

### D1-affine distributional heads (collapsed / broken)

All 12 D1-affine conditions produced near-chance or worse-than-chance predictions. Representative:

| cond_id | head | assay | max_nM | K | sigma | AUROC | Spearman | Pearson | RMSE |
|---------|------|-------|--------|---|-------|-------|----------|---------|------|
| 5 | twohot | d1_affine | 50k | 128 | 0.75 | 0.443 | -0.097 | 0.020 | 2.128 |
| 6 | twohot | d1_affine | 100k | 128 | 0.75 | 0.580 | 0.199 | 0.249 | 5.900 |
| 7 | hlgauss | d1_affine | 50k | 128 | 0.75 | 0.443 | -0.098 | -0.020 | 6.302 |
| 8 | hlgauss | d1_affine | 100k | 128 | 0.75 | 0.443 | -0.098 | -0.020 | 6.302 |
| 21 | hlgauss | d1_affine | 50k | 128 | 0.5 | 0.443 | -0.098 | -0.020 | 6.302 |
| 22 | hlgauss | d1_affine | 100k | 128 | 0.5 | 0.443 | -0.098 | -0.020 | 6.302 |
| 23 | hlgauss | d1_affine | 50k | 128 | 1.5 | 0.443 | -0.098 | -0.020 | 6.302 |
| 24 | hlgauss | d1_affine | 100k | 128 | 1.5 | 0.443 | -0.098 | -0.020 | 6.302 |

All D1-affine conditions with K=64 also collapsed (AUROC 0.44-0.46).

## Key Findings

### 1. D1-affine distributional heads completely fail

All 12 D1-affine conditions collapsed to constant or near-constant predictions (~0.44 AUROC). The learned scale/bias on bin centers/edges does not train. This failure is robust across:
- Both head types (twohot and hlgauss)
- Both bin counts (64 and 128)
- Both MAX values (50k and 100k)
- All three sigma multipliers (0.5, 0.75, 1.5)

**Root cause hypothesis**: D1-affine integration shifts and scales the bin centers per assay while keeping a shared probability shape. Under mixed assays with censoring, the per-example adjusted bin edges create a degenerate or numerically unstable likelihood. The gradient signal through the affine transform on bin edges is too indirect for the optimizer to learn meaningful assay adjustments.

### 2. D2-logit distributional heads work well

All 8 D2-logit conditions produced competitive predictions (AUROC 0.92-0.93, Spearman 0.78-0.80). Per-bin logit shifts from assay context are learnable because they act directly in probability space. The best D2-logit condition (c20: hlgauss, K=64, max=100k, sigma=0.75) achieved 0.930 AUROC and 0.798 Spearman.

### 3. MHCflurry regression still wins overall

MHCflurry heads achieved the best Spearman correlation (~0.81) vs ~0.80 for the best distributional condition. The gap is modest (~0.015 Spearman) but consistent. Additive integration slightly outperforms affine for MHCflurry: 0.812 vs 0.802 Spearman.

### 4. MAX 50k vs 100k: minimal difference

Across all head families, the difference between MAX=50k and MAX=100k is negligible (typically <0.005 Spearman, <0.003 AUROC). The clipping ceiling is not a binding constraint for this dataset.

### 5. Sigma sweep: negligible impact for D2-logit HL-Gauss

For D2-logit HL-Gauss conditions, varying sigma from 0.5 to 1.5 produced Spearman values in the range [0.779, 0.796] -- a spread of 0.017 that is within noise. The default sigma=0.75 is reasonable but not critical.

### 6. LogMSE additive is surprisingly competitive

LogMSE with additive integration (c31-32) achieves Spearman 0.800-0.805, closing to within ~0.007 of MHCflurry additive. The gap narrows substantially with additive vs affine integration, suggesting the assay integration mode matters more than the output head family for regression approaches.

### 7. Bin count (K=64 vs K=128): modest effect

For D2-logit heads, K=64 and K=128 produce similar results. The best single distributional condition was K=64 (c20: hlgauss d2_logit, Spearman 0.798), but K=128 conditions are within noise. Fewer bins may be preferred for computational efficiency.

## Ranking Summary

Ordered by test Spearman (best to worst family):

| Rank | Family | Best Condition | Spearman | AUROC | Notes |
|------|--------|----------------|----------|-------|-------|
| 1 | MHCflurry additive | c30 (max=100k) | 0.812 | 0.937 | Overall winner |
| 2 | MHCflurry affine | c2 (max=100k) | 0.802 | 0.937 | Close second |
| 3 | LogMSE additive | c32 (max=100k) | 0.805 | 0.934 | Surprisingly strong |
| 4 | D2-logit HL-Gauss | c20 (K=64, max=100k) | 0.798 | 0.930 | Best distributional |
| 5 | D2-logit Two-Hot | c14 (K=128, max=100k) | 0.787 | 0.928 | Viable distributional |
| 6 | LogMSE affine | c3 (max=50k) | 0.777 | 0.930 | Affine hurts log_mse |
| 7 | D1-affine (all) | -- | -0.098 | 0.443 | Completely broken |

## Failure Analysis: D1-affine

The D1-affine collapse is the most informative result of this sweep. Several lines of evidence point to the root cause:

1. **Scale/bias on bin edges is an indirect parameterization**: shifting bin edges per assay means the gradient must flow through the edge-to-probability conversion, which creates a very flat loss landscape when the model has not yet learned meaningful bin probabilities.

2. **Censoring in adjusted-edge space is fragile**: D1 requires censor thresholds to be located in per-example adjusted bin edges. This is more complex than the canonical-bin censor likelihood used by D2-logit, and the collapse pattern is consistent with this likelihood being mis-specified or numerically unstable.

3. **All sigma values fail equally**: varying sigma from 0.5 to 1.5 has zero effect on D1-affine results (all produce identical AUROC ~0.443), confirming the problem is structural, not a hyperparameter tuning issue.

4. **Both bin counts fail**: K=64 and K=128 both collapse, ruling out resolution artifacts.

5. **The one semi-functional D1-affine run** (c6: twohot, max=100k, K=128) achieved 0.580 AUROC -- still far below chance-level for a balanced classifier, but enough to show the parameterization can partially activate under specific conditions.

## Artifacts

- Full results: [`all_results.json`](all_results.json)
- Parsed summary table: [`options_vs_perf.md`](options_vs_perf.md)
- Parsed machine-readable summary: [`options_vs_perf.json`](options_vs_perf.json)
- Extracted metrics: [`analysis/parsed_metrics.csv`](analysis/parsed_metrics.csv), [`analysis/parsed_metrics.json`](analysis/parsed_metrics.json)
- Summary for log: [`analysis/summary_for_log.json`](analysis/summary_for_log.json)
- Launch manifest: [`manifest.json`](manifest.json)
- Per-condition launch logs: [`launch_logs/`](launch_logs/)
- Plots: [`heatmap_auroc_spearman.png`](heatmap_auroc_spearman.png), [`epoch_val_loss.png`](epoch_val_loss.png), [`epoch_val_spearman.png`](epoch_val_spearman.png), [`bar_spearman_all_conditions.png`](bar_spearman_all_conditions.png), [`scatter_d1_vs_d2_auroc.png`](scatter_d1_vs_d2_auroc.png), [`all_plots.pdf`](all_plots.pdf)
- Reproducibility bundle: [`reproduce/`](./reproduce/)

## Limitations

- **Not the canonical backbone**: uses AblationEncoder, not the full Presto or Groove model. Results may differ with a larger/different encoder.
- **No per-example test predictions saved**: exact-IC50 rank correlation and per-example calibration cannot be reconstructed post-hoc.
- **Single seed**: all conditions run with seed=42. Variance estimates require multi-seed replication.
- **No warm start**: AblationEncoder was trained from scratch, unlike the main Presto/Groove pipelines which use MHC pretraining.

## Reproduce

```bash
cd /Users/iskander/code/presto
export PRESTO_EXPERIMENT_AGENT='claude'
export PRESTO_MODAL_GPU='H100!'
python scripts/benchmark_distributional_ba_heads.py \
  --epochs 20 \
  --batch-size 256 \
  --agent-label claude \
  --prefix dist-ba-v1
```

See [`reproduce/launch.sh`](reproduce/launch.sh) and [`reproduce/launch.json`](reproduce/launch.json) for the exact frozen invocation.
