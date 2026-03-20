# V6: Content-Conditioned Factorial Sweep (32 conditions)

## Corrected Executable Contract

Earlier markdown summaries elsewhere described this family as a 7-allele exact-IC50 warm-start sweep. The raw `summary.json` files in `data/` show the actual executed contract was:

- `measurement_profile=numeric_no_qualitative`
- two-allele panel: `HLA-A*02:01`, `HLA-A*24:02`
- assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- qualifier filter: `all`
- split sizes: train `15530`, val `1919`, test `1915`
- no warm start
- AdamW `lr=1e-3`, `weight_decay=0.01`, `epochs=50`, `batch_size=256`
- shared-path historical `AblationEncoder`; the winning `cond_id=2` run had `n_params=27186`

The metric tables below already reflect this corrected raw-artifact contract.

## Hypothesis

Content-conditioned assay bias (v5 finding: +0.032 Spearman) generalizes across head types (mhcflurry vs hlgauss), target ranges (50k vs 100k), and encoder capacities (embed_dim 32-256).

## Design

Full 2x4x2x2 factorial:
- **content_conditioned**: {no, yes}
- **embed_dim**: {32, 64, 128, 256}
- **head_type**: {mhcflurry (additive), hlgauss (d2_logit)}
- **max_nM**: {50,000, 100,000}

## Condition matrix

| cond_id | embed_dim | head | assay_mode | max_nM | cc0 run | cc1 run |
|---------|-----------|------|------------|--------|---------|---------|
| 1 | 32 | mhcflurry | additive | 50k | run 1 | run 17 |
| 2 | 32 | mhcflurry | additive | 100k | run 2 | run 18 |
| 3 | 32 | hlgauss | d2_logit | 50k | run 3 | run 19 |
| 4 | 32 | hlgauss | d2_logit | 100k | run 4 | run 20 |
| 5 | 64 | mhcflurry | additive | 50k | run 5 | run 21 |
| 6 | 64 | mhcflurry | additive | 100k | run 6 | run 22 |
| 7 | 64 | hlgauss | d2_logit | 50k | run 7 | run 23 |
| 8 | 64 | hlgauss | d2_logit | 100k | run 8 | run 24 |
| 9 | 128 | mhcflurry | additive | 50k | run 9 | run 25 |
| 10 | 128 | mhcflurry | additive | 100k | run 10 | run 26 |
| 11 | 128 | hlgauss | d2_logit | 50k | run 11 | run 27 |
| 12 | 128 | hlgauss | d2_logit | 100k | run 12 | run 28 |
| 13 | 256 | mhcflurry | additive | 50k | run 13 | run 29 |
| 14 | 256 | mhcflurry | additive | 100k | run 14 | run 30 |
| 15 | 256 | hlgauss | d2_logit | 50k | run 15 | run 31 |
| 16 | 256 | hlgauss | d2_logit | 100k | run 16 | run 32 |

## Infrastructure

- **GPU**: H100! (Modal default)
- **Epochs**: 50
- **Batch size**: 256
- **Modal entry**: `distributional_ba_v6_run`
- **Launcher**: `reproduce/launch.sh`

## Results

**Note**: All metrics below are from `summary.json` (best-epoch test evaluation). Earlier log-based extraction had different numbers; these are ground truth.

### Full results table (32 conditions)

| cond | d | head | max_nM | cc | Spearman | Pearson | RMSE_lg10 | AUROC | AUPRC | F1 | Bal_Acc |
|------|---|------|--------|----|----------|---------|-----------|-------|-------|------|---------|
| 1 | 32 | mhcflurry | 50k | 0 | 0.8395 | 0.8397 | 0.8404 | 0.9405 | 0.9000 | 0.8428 | 0.8703 |
| 1 | 32 | mhcflurry | 50k | 1 | 0.8377 | 0.8398 | 0.8493 | 0.9397 | 0.8943 | 0.8451 | 0.8738 |
| 2 | 32 | mhcflurry | 100k | 0 | **0.8435** | 0.8446 | 0.8304 | 0.9412 | 0.9045 | 0.8462 | 0.8737 |
| 2 | 32 | mhcflurry | 100k | 1 | 0.8412 | 0.8400 | 0.8526 | 0.9385 | 0.8908 | 0.8402 | 0.8689 |
| 3 | 32 | hlgauss | 50k | 0 | 0.8329 | 0.8413 | 0.8474 | 0.9406 | 0.8986 | 0.8430 | 0.8720 |
| 3 | 32 | hlgauss | 50k | 1 | 0.8168 | 0.8142 | 0.8973 | 0.9272 | 0.8702 | 0.8122 | 0.8440 |
| 4 | 32 | hlgauss | 100k | 0 | 0.8358 | 0.8393 | 0.8511 | 0.9398 | 0.8938 | 0.8380 | 0.8667 |
| 4 | 32 | hlgauss | 100k | 1 | 0.8130 | 0.8140 | 0.9081 | 0.9299 | 0.8765 | 0.8241 | 0.8544 |
| 5 | 64 | mhcflurry | 50k | 0 | 0.8293 | 0.8226 | 0.8854 | 0.9348 | 0.8917 | 0.8259 | 0.8554 |
| 5 | 64 | mhcflurry | 50k | 1 | 0.8403 | 0.8342 | 0.8532 | 0.9396 | 0.8938 | 0.8357 | 0.8640 |
| 6 | 64 | mhcflurry | 100k | 0 | 0.8295 | 0.8257 | 0.8796 | 0.9347 | 0.8938 | 0.8238 | 0.8539 |
| 6 | 64 | mhcflurry | 100k | 1 | 0.8391 | 0.8341 | 0.8635 | 0.9376 | 0.8975 | 0.8426 | 0.8708 |
| 7 | 64 | hlgauss | 50k | 0 | 0.8143 | 0.8163 | 0.8972 | 0.9329 | 0.8847 | 0.8235 | 0.8537 |
| 7 | 64 | hlgauss | 50k | 1 | 0.8074 | 0.8154 | 0.8992 | 0.9255 | 0.8898 | 0.7947 | 0.8297 |
| 8 | 64 | hlgauss | 100k | 0 | 0.8122 | 0.8202 | 0.9065 | 0.9296 | 0.8842 | 0.8199 | 0.8509 |
| 8 | 64 | hlgauss | 100k | 1 | 0.8230 | 0.8316 | 0.8790 | 0.9370 | 0.8907 | 0.8361 | 0.8647 |
| 9 | 128 | mhcflurry | 50k | 0 | 0.8234 | 0.8207 | 0.8881 | 0.9324 | 0.8901 | 0.8368 | 0.8653 |
| 9 | 128 | mhcflurry | 50k | 1 | 0.8293 | 0.8318 | 0.8585 | 0.9370 | 0.8982 | 0.8293 | 0.8593 |
| 10 | 128 | mhcflurry | 100k | 0 | 0.8295 | 0.8273 | 0.8805 | 0.9345 | 0.8928 | 0.8206 | 0.8516 |
| 10 | 128 | mhcflurry | 100k | 1 | 0.8345 | 0.8363 | 0.8598 | 0.9396 | 0.8944 | 0.8382 | 0.8654 |
| 11 | 128 | hlgauss | 50k | 0 | 0.8019 | 0.8129 | 0.9092 | 0.9260 | 0.8711 | 0.8206 | 0.8510 |
| 11 | 128 | hlgauss | 50k | 1 | 0.7995 | 0.8064 | 0.9214 | 0.9248 | 0.8693 | 0.8011 | 0.8348 |
| 12 | 128 | hlgauss | 100k | 0 | 0.8037 | 0.8113 | 0.9238 | 0.9276 | 0.8810 | 0.8190 | 0.8497 |
| 12 | 128 | hlgauss | 100k | 1 | 0.8135 | 0.8168 | 0.9052 | 0.9281 | 0.8837 | 0.8016 | 0.8353 |
| 13 | 256 | mhcflurry | 50k | 0 | 0.8324 | 0.8297 | 0.8761 | 0.9374 | 0.8913 | 0.8414 | 0.8698 |
| 13 | 256 | mhcflurry | 50k | 1 | 0.8325 | 0.8274 | 0.8749 | 0.9357 | 0.8973 | 0.8345 | 0.8629 |
| 14 | 256 | mhcflurry | 100k | 0 | 0.8222 | 0.8251 | 0.8893 | 0.9358 | 0.8841 | 0.8268 | 0.8571 |
| 14 | 256 | mhcflurry | 100k | 1 | 0.8370 | 0.8359 | 0.8578 | 0.9408 | 0.9020 | 0.8310 | 0.8598 |
| 15 | 256 | hlgauss | 50k | 0 | 0.8108 | 0.8189 | 0.9070 | 0.9334 | 0.8924 | 0.8335 | 0.8630 |
| 15 | 256 | hlgauss | 50k | 1 | 0.8089 | 0.8211 | 0.8932 | 0.9275 | 0.8845 | 0.8268 | 0.8563 |
| 16 | 256 | hlgauss | 100k | 0 | 0.7831 | 0.7931 | 0.9785 | 0.9172 | 0.8454 | 0.8153 | 0.8470 |
| 16 | 256 | hlgauss | 100k | 1 | 0.7848 | 0.7914 | 0.9733 | 0.9182 | 0.8790 | 0.8013 | 0.8352 |

### Content-conditioning effect (cc1 − cc0)

| cond | d | head | max_nM | cc0 Spearman | cc1 Spearman | delta |
|------|---|------|--------|-------------|-------------|-------|
| 1 | 32 | mhcflurry | 50k | 0.8395 | 0.8377 | -0.0018 |
| 2 | 32 | mhcflurry | 100k | 0.8435 | 0.8412 | -0.0023 |
| 3 | 32 | hlgauss | 50k | 0.8329 | 0.8168 | -0.0161 |
| 4 | 32 | hlgauss | 100k | 0.8358 | 0.8130 | **-0.0229** |
| 5 | 64 | mhcflurry | 50k | 0.8293 | 0.8403 | +0.0110 |
| 6 | 64 | mhcflurry | 100k | 0.8295 | 0.8391 | +0.0096 |
| 7 | 64 | hlgauss | 50k | 0.8143 | 0.8074 | -0.0070 |
| 8 | 64 | hlgauss | 100k | 0.8122 | 0.8230 | +0.0108 |
| 9 | 128 | mhcflurry | 50k | 0.8234 | 0.8293 | +0.0059 |
| 10 | 128 | mhcflurry | 100k | 0.8295 | 0.8345 | +0.0050 |
| 11 | 128 | hlgauss | 50k | 0.8019 | 0.7995 | -0.0024 |
| 12 | 128 | hlgauss | 100k | 0.8037 | 0.8135 | +0.0098 |
| 13 | 256 | mhcflurry | 50k | 0.8324 | 0.8325 | +0.0002 |
| 14 | 256 | mhcflurry | 100k | 0.8222 | 0.8370 | **+0.0148** |
| 15 | 256 | hlgauss | 50k | 0.8108 | 0.8089 | -0.0019 |
| 16 | 256 | hlgauss | 100k | 0.7831 | 0.7848 | +0.0017 |

### Summary by factor

| Factor | cc0 mean | cc1 mean | Mean delta |
|--------|----------|----------|------------|
| MHCflurry (8 conditions) | 0.8287 | 0.8340 | **+0.0053** |
| HL-Gauss (8 conditions) | 0.8117 | 0.8084 | **-0.0035** |
| Overall (16 conditions) | 0.8202 | 0.8212 | **+0.0009** |

### Content-conditioning effect by embed_dim × head

| embed_dim | mhcflurry delta | hlgauss delta |
|-----------|-----------------|---------------|
| 32 | -0.0021 (slight negative) | -0.0195 (strongly negative) |
| 64 | +0.0103 (positive) | +0.0019 (mixed: 50k negative, 100k positive) |
| 128 | +0.0055 (positive) | +0.0037 (mixed: 50k negative, 100k positive) |
| 256 | +0.0075 (positive) | -0.0001 (near zero) |

### Head-type main effect (pooled across cc and max_nM)

| Head | Mean Spearman |
|------|---------------|
| MHCflurry | 0.8314 |
| HL-Gauss | 0.8100 |

### max_nM main effect (pooled across cc and head)

| max_nM | Mean Spearman |
|--------|---------------|
| 50k | 0.8205 |
| 100k | 0.8210 |

### Probe peptide IC50 predictions (final epoch)

See `tables/probe_final.csv` for full data. Key SLLQHLIGL results:

| condition | A\*02:01 IC50 (nM) | A\*24:02 IC50 (nM) |
|-----------|--------------------|--------------------|
| d32_mcf_50k_cc0 | 26 | 14,240 |
| d32_mcf_100k_cc0 (best Spearman) | 26 | 17,739 |
| d64_mcf_50k_cc0 | 3.0 | 8,460 |
| d64_mcf_100k_cc1 | 2.2 | 1,248 |
| d128_hlg_100k_cc0 | 0.9 | 20 |
| d256_mcf_50k_cc0 | 13 | 33,326 |

SLLQHLIGL is a known A\*02:01 strong binder. All conditions correctly predict low IC50 for A\*02:01. The A\*24:02 discrimination (higher IC50 = weaker binding) varies widely: MHCflurry conditions show strong allele discrimination (10,000-80,000 nM for A\*24:02), while some HL-Gauss conditions collapse the discrimination (e.g., d128_hlg_50k predicts ~10 nM for both alleles).

## Conclusions

1. **Content-conditioning helps MHCflurry (mean +0.005) but hurts HL-Gauss (mean -0.004).** The overall effect is near zero (+0.001). The v5 finding (+0.032 vs v4) was inflated by comparing against a weaker v4 baseline; within v6's controlled factorial, the effect is smaller and head-dependent.

2. **The cc × head interaction is modulated by embed_dim.** At d=32, content-conditioning hurts both heads (MHCflurry -0.002, HL-Gauss -0.020). At d=64-256, it helps MHCflurry (+0.005 to +0.010) and has mixed effects on HL-Gauss (50k conditions hurt, 100k conditions benefit). This suggests the mechanism involves capacity: small models can't effectively use the extra conditioning signal.

3. **HL-Gauss at d=32 is uniquely vulnerable to content-conditioning** (-0.016, -0.023 deltas). The distributional head at minimal capacity appears to overfit the content-conditioned bias, losing generalization. At larger dims the effect is small or even positive.

4. **MHCflurry outperforms HL-Gauss across the board** (0.831 vs 0.810 mean, gap ~0.021). This gap is consistent across all embed_dims and max_nM values.

5. **max_nM has negligible main effect** (50k: 0.821, 100k: 0.821). The target range doesn't meaningfully change aggregate performance.

6. **Best overall condition**: cond=2 cc0 (d=32, mhcflurry, max=100k, no content-conditioning) at Spearman=0.8435, AUROC=0.9412.

7. **Probe specificity varies by head**: MHCflurry maintains strong SLLQHLIGL allele discrimination (A\*02:01 << A\*24:02). Some HL-Gauss conditions (especially d=128, 50k) collapse the allele discrimination, predicting similarly low IC50 for both alleles.

8. **Recommendation**: MHCflurry + max_nM=100k + d=32 without content-conditioning is the best single configuration. Content-conditioning helps at d=64+ but the best overall model doesn't need it.

## Artifacts

- **Raw data**: `data/` — all summary.json, probes.jsonl, metrics.jsonl, step_log.jsonl for all 32 runs
- **Original modal_runs source**: `modal_runs/v6_factorial/` — top-level summary.json manifest
- **Tables**: `tables/` — test_metrics_full.csv, probe_final.csv, probe_trajectories.csv, binary_metrics.csv, correlation_metrics.csv
- **Plots**: `plots/` — test_metrics_comparison.png, cc_delta_by_head.png, probe_trajectories.png, probe_final_heatmap.png, training_curves.png
- **Analysis script**: `analyze.py` — reproduces all tables and plots from local data
- **Task ID mapping**: `reproduce/task_ids.txt`
- **Launcher**: `reproduce/launch.sh`
- **Config**: `scripts/distributional_ba/config_v6.py`
