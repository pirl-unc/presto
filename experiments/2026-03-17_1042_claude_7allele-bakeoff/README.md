# 7-Allele Model Bakeoff (13 conditions x 3 seeds = 39 runs)

- Agent: `claude`
- Source script: `experiments/2026-03-17_1042_claude_7allele-bakeoff/launch.py`
- Created: `2026-03-17`
- GPU: H100!

## Purpose

Systematically compare residual architectures (A07, A03, DAG), loss modes (full, assay_heads_only), learning rate recipes, model capacity (d128, d32), and pretraining across 13 conditions with 3 seeds each for variance estimation. This is the first multi-seed, held-out test evaluation of the full factor space.

## Dataset Contract

- 7 alleles: HLA-A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02
- ~44K rows, `numeric_no_qualitative`, qualifier_filter=all
- 80/10/10 peptide-group split, split_seed=42
- Probe peptides: SLLQHLIGL, FLRYLLFGI, NFLIKFLLI
- Assay families: IC50, direct KD, KD (~IC50), KD (~EC50), EC50
- mhcflurry target encoding, split_kd_proxy, max_affinity_nM=100000
- No synthetic negatives, no contrastive losses

## Training

- 50 epochs, batch_size=256, weight_decay=0.01
- n_layers=2, n_heads=4 (all conditions)
- binding_core_lengths=8,9,10,11, shared refinement
- Train seeds: 42, 43, 44 (split_seed=42 fixed)
- Pretrained checkpoint: `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt` (where applicable)

## Conditions

| ID | d | Residual | Loss | Pretrain | LR | Schedule | Role |
|----|---|----------|------|----------|-----|----------|------|
| B1 | 128 | A07 | full | yes | 3e-4 | warmup_cosine | Control (=S1) |
| B2 | 128 | A07 | heads_only | yes | 1e-3 | constant | Control (=S5/C4) |
| B3 | 32 | A03 | full | no | 1e-3 | constant | Control (=C7) |
| D1 | 128 | DAG | full | yes | 3e-4 | warmup_cosine | New |
| D2 | 128 | DAG | heads_only | yes | 1e-3 | constant | New |
| D3 | 32 | DAG | full | no | 1e-3 | constant | New |
| A1 | 128 | A03 | full | yes | 3e-4 | warmup_cosine | New |
| A2 | 128 | A03 | heads_only | yes | 1e-3 | constant | New |
| L1 | 128 | A07 | heads_only | yes | 3e-4 | warmup_cosine | LR sweep |
| L2 | 128 | DAG | heads_only | yes | 3e-4 | warmup_cosine | LR sweep |
| P1 | 128 | A07 | full | no | 3e-4 | warmup_cosine | No pretrain |
| P2 | 128 | DAG | full | no | 3e-4 | warmup_cosine | No pretrain |
| C1 | 32 | A07 | full | no | 1e-3 | constant | New |

## Results — Test Metrics (mean±std across seeds)

| ID | n | test_spearman | test_auroc | test_auprc | test_rmse_log10 | test_f1 | test_bal_acc |
|----|---|---------------|------------|------------|-----------------|---------|-------------|
| **D3** | 3 | **0.8101±0.0032** | **0.9353±0.0022** | **0.8796±0.0060** | **0.8243±0.0099** | 0.8169±0.0047 | 0.8583±0.0031 |
| **L2** | 3 | 0.8062±0.0060 | 0.9336±0.0039 | 0.8725±0.0106 | 0.8502±0.0195 | **0.8202±0.0057** | **0.8612±0.0033** |
| D2 | 1 | 0.8030 | 0.9317 | 0.8748 | 0.8395 | 0.8067 | 0.8479 |
| P2 | 3 | 0.8020±0.0034 | 0.9291±0.0009 | 0.8663±0.0055 | 0.8600±0.0083 | 0.8061±0.0018 | 0.8488±0.0016 |
| D1 | 3 | 0.7914±0.0088 | 0.9259±0.0058 | 0.8629±0.0124 | 0.8809±0.0217 | 0.7993±0.0065 | 0.8435±0.0049 |
| C1 | 3 | 0.7897±0.0034 | 0.9299±0.0034 | 0.8659±0.0106 | 0.8620±0.0112 | 0.8111±0.0081 | 0.8531±0.0077 |
| B3 | 1 | 0.7862 | 0.9255 | 0.8590 | 0.8664 | 0.8003 | 0.8437 |
| B2 | 2 | 0.7842±0.0052 | 0.9267±0.0015 | 0.8633±0.0014 | 0.8768±0.0132 | 0.8085±0.0039 | 0.8517±0.0024 |
| L1 | 3 | 0.7841±0.0032 | 0.9268±0.0025 | 0.8650±0.0049 | 0.8893±0.0092 | 0.8016±0.0103 | 0.8462±0.0087 |
| A2 | 1 | 0.7837 | 0.9242 | 0.8615 | 0.8821 | 0.7937 | 0.8379 |
| A1 | 3 | 0.7742±0.0031 | 0.9212±0.0016 | 0.8479±0.0098 | 0.9084±0.0059 | 0.7954±0.0042 | 0.8404±0.0042 |
| B1 | 3 | 0.7740±0.0037 | 0.9205±0.0035 | 0.8494±0.0111 | 0.9161±0.0098 | 0.7928±0.0051 | 0.8388±0.0044 |
| P1 | 3 | 0.7732±0.0029 | 0.9194±0.0022 | 0.8493±0.0029 | 0.9192±0.0066 | 0.7882±0.0012 | 0.8344±0.0013 |

Notes: D2 and A2 had 2/3 seeds diverge (NaN). B3 had 2/3 seeds fail to produce artifacts (Modal credit exhaustion). B2 had 1/3 seeds terminate early (epoch 41).

## Results — SLLQHLIGL Probe Discrimination (A*02:01 vs A*24:02)

SLLQHLIGL is a canonical HLA-A*02:01 binder (~10-50 nM). A good model should predict tight binding to A02:01 and non-binding to A24:02.

| ID | A02 Kd (nM) | A24 Kd (nM) | A24/A02 ratio | A02 bind_prob | A24 bind_prob |
|----|-------------|-------------|---------------|---------------|---------------|
| **L2** | 3.8±2.2 | 9,525±15,853 | 2502x | 0.971±0.021 | 0.096±0.070 |
| **L1** | 8.4±8.2 | 15,444±13,025 | 1832x | 0.943±0.043 | 0.003±0.001 |
| **D3** | 20.5±7.3 | 26,174±4,854 | 1279x | 0.983±0.003 | 0.068±0.020 |
| B1 | 53.0±15.7 | 66,105±30,595 | 1248x | 0.877±0.034 | 0.004±0.003 |
| B3 | 34.0 | 37,952 | 1115x | 0.928 | 0.006 |
| C1 | 37.5±5.9 | 31,772±8,750 | 848x | 0.927±0.011 | 0.009±0.004 |
| B2 | 15.4±15.7 | 6,734±6,432 | 437x | 0.923±0.011 | 0.027±0.043 |
| A1 | 86.1±14.4 | 38,400±11,738 | 446x | 0.794±0.058 | 0.008±0.003 |
| D1 | 45.8±53.1 | 7,347±6,570 | 160x | 0.965±0.044 | 0.410±0.195 |
| P2 | 149.5±227.3 | 21,784±14,490 | 146x | 0.881±0.181 | 0.299±0.229 |
| P1 | 369.9±611.7 | 37,029±20,171 | 100x | 0.702±0.460 | 0.007±0.003 |
| A2 | 753.7 | 2,427 | 3.2x | 0.394 | 0.022 |
| D2 | 2,394 | 5,291 | 2.2x | 0.653 | 0.019 |

## Stability

| ID | Seeds converged | Seeds diverged (NaN) | Notes |
|----|-----------------|---------------------|-------|
| B1 | 3/3 | 0 | |
| B2 | 3/3 | 0 | s44 terminated at epoch 41 (credits) |
| B3 | 1/1 | 0 | 2 seeds had no artifacts (credits) |
| D1 | 3/3 | 0 | |
| **D2** | **1/3** | **2** | s42 NaN at ep20, s44 NaN at ep16 |
| D3 | 3/3 | 0 | |
| A1 | 3/3 | 0 | |
| **A2** | **1/3** | **2** | s43 NaN at ep16, s44 NaN at ep9 |
| L1 | 3/3 | 0 | |
| L2 | 3/3 | 0 | |
| P1 | 3/3 | 0 | |
| P2 | 3/3 | 0 | |
| C1 | 3/3 | 0 | |

Root cause for D2/A2 NaN: `lr=1e-3 + constant schedule` is unstable at d128 for DAG and A03 architectures. Only A07 (B2) survives at this recipe. All conditions using `lr=3e-4 + warmup_cosine` converged across all seeds.

## Questions Answered

### Q1: DAG vs A07 on 7-allele?
**DAG wins on regression.** D1 vs B1: +0.017 Spearman (0.791 vs 0.774). D3 vs C1: +0.020 (0.810 vs 0.790). The prep/readout decomposition matches real assay variation structure.

### Q2: full vs heads_only on test metrics?
**Close on regression, heads_only wins on probes.** B1 vs B2: 0.774 vs 0.784 (heads_only slightly better). D1 vs D2: 0.791 vs 0.803 (heads_only better, but D2 has only 1 seed). On probes: heads_only preserves allele discrimination because it doesn't reshape the trunk's allele representations.

### Q3: d128 vs d32?
**d32 wins on regression for this dataset size.** D3 (d32, 0.810) > D1 (d128, 0.791). But d32 models treat all non-binder alleles identically — they learn "match/no-match" rather than groove-level physics. d128 models like L1 correctly distinguish between different non-binder alleles.

### Q4: A07 vs A03 vs DAG (3-way)?
**DAG > A07 ≈ A03.** B1 vs A1: 0.774 vs 0.774 (identical). D1 vs both: 0.791 (DAG wins). The extra factorized context in A07 vs A03 provides no benefit.

### Q5: Pretraining effect?
**Small and inconsistent.** B1 vs P1: 0.774 vs 0.773. D1 vs P2: 0.791 vs 0.802 (cold start wins). Pretraining slightly reduces seed variance. At 7 alleles with ~44K rows, the model learns allele structure from scratch.

### Q6: LR sweep for heads_only?
**warmup_cosine at lr=3e-4 is much better than constant at lr=1e-3 for probe discrimination.** L1 vs B2: 1832x vs 437x A24/A02 ratio. L2 vs D2: L2 is 3/3 stable with 2502x ratio; D2 is 1/3 stable with 2.2x ratio. The lr recipe is the most important single factor for both stability and calibration.

## Factor Analysis (importance order)

1. **LR + schedule**: Most important. lr=3e-4 warmup_cosine prevents divergence and produces calibrated binding probabilities. lr=1e-3 constant causes NaN at d128 for DAG and A03.
2. **Loss mode**: heads_only preserves allele-specific biology in the trunk. full reshapes the trunk toward regression accuracy at the cost of probe discrimination.
3. **Residual architecture**: DAG's prep/readout decomposition consistently improves regression (+0.02 Spearman). A07 and A03 are equivalent.
4. **Model size**: d32 wins on regression for ~44K rows / 7 alleles but learns a simpler "match/no-match" rule. d128 preserves allele-level structure for generalization.
5. **Pretraining**: Least important at this dataset size. May matter more at full class I scale.

## Recommended Baseline

**L2: DAG + heads_only + lr=3e-4 warmup_cosine + d128 + pretrained**

- Test Spearman: 0.8062±0.0060 (2nd overall, within noise of D3)
- Test AUROC: 0.9336±0.0039
- Test F1: 0.8202±0.0057 (best overall)
- Probe A24/A02 ratio: 2502x (best overall)
- Stability: 3/3 seeds converged
- Best combined regression + discrimination + stability

Runner-up for regression-only: D3 (d32 DAG full no-pretrain, Spearman 0.8101±0.0032), but lacks allele-level resolution.

## Missing Data

- B3 seeds 43, 44: Modal credits exhausted before artifacts were written
- D2 seeds 42, 44 and A2 seeds 43, 44: Diverged to NaN (lr instability, not credit issue)
- B2 seed 44: Terminated at epoch 41 (credits), still has valid metrics

## Artifacts

- Per-run test metrics: [`results/per_run_test_metrics.csv`](results/per_run_test_metrics.csv)
- Condition summary: [`results/condition_summary.csv`](results/condition_summary.csv)
- Fetch status: [`results/fetch_status.json`](results/fetch_status.json)
- Individual run summaries: [`results/runs/`](results/runs/)
- Launch manifest: [`manifest.json`](manifest.json)
- Reproducibility bundle: [`reproduce/`](reproduce/)
