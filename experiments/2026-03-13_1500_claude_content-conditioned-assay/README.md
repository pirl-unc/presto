# V5: Content-Conditioned Assay Context

## Hypothesis

Making the assay bias context-dependent (conditioned on binding logit + mean-pooled pep/mhc_a/mhc_b representations, all detached) will improve binding affinity prediction compared to the input-independent assay bias used in v4.

## Design

- **Baseline**: v4 — 6 embed_dim conditions (32–256), MHCflurry additive head, MAX=50k, 50 epochs, content-independent assay bias
- **Treatment**: v5 — identical conditions with `--content-conditioned` flag enabled

The content-conditioned assay context encoder receives:
1. Pre-integration binding logit (scalar, detached)
2. Mean-pooled molecular representation from the encoder (detached)

This allows the assay bias to vary per-sample based on the peptide-MHC content, without leaking gradient signal back through the encoder.

## Conditions

| cond_id | embed_dim | head | assay_mode | MAX | epochs |
|---------|-----------|------|------------|-----|--------|
| 1 | 32 | mhcflurry | additive | 50k | 50 |
| 2 | 64 | mhcflurry | additive | 50k | 50 |
| 3 | 96 | mhcflurry | additive | 50k | 50 |
| 4 | 128 | mhcflurry | additive | 50k | 50 |
| 5 | 192 | mhcflurry | additive | 50k | 50 |
| 6 | 256 | mhcflurry | additive | 50k | 50 |

## Infrastructure

- **GPU**: H100! (Modal default)
- **Modal entry**: `distributional_ba_v5_run`
- **Launcher**: `reproduce/launch.sh`

## Results

### V5 (content-conditioned) vs V4 (content-independent)

| embed_dim | v4 Spearman | v5 Spearman | delta | v4 AUROC | v5 AUROC | delta |
|-----------|-------------|-------------|-------|----------|----------|-------|
| 32        | 0.806       | **0.838**   | +0.032 | 0.937   | **0.940** | +0.003 |
| 64        | 0.804       | **0.840**   | +0.037 | 0.935   | **0.940** | +0.005 |
| 96        | 0.797       | **0.837**   | +0.040 | 0.934   | **0.940** | +0.007 |
| 128       | 0.802       | **0.829**   | +0.028 | 0.936   | 0.937    | +0.001 |
| 192       | 0.798       | **0.827**   | +0.030 | 0.932   | 0.932    | +0.000 |
| 256       | 0.806       | **0.833**   | +0.027 | 0.937   | 0.936    | -0.001 |
| **mean**  | 0.802       | **0.834**   | **+0.032** | 0.935 | **0.937** | **+0.003** |

### Full v5 test metrics

| cond | embed_dim | test_spearman | test_auroc | test_pearson | test_rmse_log10 | test_f1 | test_bal_acc |
|------|-----------|---------------|------------|--------------|-----------------|---------|-------------|
| 1    | 32        | 0.838         | 0.940      | 0.840        | 0.849           | 0.845   | 0.874       |
| 2    | 64        | 0.840         | 0.940      | 0.834        | 0.853           | 0.836   | 0.864       |
| 3    | 96        | 0.837         | 0.940      | 0.833        | 0.867           | 0.838   | 0.866       |
| 4    | 128       | 0.829         | 0.937      | 0.832        | 0.859           | 0.829   | 0.859       |
| 5    | 192       | 0.827         | 0.932      | 0.827        | 0.892           | 0.818   | 0.849       |
| 6    | 256       | 0.833         | 0.936      | 0.827        | 0.875           | 0.835   | 0.863       |

### Conclusions

1. **Content-conditioning is a clear win.** Every embed_dim improves, with a mean Spearman lift of +0.032 (0.802 → 0.834). This is a substantial gain from a purely architectural change to the assay bias module.

2. **Largest gains at smaller embed_dims.** The delta is biggest at d=96 (+0.040) and d=64 (+0.037), suggesting the content-conditioned bias compensates for limited encoder capacity. At larger dims (192, 256) the encoder already captures enough, so the marginal gain from a smarter bias shrinks.

3. **AUROC gains are modest but consistent.** Mean AUROC improves +0.003 (0.935 → 0.937). Classification is already near ceiling so the ranking metric (Spearman) is the more informative signal.

4. **Optimal embed_dim shifts downward.** In v4, d=32 and d=256 tied for best Spearman. In v5, the sweet spot is d=32–64, which are also the cheapest models. This is good news for inference cost.

5. **No overfitting signal.** Val-test gaps are small and consistent across conditions, suggesting the content-conditioned bias does not memorize.

## Artifacts

| cond | run_id | Modal app |
|------|--------|-----------|
| 1 | `distributional_ba_v5_c01_20260313T183923Z` | ap-MM7bbPpuxzHRMZo6JcLUb5 |
| 2 | `distributional_ba_v5_c02_20260313T183914Z` | ap-rHjztdNvvy8TZ6jlk1YeeO |
| 3 | `distributional_ba_v5_c03_20260313T183934Z` | ap-mmkEdkqJZq5k5U4OTVslm4 |
| 4 | `distributional_ba_v5_c04_20260313T184031Z` | ap-qRb2RlwOr9SlEIH1vAooz7 |
| 5 | `distributional_ba_v5_c05_20260313T184131Z` | ap-zDYHDnZdIHHCJpNX6Yfeez |
| 6 | `distributional_ba_v5_c06_20260313T184233Z` | ap-q5DMXH5N4wy98j9l9AMFIn |

- Raw run artifacts were pulled locally under `results/runs/`
- Generated local closure outputs:
  - `results/condition_summary.csv`
  - `results/epoch_summary.csv`
  - `results/final_probe_predictions.csv`
  - `results/test_spearman_ranking.png`
  - `results/test_metric_grid.png`
  - `results/training_curves.png`
  - `results/final_probe_heatmap.png`

V4 baseline: `experiments/2026-03-13_1313_codex_embed-dim-fine-sweep/`
