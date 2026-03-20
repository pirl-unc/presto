# EXP-20: Main-Path Rebuild of EXP-16

## Goal

Restore the exact historical EXP-16 winner as a first-class shared-code positive control, then rerun the full v6 factorial on the actual executable contract with a modern shared-path `groove` backend as the challenger baseline.

## Provenance

- Agent: `codex`
- Source script: `code/launch.py`
- Git commit: `e17aa284c89767d1b9827753dd7dd26c5750171e` (`main`, dirty)
- Repro bundle:
  - `reproduce/launch.sh`
  - `reproduce/launch.json`
  - `reproduce/source/benchmark_distributional_ba_v6_backbone_compare.py` (historical launch snapshot)

## Corrected Executable Contract

This rebuild uses the raw EXP-16 winner contract from `2026-03-13_1600_claude_v6-factorial-32/data/distributional_ba_v6_c02_cc0_20260313T202238Z/summary.json`, not the older 7-allele warm-start markdown description.

- Source: `data/merged_deduped.tsv`
- Alleles: `HLA-A*02:01`, `HLA-A*24:02`
- Measurement profile: `numeric_no_qualitative`
- Assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- Qualifier filter: `all`
- Split: `peptide_group_80_10_10_seed42`
- Split sizes: train `15530`, val `1919`, test `1915`
- Batch size: `256`
- Epochs: `50`
- Optimizer: `AdamW`
- Learning rate: `1e-3`
- Weight decay: `0.01`
- Warm start: none
- Requested Modal GPU: `H100!`

## Shared-Path Fixes

- Restored the historical EXP-16 encoder as `encoder_backbone=historical_ablation` in `scripts/distributional_ba/encoders.py`
- Kept the current shared-path encoder as `encoder_backbone=groove`
- Added explicit `--encoder-backbone` and `--measurement-profile` controls to the shared trainer and Modal wrapper
- Added regression tests that freeze the historical positive-control contract at `n_params=27186`

## Verification Before Sweep

- `python -m pytest tests/test_distributional_ba.py -q`
  - result: `64 passed`
- Local dry-runs:
  - historical positive control (`cond_id=2`, `historical_ablation`) completed with `n_params=27186`
  - groove challenger (`cond_id=2`, `groove`) completed with `n_params=54803`
- Modal smoke runs:
  - `exp16-v6-smoke-ablation` confirmed the restored positive control path on Modal
  - `exp16-v6-smoke-groove` confirmed the groove path on Modal

## Sweep Design

- Fixed v6 condition matrix: `cond_id=1..16`
- `content_conditioned in {false, true}`
- `encoder_backbone in {historical_ablation, groove}`
- Total conditions: `64`

## Execution Notes

- Initial detached launch completed for all `64` conditions.
- One run failed for infrastructure reasons rather than model reasons:
  - original app `ap-xqlkN8TwKDBASBMfPoQyvA`
  - run id `dist-ba-v6-mainpath-groove-c08-cc0`
  - failure: GPU Xid `66` / `torch.AcceleratorError: CUDA-capable device(s) is/are busy or unavailable`
  - relaunch app `ap-uXKnZE0Q5GQpSDSI1KXB8M` completed normally

## Positive-Control Parity

The restored historical backend exactly reproduced the original EXP-16 winner.

- Original raw artifact: `2026-03-13_1600_claude_v6-factorial-32/data/distributional_ba_v6_c02_cc0_20260313T202238Z/summary.json`
- Rebuilt shared-path run: `results/runs/dist-ba-v6-mainpath-ablation-c02-cc0/summary.json`
- Exact matches:
  - config: `cond_id=2`, `content_conditioned=false`, `embed_dim=32`, `head_type=mhcflurry`, `max_nM=100000`
  - data contract: same alleles and split sizes
  - parameter count: `27186`
  - held-out test metrics: Spearman `0.8435156345`, Pearson `0.8445964456`, RMSE_log10 `0.8303825259`, AUROC `0.9412032366`, AUPRC `0.9044540524`, F1 `0.8462029317`

## Results

### Backend Summary

| encoder_backbone | runs | mean test Spearman | max test Spearman | mean test AUROC | mean test RMSE_log10 |
|------------------|------|--------------------|-------------------|-----------------|----------------------|
| `groove` | 32 | `0.8221` | `0.8463` | `0.9351` | `0.8863` |
| `historical_ablation` | 32 | `0.8220` | `0.8435` | `0.9333` | `0.8855` |

Groove won `19 / 32` matched condition pairs by test Spearman, with mean paired delta `+0.00015`, so the family-level difference is effectively negligible even though the best single groove condition is stronger.

### Best Conditions

| Rank | run_id | backbone | label | test Spearman | test AUROC | test AUPRC | test RMSE_log10 |
|------|--------|----------|-------|---------------|------------|------------|-----------------|
| 1 | `dist-ba-v6-mainpath-groove-c01-cc0` | `groove` | `c01_mhcflurry_additive_max50k_d32` | `0.8463` | `0.9492` | `0.9174` | `0.8347` |
| 2 | `dist-ba-v6-mainpath-groove-c02-cc0` | `groove` | `c02_mhcflurry_additive_max100k_d32` | `0.8443` | `0.9455` | `0.9095` | `0.8455` |
| 3 | `dist-ba-v6-mainpath-ablation-c02-cc0` | `historical_ablation` | `c02_mhcflurry_additive_max100k_d32` | `0.8435` | `0.9412` | `0.9045` | `0.8304` |
| 4 | `dist-ba-v6-mainpath-groove-c02-cc1` | `groove` | `c02_mhcflurry_additive_max100k_d32` | `0.8415` | `0.9452` | `0.9089` | `0.8409` |

### d32 MHCflurry Comparison

| run_id | backbone | cc | max_nM | n_params | test Spearman | test AUROC | test RMSE_log10 |
|--------|----------|----|--------|----------|---------------|------------|-----------------|
| `dist-ba-v6-mainpath-groove-c01-cc0` | `groove` | no | `50k` | `54803` | `0.8463` | `0.9492` | `0.8347` |
| `dist-ba-v6-mainpath-groove-c02-cc0` | `groove` | no | `100k` | `54803` | `0.8443` | `0.9455` | `0.8455` |
| `dist-ba-v6-mainpath-ablation-c02-cc0` | `historical_ablation` | no | `100k` | `27186` | `0.8435` | `0.9412` | `0.8304` |
| `dist-ba-v6-mainpath-ablation-c01-cc0` | `historical_ablation` | no | `50k` | `27186` | `0.8395` | `0.9405` | `0.8404` |

Relative to the exact historical positive control, the best groove model improves test Spearman by `+0.0028`, AUROC by `+0.0080`, and AUPRC by `+0.0130`, while giving up some RMSE/F1/balanced-accuracy.

## Winner / Preferred Condition

Use `dist-ba-v6-mainpath-groove-c01-cc0` as the new shared-code baseline best model:

- `encoder_backbone=groove`
- `cond_id=1`
- `label=c01_mhcflurry_additive_max50k_d32`
- no content conditioning
- `n_params=54803`
- held-out test: Spearman `0.8463010192`, Pearson `0.8450502753`, RMSE_log10 `0.8346529603`, AUROC `0.9491695762`, AUPRC `0.9174090624`

Keep `dist-ba-v6-mainpath-ablation-c02-cc0` as the positive control because it is now an exact shared-path reproduction of the original EXP-16 winner.

## Artifacts

- Topline tables:
  - `results/condition_summary.csv`
  - `results/backend_summary.csv`
  - `results/backend_condition_comparison.csv`
- Per-epoch / probe artifacts:
  - `results/epoch_summary.csv`
  - `results/final_probe_predictions.csv`
- Plots:
  - `results/test_spearman_ranking.png`
  - `results/test_metric_grid.png`
  - `results/training_curves.png`
  - `results/final_probe_heatmap.png`
  - `results/backend_condition_heatmap.png`
  - `results/backend_metric_bars.png`
- Raw run artifacts:
  - `results/runs/`
- Launch records:
  - `manifest.json`
  - `launch_logs/`

## Takeaways

1. The shared main path can now reproduce the original EXP-16 winner exactly rather than only approximately.
2. The strongest new shared-code baseline is no longer the historical EXP-16 winner; it is the groove-backed `d32 + mhcflurry + max_nM=50k + no content-conditioning` condition.
3. `mhcflurry` remains the dominant head family. The top four conditions are all `mhcflurry`.
4. Content conditioning is still not needed for the best single model. For groove it is slightly negative on average (`0.8231` for `cc0` vs `0.8212` for `cc1`); for historical ablation it is slightly positive (`0.8215` vs `0.8224`).
