# Clean Distributional vs Regression BA Heads

Status: completed

## Runtime Boundary

- Self-contained package: `code/distributional_ba/`
- Self-contained launcher: `code/launch.py`
- Modal shim only: `scripts/train_modal.py::distributional_ba_clean_run`
- Shared `scripts/distributional_ba/` is not part of this experiment runtime.

## Reproducibility

- Agent/model: Codex / GPT-5
- Git commit: `e01eea1e91eb3b07ac5a8d75e65956ef688cccfb` (`main`, dirty)
- Repro bundle:
  - `reproduce/launch.sh`
  - `reproduce/launch.json`
  - `reproduce/source/launch.py`
- Requested Modal GPU: `H100!`
- Observed hardware evidence:
  - saved Modal app logs prove `device=cuda`
  - the stopped-app history available post-run did not preserve exact GPU SKU or memory in the harvested logs

## Fixed Contract

- Source data: `data/merged_deduped.tsv`
- Panel: `HLA-A*02:01`, `HLA-A*24:02`, `HLA-A*03:01`, `HLA-A*11:01`, `HLA-A*01:01`, `HLA-B*07:02`, `HLA-B*44:02`
- Assay families included:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- Qualifier policy: `qualifier_filter=all`
- Split policy: deterministic peptide-group split, seed `42`, `80/10/10`
- Effective split sizes:
  - train `32,805`
  - val `4,184`
  - test `4,060`
- Held-out exact-value subset sizes used for Spearman/Pearson/RMSE/<=500 nM metrics:
  - val exact `3,033`, censored `1,151`
  - test exact `2,983`, censored `1,077`
- Training contract:
  - cold start fixed backbone (`FixedBackbone`, embed/layers/heads/ff = `128/2/4/128`)
  - batch size `256`
  - epochs `10`
  - optimizer `AdamW(weight_decay=0.01)`
  - lr `1e-4`
  - schedule `warmup_cosine`
  - warmup fraction `0.1`
  - min lr scale `0.1`
  - seed `42`
- Probe panel:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
  - `IMLEGETKL`

## Conditions

- `mhcflurry_{50k,200k}`
- `log_mse_{50k,200k}`
- `twohot_d2_logit_{50k,200k}_K{64,128}`
- `hlgauss_d2_logit_{50k,200k}_K{64,128}`

## Launch History

- Initial detached launch `dist-ba-clean-c01..c12` failed immediately because the Modal image excluded `experiments/**`, so the experiment-local code directory was missing in-container.
- Fixed by explicitly adding `experiments/2026-03-13_1445_codex_clean-distributional-ba-heads/code` to the Modal image in `scripts/train_modal.py`.
- Verified the fix with non-detached smoke run `dist-ba-clean-fix-sync-c01` before relaunching the full corrected sweep.
- Final completed sweep:
  - `dist-ba-clean-fix-c01` .. `dist-ba-clean-fix-c12`
  - app ids are frozen in `manifest.json`

## Closure Outputs

- Raw run artifacts were pulled locally under `results/runs/dist-ba-clean-fix-cXX/`
- Generated from local artifacts via `analysis/aggregate_results.py`:
  - `results/condition_summary.csv`
  - `results/condition_summary.json`
  - `results/family_summary.csv`
  - `results/cap_summary.csv`
  - `results/best_by_family.csv`
  - `results/per_allele_metrics.csv`
  - `results/final_probe_predictions.csv`
  - `results/epoch_summary.csv`
  - `results/artifact_inventory.csv`
  - `results/metric_verification.csv`
  - `results/summary_bundle.json`
  - `results/test_spearman_ranking.png`
  - `results/val_spearman_curves.png`
  - `results/final_probe_heatmap.png`

## Metric Verification

- Held-out point metrics were recomputed from `val_predictions.csv` and `test_predictions.csv` for every run.
- Verified metrics:
  - Spearman
  - Pearson
  - RMSE in `log10(nM)`
  - `<=500 nM` accuracy
  - balanced accuracy
  - precision
  - recall
  - F1
  - AUROC
  - AUPRC
- Maximum absolute summary-vs-recomputed discrepancy across all verified metrics and all 12 runs: `3.4388e-04`
- All expected raw files were present for all 12 corrected runs.

## Results

Important comparison note: `val_loss` / `test_loss` are not directly comparable across regression and distributional head families because the loss definitions differ. Cross-family ranking should use held-out Spearman, Pearson, RMSE, AUROC, AUPRC, and probe behavior.

### Family means

| head_type | test_spearman | test_rmse_log10 | test_auroc | test_auprc | mean_epoch_time_s |
| --------- | ------------- | --------------- | ---------- | ---------- | ----------------- |
| mhcflurry | 0.6129 | 1.1570 | 0.8083 | 0.7942 | 4.3330 |
| log_mse   | 0.5970 | 1.1865 | 0.8019 | 0.7857 | 3.9458 |
| hlgauss   | 0.5249 | 1.2558 | 0.7635 | 0.7510 | 4.4775 |
| twohot    | 0.5169 | 1.2654 | 0.7588 | 0.7469 | 4.8296 |

### Condition ranking by held-out test Spearman

| label | head_type | max_nM | n_bins | test_spearman | test_rmse_log10 | test_auroc | test_auprc | mean_epoch_time_s |
| ----- | --------- | ------ | ------ | ------------- | --------------- | ---------- | ---------- | ----------------- |
| c02_mhcflurry_additive_max200k | mhcflurry | 200000 | 128 | 0.6209 | 1.1507 | 0.8124 | 0.7973 | 4.5867 |
| c01_mhcflurry_additive_max50k | mhcflurry | 50000 | 128 | 0.6048 | 1.1632 | 0.8042 | 0.7910 | 4.0793 |
| c04_log_mse_additive_max200k | log_mse | 200000 | 128 | 0.5995 | 1.1864 | 0.8030 | 0.7866 | 4.1128 |
| c03_log_mse_additive_max50k | log_mse | 50000 | 128 | 0.5946 | 1.1865 | 0.8007 | 0.7848 | 3.7788 |
| c12_hlgauss_d2_logit_max200k_K128_s0.75 | hlgauss | 200000 | 128 | 0.5381 | 1.2524 | 0.7709 | 0.7580 | 4.6476 |
| c11_hlgauss_d2_logit_max200k_K64_s0.75 | hlgauss | 200000 | 64 | 0.5339 | 1.2598 | 0.7675 | 0.7554 | 4.2983 |
| c08_twohot_d2_logit_max200k_K128 | twohot | 200000 | 128 | 0.5331 | 1.2587 | 0.7682 | 0.7546 | 6.4767 |
| c07_twohot_d2_logit_max200k_K64 | twohot | 200000 | 64 | 0.5237 | 1.2735 | 0.7617 | 0.7503 | 4.1976 |
| c09_hlgauss_d2_logit_max50k_K64_s0.75 | hlgauss | 50000 | 64 | 0.5176 | 1.2500 | 0.7592 | 0.7448 | 4.6980 |
| c10_hlgauss_d2_logit_max50k_K128_s0.75 | hlgauss | 50000 | 128 | 0.5101 | 1.2612 | 0.7565 | 0.7458 | 4.2661 |
| c05_twohot_d2_logit_max50k_K64 | twohot | 50000 | 64 | 0.5092 | 1.2593 | 0.7541 | 0.7413 | 3.9341 |
| c06_twohot_d2_logit_max50k_K128 | twohot | 50000 | 128 | 0.5014 | 1.2701 | 0.7512 | 0.7412 | 4.7102 |

### Best conditions by family

- `mhcflurry`: `c02_mhcflurry_additive_max200k`
  - test Spearman `0.6209`
  - test Pearson `0.6209`
  - test RMSE `1.1507`
  - test AUROC `0.8124`
  - test AUPRC `0.7973`
  - final probes:
    - `HLA-A*02:01 / SLLQHLIGL`: `70.2 nM`
    - `HLA-A*02:01 / FLRYLLFGI`: `98.2 nM`
    - `HLA-A*24:02 / NFLIKFLLI`: `12,580.9 nM`
- `log_mse`: `c04_log_mse_additive_max200k`
  - test Spearman `0.5995`
  - test AUROC `0.8030`
- `hlgauss`: `c12_hlgauss_d2_logit_max200k_K128_s0.75`
  - test Spearman `0.5381`
  - test AUROC `0.7709`
  - test AUPRC `0.7580`
  - `test_coverage_90`: `0.9226`
  - `test_pit_ks`: `0.0950`
- `twohot`: `c08_twohot_d2_logit_max200k_K128`
  - test Spearman `0.5331`
  - test AUROC `0.7682`
  - test AUPRC `0.7546`
  - `test_coverage_90`: `0.9249`
  - `test_pit_ks`: `0.1006`

## Takeaway

1. `mhcflurry` remains the clear winner under the fixed self-contained 10-epoch benchmark, and `max_nM=200k` is its best setting in this sweep.
2. `log_mse` is still the nearest regression alternative, about `0.021` test-Spearman behind the winning `mhcflurry` condition and slightly faster per epoch.
3. D2-logit distributional heads are viable but materially behind the regression heads on held-out ranking and classification metrics in this contract.
4. Across every head family tested here, `max_nM=200k` improved held-out Spearman over `50k`.
5. Within the distributional heads, `hlgauss` was slightly stronger than `twohot` on average, and the strongest distributional settings were the `200k` runs with `K=128`.
6. This clean rerun supports the earlier qualitative conclusion from the larger AblationEncoder sweeps, but now under the intended self-contained benchmark contract:
   - `mhcflurry` best
   - `log_mse` second
   - D2-logit heads plausible
   - no reason to resurrect D1-affine
