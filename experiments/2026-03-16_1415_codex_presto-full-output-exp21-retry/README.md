# Presto Full-Output Retry on EXP-21 Contract

- Agent: `codex`
- Source script: `code/launch.py`
- Source baseline: `experiments/2026-03-15_1226_codex_exp21-seed-epoch-confirmation`
- Status: `completed`
- Result: `full-output Presto is now viable on the EXP-21 contract, but it does not beat the groove baseline`

## Goal

Retry the "full Presto on EXP-21" question carefully instead of repeating the earlier collapsed seq-only failure mode.

The earlier `presto-mainpath-affinity-seqonly` attempt used a much narrower output contract:

- `affinity_loss_mode=assay_heads_only`
- `affinity_assay_residual_mode=pooled_single_output`
- a weaker positional/core-window contract
- split seed accidentally tied to training seed

This retry keeps the exact EXP-21 data contract but restores the richer Presto multi-output head family:

- sequence-only inputs: `nflank`, `peptide`, `cflank`, `mhc_a`, `mhc_b`
- supervised outputs across `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, and `EC50`
- shared latent trunk with assay-specific outputs, no assay-selector inputs
- separate split seed `42` and training seed `43`

## Dataset Contract

- Source: `data/merged_deduped.tsv`
- Alleles: `HLA-A*02:01`, `HLA-A*24:02`
- Input fields: `nflank`, `peptide`, `cflank`, `mhc_a`, `mhc_b`
- Supervised assay families: `IC50`, `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- Measurement profile: `numeric_no_qualitative`
- Qualifier filter: `all`
- Split policy: `peptide_group_80_10_10_seed42`
- Split sizes: train `15530`, val `1919`, test `1915`

## Training Contract

- Epochs: `50`
- Batch size: `256`
- Optimizer: `AdamW`
- LR: `1e-3`
- Weight decay: `0.01`
- Requested Modal GPU: `H100!`
- `d_model=32`, `n_layers=2`, `n_heads=4`
- `peptide_pos_mode=concat_start_end_frac`
- `groove_pos_mode=concat_start_end_frac`
- `binding_core_lengths=8,9,10,11`
- `binding_core_refinement=shared`
- `binding_kinetic_input_mode=affinity_vec`
- `binding_direct_segment_mode=off`
- synthetic negatives: off
- ranking losses: off

## Tested Conditions

| condition | residual mode | target encoding | test Spearman | test AUROC | test AUPRC | test RMSE log10 | test F1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `PF07_mhcflurry_100k_full` | `shared_base_factorized_context_plus_segment_residual` | `mhcflurry` | `0.84410185` | `0.93680900` | `0.89547038` | `0.86136419` | `0.84720496` |
| `PF03_log10_100k_full` | `shared_base_segment_residual` | `log10` | `0.82043570` | `0.92695010` | `0.88114345` | `0.92678767` | `0.83105591` |

Validation metrics for the stronger `PF07` condition:

- val Spearman `0.84195828`
- val AUROC `0.94081622`
- val AUPRC `0.90512663`
- val RMSE log10 `0.84019047`

## Comparison To EXP-21

Reference winner to beat at the time:

- `dist-ba-v6-confirm-groove_c02-c02-cc0-e050-s43`
- test Spearman `0.85413903`
- test AUROC `0.94411862`
- test AUPRC `0.91761374`
- test RMSE log10 `0.81867343`

`PF07_mhcflurry_100k_full` is the best full-Presto retry, but it still trails the EXP-21 winner:

- test Spearman: `-0.01003718`
- test AUROC: `-0.00730962`
- test AUPRC: `-0.02214336`
- test RMSE log10: `+0.04269075`

Important nuance: the classification-style metrics are competitive. Relative to the EXP-21 winner, `PF07` is slightly better on thresholded `accuracy`, `balanced_accuracy`, and `F1`, but it is worse on the primary regression/ranking metrics.

Historical caveat added on 2026-03-16: that EXP-21 comparator is now known to be an assay-conditioned legacy benchmark, not an honest no-assay-input baseline. See [2026-03-16_2142_codex_exp21-honest-no-assay-repeat](../2026-03-16_2142_codex_exp21-honest-no-assay-repeat/).

## Interpretation

- The careful retry succeeded technically. Full-output Presto no longer collapses on this contract.
- The earlier failure was not evidence that full Presto could not learn this dataset; it was mostly a bad model/output contract.
- `PF07` is the right direction:
  - `mhcflurry` target encoding is better than `log10`
  - `shared_base_factorized_context_plus_segment_residual` is better than the simpler segment residual
- Even with that stronger configuration, the dedicated EXP-21 groove baseline looked better on the metrics that matter most for this benchmark.

The current conclusion is therefore:

- full-output Presto is now a credible model family on the EXP-21 dataset
- at the time of this experiment, the assay-conditioned EXP-21 groove baseline still looked like the empirical winner
- after the honest no-assay repeat, the cleaner current no-assay-input baseline is the PF07 untied control rather than EXP-21 `groove c02`
- this experiment should be treated as a recovery of full-Presto viability, not a baseline promotion

## Artifact Notes

- Raw run artifacts were fetched locally under `results/runs/`.
- Derived summaries/plots were regenerated with the shared aggregation tool.
- The focused binding runner emits:
  - `summary.json`
  - `probe_affinity_over_epochs.json`
  - `probe_affinity_over_epochs.csv`
  - `val_predictions.csv`
  - `test_predictions.csv`

## Artifacts

- launch manifest: `manifest.json`
- launch logs:
  - `launch_logs/presto-exp21-full-20260316a-pf03_log10_100k_full-e050-s43.log`
  - `launch_logs/presto-exp21-full-20260316a-pf07_mhcflurry_100k_full-e050-s43.log`
- raw run dirs:
  - `results/runs/presto-exp21-full-20260316a-pf03_log10_100k_full-e050-s43/`
  - `results/runs/presto-exp21-full-20260316a-pf07_mhcflurry_100k_full-e050-s43/`
- summary tables:
  - `results/condition_summary.csv`
  - `results/epoch_summary.csv`
  - `results/final_probe_predictions.csv`
  - `results/summary_bundle.json`
- plots:
  - `results/test_spearman_ranking.png`
  - `results/test_metric_grid.png`
  - `results/training_curves.png`
  - `results/final_probe_heatmap.png`
- reproduce bundle: [`reproduce/`](./reproduce/)

## Decision

Do not promote this family over EXP-21 yet.

If full Presto is going to replace the current groove baseline, the next experiment should start from `PF07` and target the remaining gap directly, rather than revisiting the weaker `PF03` branch.

## Handoff

- Status: closed
- Next Step: treat `PF07` as the full-Presto configuration to extend if you want another main-path attempt
- Open Questions:
  - can `PF07` close the remaining gap with a short schedule or weight sweep without giving up the full multi-output contract?
  - should the next comparison add a groove-style narrower affinity objective inside full Presto, or would that undercut the point of the richer output family?
