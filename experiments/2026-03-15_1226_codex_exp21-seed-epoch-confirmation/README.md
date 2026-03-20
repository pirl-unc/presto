# EXP-21 Seed + Epoch Confirmation Sweep

- Agent: `codex`
- Git commit: `e17aa284c89767d1b9827753dd7dd26c5750171e` (`main`, dirty)
- Source script: `code/launch.py`
- Analysis script: `analysis/aggregate_results.py`
- Requested Modal GPU: `H100!`
- Experiment dir: `experiments/2026-03-15_1226_codex_exp21-seed-epoch-confirmation`

Historical note (2026-03-16):

- this benchmark path uses assay embeddings from `binding_context` inside the legacy distributional BA model
- it is therefore an assay-conditioned benchmark, not a canonical no-assay-input baseline
- see [2026-03-16_2142_codex_exp21-honest-no-assay-repeat](../2026-03-16_2142_codex_exp21-honest-no-assay-repeat/) for the honest repeat of the same family with assay inputs disabled

## Goal

This experiment confirms whether the new shared-code EXP-20 groove winner is actually more robust than the exact historical EXP-16 positive control, and whether extending training beyond `50` epochs improves or hurts the ranking.

The sweep is:

- `3` model families
  - `groove c01`: `cond_id=1`, `mhcflurry`, `max_nM=50k`, `d=32`
  - `groove c02`: `cond_id=2`, `mhcflurry`, `max_nM=100k`, `d=32`
  - `historical c02`: exact historical-ablation positive control for `cond_id=2`
- `3` epoch budgets: `50`, `100`, `200`
- `4` fresh seeds: `42`, `43`, `44`, `45`
- total: `36` fresh runs, no reuse

## Final Result

The best new baseline from the full `36 / 36` rerun is:

- `groove c02`, `50` epochs
- mean test Spearman across seeds: `0.847549`
- mean test AUROC: `0.941650`
- mean test AUPRC: `0.913806`
- mean test RMSE log10: `0.839834`

The best single run in the full sweep is:

- `dist-ba-v6-confirm-groove_c02-c02-cc0-e050-s43`
- test Spearman `0.854139`
- test AUROC `0.944119`
- test AUPRC `0.917614`
- test RMSE log10 `0.818673`
- best val loss `0.025500` at epoch `25`
- best val Spearman `0.845193` at epoch `39`

The exact historical positive control remained competitive, but it did not win:

- `historical c02`, `50` epochs mean test Spearman: `0.840384`
- original parity seed remains intact:
  - `dist-ba-v6-confirm-historical_c02-c02-cc0-e050-s42`
  - test Spearman `0.843516`
  - this matches the historical EXP-16 positive-control metric

## Main Takeaways

1. Within this assay-conditioned benchmark family, `groove c02` is the robust winner, not `groove c01`.
2. `50` epochs is the best schedule in this family.
3. Longer schedules hurt rather than help on this contract:
   - overall mean test Spearman by epoch budget:
     - `50`: `0.844027`
     - `100`: `0.836819`
     - `200`: `0.830245`
4. The old historical-ablation positive control is still strong, but both groove variants beat it on full-sweep mean performance.

## Dataset Contract

- source: `data/merged_deduped.tsv`
- alleles: `HLA-A*02:01`, `HLA-A*24:02`
- measurement profile: `numeric_no_qualitative`
- assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- qualifier filter: `all`
- split: `peptide_group_80_10_10_seed42`
- split sizes: `15,530 / 1,919 / 1,915` train / val / test

## Training Contract

- config version: `v6`
- batch size: `256`
- optimizer: `AdamW`
- lr: `1e-3`
- weight decay: `0.01`
- warm start: none
- content conditioning: disabled for all runs
- seeds: `42`, `43`, `44`, `45`
- epochs: `50`, `100`, `200`

## Execution Note

The first bulk relaunch path for this experiment was invalid: it treated early detached Modal app ids as durable launch success, but most of those runs never reached the checkpoint volume. The final closed result in this directory comes from the corrected background-detach launch path that kept local `modal run --detach` processes alive until the remote runs were fully established.

All `36` raw run directories are now present locally under `results/runs/`.

## Model Summary

| model | runs | mean test Spearman | max test Spearman | mean test AUROC | mean test AUPRC | mean test RMSE log10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `groove c02` | 12 | **`0.839279`** | **`0.854139`** | `0.937187` | `0.906572` | `0.865560` |
| `groove c01` | 12 | `0.837644` | `0.846301` | **`0.938916`** | **`0.908707`** | **`0.860978`** |
| `historical c02` | 12 | `0.834168` | `0.844117` | `0.933470` | `0.898784` | `0.877360` |

Interpretation:

- `groove c02` wins on mean test Spearman and on the best single run.
- `groove c01` keeps slightly better average AUROC / AUPRC / RMSE, but not enough to offset the Spearman gap on this benchmark.
- `historical c02` remains a useful positive control, but it is not the best baseline anymore.

## Epoch Budget Summary

| model | epochs | runs | mean test Spearman | std | mean test AUROC | mean test AUPRC | mean test RMSE log10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `groove c02` | 50 | 4 | **`0.847549`** | `0.004555` | `0.941650` | `0.913806` | `0.839834` |
| `groove c01` | 50 | 4 | `0.844148` | `0.002562` | **`0.942571`** | **`0.916469`** | `0.841260` |
| `historical c02` | 50 | 4 | `0.840384` | `0.004091` | `0.936556` | `0.904899` | `0.856298` |
| `groove c02` | 100 | 4 | `0.839175` | `0.002217` | `0.936669` | `0.906141` | `0.867995` |
| `groove c01` | 100 | 4 | `0.838587` | `0.002547` | `0.939570` | `0.910417` | `0.860988` |
| `historical c02` | 100 | 4 | `0.832697` | `0.004225` | `0.932124` | `0.897192` | `0.881375` |
| `groove c02` | 200 | 4 | `0.831113` | `0.003105` | `0.933242` | `0.899769` | `0.888851` |
| `groove c01` | 200 | 4 | `0.830197` | `0.002754` | `0.934607` | `0.899236` | `0.880686` |
| `historical c02` | 200 | 4 | `0.829424` | `0.004678` | `0.931729` | `0.894262` | `0.894406` |

Interpretation:

- `50` epochs is clearly best.
- `100` epochs is consistently worse than `50`, but still better than `200`.
- `200` epochs degrades all three model families on this contract.

## Top Runs

| rank | run id | model | epochs | seed | test Spearman | test AUROC | test AUPRC | test RMSE log10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `dist-ba-v6-confirm-groove_c02-c02-cc0-e050-s43` | `groove c02` | 50 | 43 | **`0.854139`** | `0.944119` | `0.917614` | **`0.818673`** |
| 2 | `dist-ba-v6-confirm-groove_c02-c02-cc0-e050-s45` | `groove c02` | 50 | 45 | `0.847037` | `0.936463` | `0.913496` | `0.851523` |
| 3 | `dist-ba-v6-confirm-groove_c01-c01-cc0-e050-s42` | `groove c01` | 50 | 42 | `0.846301` | **`0.949170`** | `0.917409` | `0.834653` |
| 4 | `dist-ba-v6-confirm-groove_c01-c01-cc0-e050-s43` | `groove c01` | 50 | 43 | `0.846206` | `0.943428` | **`0.919325`** | `0.829129` |
| 5 | `dist-ba-v6-confirm-groove_c02-c02-cc0-e050-s44` | `groove c02` | 50 | 44 | `0.844728` | `0.940477` | `0.914625` | `0.843666` |

Across the `50`-epoch seed winners:

- seed `42`: `groove c01`
- seeds `43`, `44`, `45`: `groove c02`

So `groove c02` wins `3 / 4` seed matchups at the best schedule.

## Artifacts

- launch manifest: `manifest.json`
- raw run artifacts: `results/runs/`
- condition table: `results/condition_summary.csv`
- epoch table: `results/epoch_summary.csv`
- model rollups:
  - `results/model_summary.csv`
  - `results/model_epoch_summary.csv`
  - `results/model_seed_summary.csv`
  - `results/model_summary.json`
- plots:
  - `results/test_spearman_ranking.png`
  - `results/test_metric_grid.png`
  - `results/training_curves.png`
  - `results/seed_spearman_boxplot.png`
  - `results/epoch_budget_comparison.png`
- probe predictions: `results/final_probe_predictions.csv`
- reproduce bundle: `reproduce/`

## Decision

Historical decision at the time: promote `groove c02` with `50` epochs as the seed-robust winner of this assay-conditioned benchmark family.

Current caveat: after [2026-03-16_2142_codex_exp21-honest-no-assay-repeat](../2026-03-16_2142_codex_exp21-honest-no-assay-repeat/), this run family should no longer be treated as the active honest baseline because it depends on assay embeddings in the legacy benchmark path.

Do not extend this family to `100` or `200` epochs by default; the schedule sweep argues against it.
