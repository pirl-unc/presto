# 2026-03-15 EXP-21 Seed + Epoch Confirmation Sweep

## Goal

Confirm whether the new EXP-20 groove winner is robust enough to become the canonical shared-code baseline, and combine that with a schedule sweep so the decision is not made on one seed and one epoch budget.

## Why this is the right next step

EXP-20 established two important facts:

1. the historical EXP-16 winner can now be reproduced exactly through the shared path
2. the best single new shared-path run is `groove + cond_id=1 + cc0`, but it only beat the exact positive control by about `+0.0028` Spearman on one seed

That gap is real enough to justify follow-up, but still small enough that we should verify:

- seed robustness
- whether the advantage survives at `100` and `200` epochs
- whether `groove c01` really beats both `groove c02` and the historical positive control on average, not just on one lucky draw

## Fixed executable contract

- source: `data/merged_deduped.tsv`
- alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
- measurement profile: `numeric_no_qualitative`
- assay families:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- qualifier filter: `all`
- split: `peptide_group_80_10_10_seed42`
- batch size: `256`
- optimizer: `AdamW`
- learning rate: `1e-3`
- weight decay: `0.01`
- warm start: none
- GPU: `H100!`

## Models to compare

Three fresh-run families only:

1. `groove`, `cond_id=1`, `content_conditioned=false`
   - label: `c01_mhcflurry_additive_max50k_d32`
   - current best single EXP-20 run

2. `groove`, `cond_id=2`, `content_conditioned=false`
   - label: `c02_mhcflurry_additive_max100k_d32`
   - same backbone / same head family / alternate target range

3. `historical_ablation`, `cond_id=2`, `content_conditioned=false`
   - label: `c02_mhcflurry_additive_max100k_d32`
   - exact historical EXP-16 positive control

## Sweep axes

- seeds: `42`, `43`, `44`, `45`
- epochs: `50`, `100`, `200`

Total:

- `3 models × 4 seeds × 3 epoch budgets = 36 runs`

No reuse of old runs. Every condition gets a new run id and fresh artifact directory.

## Questions to answer

1. Does `groove c01 cc0` still have the best mean test Spearman across seeds?
2. Does `groove c01` remain preferred at `100` or `200` epochs, or was `50` epochs a lucky schedule?
3. Does the historical positive control remain the best-calibrated model on RMSE / F1 / balanced accuracy even if groove wins Spearman?
4. Does `groove c02` close the gap or surpass `groove c01` once seeds and epochs are averaged?

## Decision rule

Promote `groove c01 cc0` as the canonical shared-code baseline only if:

- it has the best mean test Spearman across the 4-seed panel for at least one epoch budget
- it is not obviously dominated on RMSE / AUROC / AUPRC by the historical positive control
- its advantage is not just one-seed noise

If results are mixed:

- prefer the model with the best mean test Spearman at the shortest stable schedule
- use RMSE / AUROC / AUPRC as tie-breakers

## Required outputs

- all raw run artifacts under `results/runs/`
- `condition_summary.csv/json`
- schedule / seed grouped summaries
- plots:
  - ranking plot
  - seed-distribution plot
  - epoch-budget comparison plot
  - training curves
- README update
- canonical `experiment_log.md` entry
