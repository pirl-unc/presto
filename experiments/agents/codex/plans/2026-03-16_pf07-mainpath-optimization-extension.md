# PF07 Main-Path Optimization Extension

## Goal

Extend the best current full-Presto condition, `PF07_mhcflurry_100k_full`, without changing the data contract or the main Presto model path.

The question is not "does full Presto work at all?" That was answered by `2026-03-16_1415_codex_presto-full-output-exp21-retry`.

The new question is:

- can the remaining gap to EXP-21 be reduced by better optimization settings alone?

## Why this is the right extension

`PF07` was already the strongest full-Presto condition:

- same 2-allele EXP-21 dataset contract
- same sequence-only input policy
- richer multi-output assay supervision
- test Spearman `0.8441`

That leaves only about `0.0100` Spearman to the current EXP-21 winner. That is small enough that schedule/LR may matter.

Historical context from `2026-03-12_1643_codex_broad-lr-schedule-round1` also points in the same direction:

- `A07_mhcflurry_100k` preferred lower LR than the current PF07 retry
- `warmup_cosine` and `onecycle` were competitive or better than constant
- high LR `8e-4` was unstable or degraded

So the clean next step is not another architecture sweep. It is a small PF07-only optimization sweep.

## Codepath Verification

This experiment must stay on the same main Presto codepath as the successful PF07 retry:

- launcher calls `scripts/train_modal.py::focused_binding_run`
- Modal entrypoint runs `python -m presto.scripts.focused_binding_probe`
- the runner constructs `Presto(...)`
- training/eval call `model.forward_affinity_only(...)`
- `forward_affinity_only()` internally calls `Presto.forward()` and only filters outputs afterward

So this is already main-path Presto, not the separate groove/distributional BA benchmark family.

## Fixed Contract

### Dataset

- source: `data/merged_deduped.tsv`
- alleles: `HLA-A*02:01`, `HLA-A*24:02`
- measurement profile: `numeric_no_qualitative`
- qualifier filter: `all`
- split policy: peptide-group `80 / 10 / 10`
- split seed: `42`

### Model

- inputs only: `nflank`, `peptide`, `cflank`, `mhc_a`, `mhc_b`
- `d_model=32`
- `n_layers=2`
- `n_heads=4`
- `peptide_pos_mode=concat_start_end_frac`
- `groove_pos_mode=concat_start_end_frac`
- `binding_core_lengths=8,9,10,11`
- `binding_core_refinement=shared`
- `binding_kinetic_input_mode=affinity_vec`
- `binding_direct_segment_mode=off`
- `affinity_loss_mode=full`
- `affinity_target_encoding=mhcflurry`
- `affinity_assay_residual_mode=shared_base_factorized_context_plus_segment_residual`
- `kd_grouping_mode=split_kd_proxy`
- `max_affinity_nM=100000`

### Training

- train seed: `43`
- epochs: `50`
- batch size: `256`
- optimizer: `AdamW`
- weight decay: `0.01`
- synthetic negatives: off
- ranking losses: off
- requested GPU: `H100!`

## Conditions

### Positive Control

1. `PF07_ctrl_lr1e3_constant`
- `lr=1e-3`
- `lr_schedule=constant`

### Historical A07-inspired optimization variants

2. `PF07_lr2p8e4_warmup_cosine`
- `lr=2.8e-4`
- `lr_schedule=warmup_cosine`

3. `PF07_lr2p8e4_onecycle`
- `lr=2.8e-4`
- `lr_schedule=onecycle`

4. `PF07_lr1e4_warmup_cosine`
- `lr=1e-4`
- `lr_schedule=warmup_cosine`

5. `PF07_lr1e4_constant`
- `lr=1e-4`
- `lr_schedule=constant`

## Decision Rule

- Promote nothing unless a PF07 variant clearly improves the primary metric on the same contract.
- Primary metric: held-out test Spearman.
- Secondary metrics: test AUROC, test AUPRC, test RMSE log10.
- If a new variant ties on Spearman but is better on RMSE/AUPRC and remains equally simple, it is worth considering.
- If all variants trail the current PF07 control, conclude that optimization schedule/LR is probably not the main remaining bottleneck.

## Closure Requirements

- raw Modal artifacts fetched into `results/runs/`
- aggregated summary tables under `results/`
- experiment README updated with direct comparison to:
  - the previous PF07 retry
  - the EXP-21 groove winner
- canonical log updated in `experiments/experiment_log.md`
