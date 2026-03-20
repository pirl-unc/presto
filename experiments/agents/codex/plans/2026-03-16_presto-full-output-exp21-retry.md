# Presto Full-Output EXP-21-Contract Retry

## Goal

Retry the "full Presto on the EXP-21 dataset" question with a cleaner contract:

- keep the EXP-21 data selection fixed
- keep the sequence-only assay-input policy fixed
- use Presto's richer multi-output affinity head families instead of the earlier collapsed `pooled_single_output` setup
- make the split seed and training seed explicit and separate

## Why the prior retry was not enough

The earlier experiment `2026-03-16_1010_codex_presto-mainpath-affinity-seqonly-replication` was a useful failure, but it was not the strongest or cleanest Presto comparison because:

1. it used `affinity_assay_residual_mode=pooled_single_output`, which collapses KD / IC50 / EC50 structure too aggressively
2. it used `affinity_loss_mode=assay_heads_only`, which ignores the direct affinity-probe branch and broader full-loss coupling
3. it used `peptide_pos_mode=triple` and `groove_pos_mode=sequential`, while stronger historical Presto runs used the `concat_start_end_frac` positional base
4. it conflated dataset split seed and training seed, so it could not truly match the EXP-21 contract of fixed split `42` plus winner seed `43`

## Fixed contract

### Dataset

- source: `data/merged_deduped.tsv`
- alleles: `HLA-A*02:01`, `HLA-A*24:02`
- measurement profile: `numeric_no_qualitative`
- qualifier filter: `all`
- split policy: peptide-group `80 / 10 / 10`
- split seed: `42`

### Training

- train seed: `43`
- epochs: `50`
- batch size: `256`
- optimizer: `AdamW`
- lr: `1e-3`
- weight decay: `0.01`
- synthetic negatives: disabled
- ranking losses: disabled
- warm start: none on the first retry
- requested GPU: `H100!`

### Shared architecture

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

## Conditions

### PF03_log10_100k_full

- residual mode: `shared_base_segment_residual`
- KD grouping: `split_kd_proxy`
- loss mode: `full`
- target encoding: `log10`
- `max_affinity_nM=100000`

Rationale:
- `A03 log10_100k` was the strongest broad-contract canonical Presto target-space setting in prior bakeoffs
- `full` loss beat `assay_heads_only` in Claude's 7-allele Presto-vs-Groove comparison

### PF07_mhcflurry_100k_full

- residual mode: `shared_base_factorized_context_plus_segment_residual`
- KD grouping: `split_kd_proxy`
- loss mode: `full`
- target encoding: `mhcflurry`
- `max_affinity_nM=100000`

Rationale:
- this is the closest richer Presto analogue to the EXP-21 winner's bounded `mhcflurry` target contract
- the factorized-plus-segment residual path is the most explicit multi-output affinity-head family in current Presto

## Required code changes

### Focused runner seed split

Add `--split-seed` to `scripts/focused_binding_probe.py`:

- default behavior:
  - if `--split-seed` is omitted, preserve existing behavior by using `seed`
- use `split_seed` for:
  - `_prepare_real_binding_state(...)`
  - split/balance/cache dataset contract metadata
- use `seed` as the training seed for:
  - `torch.manual_seed`
  - `random.seed`
  - dataloader shuffling
  - per-epoch synthetic generation

### Experiment-local launcher

Create a fresh experiment directory with:

- `README.md`
- `code/launch.py`
- `reproduce/launch.sh`
- `reproduce/launch.json`

The launcher should:

- initialize the experiment via `experiment_registry.initialize_experiment_dir`
- launch both conditions through `scripts/train_modal.py::focused_binding_run`
- record `required_files` at least:
  - `summary.json`
  - `metrics.jsonl`
  - `probes.jsonl`
  - `step_log.jsonl`
  - `val_predictions.csv`
  - `test_predictions.csv`

## Closure and evaluation

After the runs finish:

- fetch all raw artifacts into `results/runs/`
- aggregate summaries into `results/condition_summary.csv`
- compare held-out validation and test metrics against:
  - the failed `presto-mainpath-affinity-seqonly` retry
  - the EXP-21 `groove c02` baseline
- update the experiment README and `experiments/experiment_log.md`

## Decision rule

- If either Presto condition becomes biologically plausible and materially improves over the failed seq-only retry, record it as the best current full-Presto broad-binding baseline.
- If both still trail EXP-21 by a large margin, conclude that the issue is no longer "wrong launcher / wrong head collapse" and instead lies deeper in optimization, target choice, or architectural scale.
