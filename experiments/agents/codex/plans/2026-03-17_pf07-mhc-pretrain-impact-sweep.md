# PF07 MHC Pretrain Impact Sweep

## Objective

Measure whether short MHC class/species pretraining improves the current honest PF07 affinity models when the checkpoint width matches the downstream model.

## Why A New Pretrain Is Required

- The historical warm-start checkpoint `mhc-pretrain-20260308b` is `d_model=128`.
- The current honest PF07 winner is `d_model=32`, `n_layers=2`, `n_heads=4`.
- Using the old checkpoint would either fail or create a mismatched partial-load comparison.
- Therefore this experiment must use fresh `d32` MHC-only pretrains.

## Fixed Downstream Contract

- Main `Presto` path via `focused_binding_run`
- Inputs only:
  - `nflank`
  - `peptide`
  - `cflank`
  - `mhc_a`
  - `mhc_b`
- Dataset:
  - `data/merged_deduped.tsv`
  - alleles `HLA-A*02:01`, `HLA-A*24:02`
  - `measurement_profile=numeric_no_qualitative`
  - `qualifier_filter=all`
  - peptide-group split `80/10/10`
  - split seed `42`
  - train seed `43`
- Model size:
  - `d_model=32`
  - `n_layers=2`
  - `n_heads=4`
- Training:
  - `50` epochs
  - batch size `256`
  - `AdamW`
  - `lr=1e-3`
  - `weight_decay=0.01`
  - `mhcflurry`
  - `split_kd_proxy`
  - requested GPU `H100!`
- No assay-selector inputs

## Downstream Variants

1. `pf07_control_constant`
   - `shared_base_factorized_context_plus_segment_residual`
2. `pf07_dag_method_leaf_constant`
   - `dag_method_leaf`
3. `pf07_dag_prep_readout_leaf_constant`
   - `dag_prep_readout_leaf`

These give one flat control, one strong structured comparator, and the current best honest model.

## Pretraining Variants

1. `pretrain_0ep`
   - no warm start
2. `pretrain_1ep`
   - fresh `d32` MHC pretrain for `1` epoch
3. `pretrain_2ep`
   - fresh `d32` MHC pretrain for `2` epochs

MHC pretraining contract:

- script: `scripts/train_modal.py --mode mhc_pretrain`
- `d_model=32`
- `n_layers=2`
- `n_heads=4`
- checkpoint name: `mhc_pretrain.pt`
- same repo code and experiment family timestamp

## Intended Grid

- `3` downstream variants x `3` pretrain durations = `9` downstream runs
- plus `2` MHC pretrain runs

## Experiment Directory Structure

- top-level experiment dir:
  - `experiments/YYYY-MM-DD_HHMM_codex_pf07-mhc-pretrain-impact-sweep`
- manifests:
  - `manifest_pretrain.json`
  - `manifest.json` for downstream runs
- fetched artifacts:
  - `results/pretrains/`
  - `results/runs/`
- reproduction bundle:
  - `reproduce/`

## Launcher Behavior

- `code/launch.py` supports phases:
  - `pretrain`
  - `finetune`
  - `all`
- `all`:
  - writes both manifests
  - launches the `1`-epoch and `2`-epoch MHC pretrains
  - optionally waits for their checkpoints on Modal
  - launches the `9` downstream PF07 runs once checkpoints exist
- all run IDs include stable condition names and pretrain duration labels

## Collection / Closure

- Fetch pretrains:
  - `python scripts/fetch_experiment_modal_runs.py --experiment-dir ... --manifest manifest_pretrain.json --results-subdir results/pretrains`
- Fetch downstream:
  - `python scripts/fetch_experiment_modal_runs.py --experiment-dir ...`
- Aggregate downstream:
  - `python scripts/aggregate_summary_runs.py --experiment-dir ...`
- Final writeup must compare:
  - no-pretrain vs `1` epoch vs `2` epochs
  - effect by downstream architecture
  - whether pretraining changes the current model-to-beat

## Decision Rule

- Primary metric: held-out test Spearman
- Secondary: AUROC, AUPRC, RMSE log10
- Promote a pretrained model only if it beats the no-pretrain version of the same architecture on test Spearman without a concerning RMSE collapse
- If pretraining helps only the weaker variants but not the winner, document that explicitly and keep the current honest winner unchanged
