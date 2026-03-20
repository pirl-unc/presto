# PF07 DAG Epoch-Curve Rerun

## Objective

Rerun the leading honest PF07 DAG conditions with richer per-epoch metrics and locally reproducible plots so we can inspect training dynamics rather than only final summaries.

This experiment should answer:
- whether the leading DAG variants separate on validation AUROC / AUPRC / Spearman over time
- whether the winner changes once we look at full trajectories instead of only epoch-50 terminal metrics
- how all probe-output heads evolve across epochs for the tracked peptide/allele panel

## Fixed Contract

- Dataset: `data/merged_deduped.tsv`
- Alleles: `HLA-A*02:01`, `HLA-A*24:02`
- Measurement profile: `numeric_no_qualitative`
- Qualifier filter: `all`
- Split policy: peptide-group `80/10/10`
- Split seed: `42`
- Train seed: `43`
- Inputs only:
  - `nflank`
  - `peptide`
  - `cflank`
  - `mhc_a`
  - `mhc_b`
- Forbidden:
  - assay selector / assay context inputs into the predictive trunk
- Shared model/training settings:
  - main `Presto` path
  - `d_model=32`
  - `n_layers=2`
  - `n_heads=4`
  - `mhcflurry`
  - `split_kd_proxy`
  - `50` epochs
  - batch size `256`
  - `AdamW`
  - weight decay `0.01`
  - requested GPU `H100!`

## Condition Grid

1. `pf07_control_constant`
   - residual mode: `shared_base_factorized_context_plus_segment_residual`
   - `lr=1e-3`
   - `lr_schedule=constant`
2. `pf07_dag_family_constant`
   - residual mode: `dag_family`
   - `lr=1e-3`
   - `lr_schedule=constant`
3. `pf07_dag_method_leaf_constant`
   - residual mode: `dag_method_leaf`
   - `lr=1e-3`
   - `lr_schedule=constant`
4. `pf07_dag_prep_readout_leaf_constant`
   - residual mode: `dag_prep_readout_leaf`
   - `lr=1e-3`
   - `lr_schedule=constant`
5. `pf07_dag_method_leaf_warmup_cosine`
   - residual mode: `dag_method_leaf`
   - `lr=3e-4`
   - `lr_schedule=warmup_cosine`
6. `pf07_dag_prep_readout_leaf_warmup_cosine`
   - residual mode: `dag_prep_readout_leaf`
   - `lr=3e-4`
   - `lr_schedule=warmup_cosine`

Why this grid:
- rerun the four directly comparable conditions from the completed DAG sweep
- add only two extra conditions that are likely to alter the curve shape materially
- avoid a large optimizer sweep before we can even inspect trajectory behavior

## Required Logging Changes

### Per-epoch validation metrics

For selected epochs, and for this experiment every epoch:
- compute validation-set held-out metrics with the same evaluator used for final metrics
- store at least:
  - `val_spearman`
  - `val_pearson`
  - `val_rmse_log10`
  - `val_auroc`
  - `val_auprc`
  - `val_accuracy`
  - `val_balanced_accuracy`
  - `val_precision`
  - `val_recall`
  - `val_f1`

Implementation intent:
- do not write full validation prediction dumps every epoch
- write compact per-epoch metric summaries only
- keep final `val_predictions.csv` and `test_predictions.csv` as the terminal artifacts

### Probe-output trajectories

Keep the existing per-epoch probe rows, but also generate plots for all affinity outputs:
- `KD_nM`
- `IC50_nM`
- `EC50_nM`
- `KD_proxy_ic50_nM`
- `KD_proxy_ec50_nM`
- `binding_affinity_probe_kd`

Expected artifact shapes:
- per-run:
  - `epoch_metrics.csv`
  - `epoch_metrics.json`
  - `val_metrics_over_epochs.png`
  - `probe_affinity_over_epochs.csv`
  - `probe_affinity_over_epochs.json`
  - `probe_affinity_over_epochs.png`
  - `probe_affinity_all_outputs_over_epochs.png`
- experiment-level:
  - `results/epoch_metrics_by_condition.csv`
  - `results/epoch_metrics_by_condition.json`
  - `results/val_spearman_over_epochs.png`
  - `results/val_auroc_over_epochs.png`
  - `results/val_auprc_over_epochs.png`
  - optionally a combined multi-panel metric figure

## Reproducibility / Closure

- The experiment directory must be self-contained:
  - launcher under `code/launch.py`
  - analysis helper under `analysis/` if needed
  - frozen reproducibility bundle under `reproduce/`
  - locally fetched raw run artifacts under `results/runs/`
  - local aggregated CSV/JSON sufficient to recreate every curve plot
- README must include:
  - exact grid
  - exact metric/plot artifact locations
  - terminal held-out metrics
  - a short training-dynamics interpretation
- Update:
  - `experiments/experiment_log.md`
  - `experiments/model_to_beat.md` only if the practical baseline changes

## Decision Rule

- Primary metric remains held-out test Spearman.
- Validation curves are interpretive, not promotion criteria on their own.
- If a scheduled variant improves curve shape but not terminal test metrics, record it as informative but do not promote it.
- If a rerun materially changes the ranking among the top DAG variants, update the stable baseline doc accordingly.
