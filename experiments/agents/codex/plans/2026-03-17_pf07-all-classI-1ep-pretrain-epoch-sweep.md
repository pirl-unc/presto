# PF07 All-Class-I 1ep-Pretrain Epoch Sweep

## Objective

Rerun the recent honest PF07 comparison family on the rebuilt canonical merged dataset, using the `mhcseqs`-first sequence resolver and a broader all-class-I numeric binding contract.

This experiment should answer:
- whether the rebuilt merged dataset changes the ranking among the leading honest PF07 variants
- whether the broader all-class-I contract benefits from longer training
- whether the current sentry probes remain stable when training on the wider allele set

## Fixed Contract

- Dataset:
  - canonical `data/merged_deduped.tsv`
  - rebuilt on 2026-03-17 with bagged MHC restriction fields
  - Modal `presto-data` volume must be refreshed from the rebuilt canonical file before launch
- Sequence resolution:
  - `mhcseqs`-first exact sequence lookup
  - local `mhc_index.csv` only as fallback
- Inputs only:
  - `nflank`
  - `peptide`
  - `cflank`
  - `mhc_a`
  - `mhc_b`
- Forbidden:
  - assay selector / assay context inputs into the predictive trunk
- Shared training/data settings:
  - source: `iedb`
  - `train_all_alleles=true`
  - `train_mhc_class_filter=I`
  - `measurement_profile=numeric_no_qualitative`
  - `qualifier_filter=all`
  - peptide-group split `80/10/10`
  - split seed `42`
  - train seed `43`
  - main `Presto` path via `focused_binding_run`
  - `d_model=32`
  - `n_layers=2`
  - `n_heads=4`
  - batch size `256`
  - `AdamW`
  - `lr=1e-3`
  - `lr_schedule=constant`
  - `weight_decay=0.01`
  - `affinity_loss_mode=full`
  - `affinity_target_encoding=mhcflurry`
  - `kd_grouping_mode=split_kd_proxy`
  - `max_affinity_nM=100000`
  - `no_synthetic_negatives=true`
  - requested GPU `H100!`

## Fixed Warm Start

- Use the existing `1`-epoch `d32` MHC class/species pretrain checkpoint:
  - `/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt`
- Do not vary pretrain duration in this sweep.

## Condition Grid

Residual variants:
1. `pf07_control_constant`
   - `shared_base_factorized_context_plus_segment_residual`
2. `pf07_dag_method_leaf_constant`
   - `dag_method_leaf`
3. `pf07_dag_prep_readout_leaf_constant`
   - `dag_prep_readout_leaf`

Epoch budgets:
1. `10`
2. `25`
3. `50`

Total grid:
- `3` residual variants x `3` epoch budgets = `9` runs

## Probe Contract

Keep the same sentry panel:
- alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
- peptides:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
  - `IMLEGETKL`

Expected per-run outputs:
- `summary.json`
- `val_predictions.csv`
- `test_predictions.csv`
- `epoch_metrics.csv`
- `epoch_metrics.json`
- `probe_affinity_over_epochs.csv`
- `probe_affinity_over_epochs.json`
- `val_metrics_over_epochs.png`
- `probe_affinity_all_outputs_over_epochs.png`

## Launcher / Analysis Structure

Experiment family:
- `experiments/2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep`

Files:
- `code/launch.py`
- `analysis/aggregate_epoch_curves.py`
- `manifest.json`
- `results/runs/`
- `results/*.csv`
- `results/*.json`
- `results/*.png`

## Pre-Launch Checks

1. Confirm `focused_binding_probe` still supports:
   - `--train-all-alleles`
   - `--train-mhc-class-filter I`
   - per-epoch validation metrics
2. Confirm the fixed pretrain checkpoint exists on `presto-checkpoints`
3. Refresh `/merged_deduped.tsv` in the `presto-data` Modal volume from the rebuilt canonical local file
4. Ensure the Modal image installs `mhcseqs`, otherwise the run silently falls back to index-only sequence resolution

## Decision Rule

- Primary metric: held-out test Spearman
- Secondary:
  - AUROC
  - AUPRC
  - RMSE log10
- Promote a new baseline only if it materially beats the current honest PF07 winner on held-out test Spearman under the broader all-class-I contract.
- Use the epoch curves to decide whether `50` epochs is still justified on the widened dataset.
