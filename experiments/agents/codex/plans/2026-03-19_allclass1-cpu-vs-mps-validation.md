# Realistic All-Class-I CPU vs MPS Validation

## Goal

Validate Apple Silicon `mps` on a materially larger honest PF07 class-I training slice after the seeded manual-dropout fix, so local training guidance is based on more than the 200-row smoke runs.

## Why This Is Next

- The dropout contract itself is now hardware-independent when `manual_dropout` is forced on both CPU and MPS.
- The remaining uncertainty is whether a larger honest all-class-I run still stays numerically stable and close enough to CPU to recommend `mps` for local focused training.
- The active all-class-I local rerun launcher is the right code path to validate, but it needs small-run controls so we do not have to rerun the full 9-condition family.

## Fixed Modeling Contract

- Main `Presto` focused affinity path
- Honest inputs only:
  - `nflank`
  - `peptide`
  - `cflank`
  - `mhc_a`
  - `mhc_b`
- No assay-selector inputs
- Supervised affinity-family outputs:
  - `IC50`
  - `KD`
  - `KD(~IC50)`
  - `KD(~EC50)`
  - `EC50`
- Warm start:
  - checked-in `1`-epoch MHC pretrain checkpoint from `2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep`

## Validation Contract

- Dataset:
  - canonical rebuilt `data/merged_deduped.tsv`
  - `source=iedb`
  - `train_mhc_class_filter=I`
  - `train_all_alleles=True`
  - `measurement_profile=numeric_no_qualitative`
  - `qualifier_filter=all`
- Downstream condition:
  - `pf07_dag_prep_readout_leaf_constant`
- Compare:
  - `device=cpu`
  - `device=mps`
- Force identical dropout implementation on both:
  - `--mps-safe-mode manual_dropout`
- Keep sentry alleles/peptides:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
  - `IMLEGETKL`

## Scope Controls

- Add launcher support for:
  - `--condition-keys`
  - `--max-records`
- Use the new controls to run one reduced, realistic validation rather than the full 9-condition family.

## Success Criterion

- Both CPU and MPS runs complete locally without `non_finite_train_loss`.
- Both runs produce full local artifacts:
  - `summary.json`
  - `epoch_metrics.csv/json`
  - `probe_affinity_over_epochs.csv/json`
  - `val_predictions.csv`
  - `test_predictions.csv`
- The experiment README states clearly:
  - dropout implementation was matched across hardware
  - whether remaining CPU-vs-MPS drift is small enough to treat MPS as usable for local focused training
