# CPU vs MPS Focused PF07 Smoke Compare

## Goal

Run a matched tiny local comparison between CPU and Apple Silicon MPS on the same focused PF07 training path so we can decide whether MPS is actually safe enough for local continuation work.

## Why This Exists

- The local non-Modal resume path is now wired.
- MPS is detectable and selectable on this machine.
- But the first tiny MPS smoke run reached real setup and summary writing, then diverged with `non_finite_train_loss`.
- A matched CPU run on the same tiny contract completed.

So the next step is not a guess; it is a registered apples-to-apples smoke comparison.

## Fixed Contract

- Runner: `presto.scripts.focused_binding_probe`
- Model:
  - `dag_prep_readout_leaf`
  - `d_model=32`
  - `n_layers=2`
  - `n_heads=4`
  - `mhcflurry`
  - `split_kd_proxy`
- Inputs:
  - `nflank`
  - `peptide`
  - `cflank`
  - `mhc_a`
  - `mhc_b`
- Warm start:
  - `experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt`
- Tiny smoke data contract:
  - `source=iedb`
  - class I only
  - `measurement_profile=numeric_no_qualitative`
  - `qualifier_filter=all`
  - `train_all_alleles`
  - `max_records=200`
  - `val_fraction=0.1`
  - `test_fraction=0.1`
- Probe panel:
  - alleles: `HLA-A*02:01`, `HLA-A*24:02`
  - peptides: `SLLQHLIGL`, `FLRYLLFGI`
- Optimization/runtime:
  - `epochs=1`
  - `batch_size=8`
  - `lr=1e-3`
  - `weight_decay=0.01`
  - `num_workers=0`
  - `pin_memory=false`
  - `persistent_workers=false`

## Compared Conditions

1. `cpu`
2. `mps`

Same everything else:

- same seed
- same split seed
- same warm start
- same tiny dataset cap

## Outputs To Compare

- whether the run diverged
- divergence reason, if any
- whether summary artifacts were written
- final held-out validation/test metrics, if present
- any device-specific warnings of note

## Expected Interpretation

- If CPU and MPS both complete and metrics are close:
  - MPS is viable for local continuation
- If CPU completes and MPS diverges:
  - MPS is still experimental / not safe enough for the main local continuation path
- If both diverge:
  - the smoke contract itself is too brittle to answer the device question

## Non-Goals

- no full epoch sweep
- no large all-class-I local rerun
- no architecture change
- no attempt to “fix” MPS numerics in this same experiment unless the failure mode is immediately obvious
