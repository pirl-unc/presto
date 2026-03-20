# PF07 Sequence-Only All-Head Probe Rerun (2026-03-16)

## Goal

Rerun the PF07 optimizer comparison after fixing the affinity-only codepath so it truly ignores assay-selector context, while also extending the saved probe artifact to emit every affinity-family output head for the tracked peptide / allele panel.

## Why This Exists

- The previous PF07 experiments reported `affinity_input_contract=sequence_only`, but the focused runner still passed `binding_context` into `Presto.forward_affinity_only()`.
- The saved probe artifact only preserved `KD`, `IC50`, and the direct probe head, which is not enough to inspect `EC50` or the split proxy-KD heads.
- The user specifically wants to understand the `SLLQHLIGL / HLA-A*24:02` IC50 behavior and whether disagreement among heads suggests a mild cross-head regularization experiment.

## Fixed Contract

- Main model path: `Presto.forward_affinity_only()`
- Inputs: `nflank`, `peptide`, `cflank`, `mhc_a`, `mhc_b`
- Assay-selector inputs: forbidden and ignored in the affinity-only path
- Dataset: `data/merged_deduped.tsv`
- Alleles: `HLA-A*02:01`, `HLA-A*24:02`
- Measurement profile: `numeric_no_qualitative`
- Qualifier filter: `all`
- Assay families supervised: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- Split: peptide-group `80/10/10`, split seed `42`
- Train seed: `43`
- Epochs: `50`
- Batch size: `256`
- Optimizer: `AdamW`
- Weight decay: `0.01`
- Synthetic negatives: `off`
- Ranking losses: `off`
- GPU: `H100!`

## Conditions

1. `PF07_ctrl_lr1e3_constant`
2. `PF07_lr2p8e4_warmup_cosine`
3. `PF07_lr2p8e4_onecycle`
4. `PF07_lr1e4_warmup_cosine`
5. `PF07_lr1e4_constant`

## Probe Panel

- `SLLQHLIGL`
- `FLRYLLFGI`
- `NFLIKFLLI`
- `IMLEGETKL`

## Required Outputs

- Per-run `summary.json`
- `val_predictions.csv`
- `test_predictions.csv`
- `probe_affinity_over_epochs.csv`
- `probe_affinity_over_epochs.json`

The probe artifact must include:
- `KD_nM`
- `IC50_nM`
- `EC50_nM`
- `KD_proxy_ic50_nM`
- `KD_proxy_ec50_nM`
- direct `binding_affinity_probe_kd`

## Post-Run Analysis

- Compare all-head probe predictions across the five optimizer settings
- Inspect the real support rows for `SLLQHLIGL / HLA-A*24:02` by measurement type, qualifier, assay method, and train/val/test placement
- Decide whether a future weak agreement regularizer between related affinity outputs is justified
