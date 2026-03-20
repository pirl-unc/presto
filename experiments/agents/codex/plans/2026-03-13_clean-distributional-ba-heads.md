# 2026-03-13 Clean Distributional vs Regression BA Heads

## Goal

Run a fair head-only benchmark on the broad 7-allele numeric class-I binding contract where the backbone, optimizer, LR schedule, batch size, epoch budget, warm start, and data split are all fixed. Only the output head / target encoding varies.

## Fixed contract

- Source: `data/merged_deduped.tsv`
- Alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- Assay families included:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- Qualifier policy: `all` (exact and censored `>` rows)
- Split: peptide-group `80/10/10`, seed `42`
- Batch size: `256`
- Epochs: `10`
- GPU: `H100!`
- Optimizer: `AdamW(weight_decay=0.01)`
- LR schedule: fixed broad-contract winner for all conditions
- Warm start: fixed and identical for all conditions
- No synthetics
- No ranking losses
- Probes:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`

## Backbone

Use a self-contained fixed backbone under `experiments/2026-03-13_1445_codex_clean-distributional-ba-heads/code/`, not `AblationEncoder` imported from another experiment or any shared `scripts/distributional_ba/` runner.

Backbone choice:
- segment-wise groove-transformer style encoder
- shared across all conditions
- best broad-contract positional family from prior sweeps:
  - `mlp(concat(start,end,nterm_frac,cterm_frac))`
- identical parameters across all runs

## Runtime boundary

- The experiment-local package is the source of truth for:
  - condition definitions
  - backbone
  - train/eval loop
  - metrics export
  - launch behavior
- Modal may use a thin wrapper only to mount data/checkpoints and prepend the experiment `code/` directory to `PYTHONPATH`.
- Do not route this benchmark through `scripts/distributional_ba/`; that shared path is considered mutable/stale for this rerun.

## Conditions (12)

Regression:
- `mhcflurry_50k`
- `mhcflurry_200k`
- `log_mse_50k`
- `log_mse_200k`

Distributional (D2 only):
- `twohot_d2_logit_50k_K64`
- `twohot_d2_logit_50k_K128`
- `twohot_d2_logit_200k_K64`
- `twohot_d2_logit_200k_K128`
- `hlgauss_d2_logit_50k_K64`
- `hlgauss_d2_logit_50k_K128`
- `hlgauss_d2_logit_200k_K64`
- `hlgauss_d2_logit_200k_K128`

## Required outputs

Per run:
- `step_log.jsonl`
- `metrics.jsonl`
- `probes.jsonl`
- `val_predictions.parquet` or `.csv`
- `test_predictions.parquet` or `.csv`
- `summary.json`

Family level:
- `options_vs_perf.md`
- `options_vs_perf.json`
- plots for:
  - val/test loss
  - Spearman/Pearson
  - `<=500 nM` AUROC/AUPRC/accuracy/balanced accuracy
  - probe trajectories
  - distributional calibration where applicable

## Questions to answer

1. Does `D2-logit` become competitive when the benchmark is fair?
2. Does `200k` improve weak-tail calibration relative to `50k`?
3. Does `K=64` beat `K=128` on this censored mixed-assay contract?
4. Does `mhcflurry` remain the strongest baseline under the current best training contract?
