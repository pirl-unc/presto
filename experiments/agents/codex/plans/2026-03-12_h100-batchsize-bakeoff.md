# 2026-03-12 H100! Batch-Size Bakeoff

## Purpose

Measure how batch size changes broad-contract training behavior on the current canonical Presto assay-head neighborhood when hardware is fixed to `H100!`.

This is a pure batch-size experiment:
- same data
- same number of epochs
- same warm start
- same no-synthetic / no-ranking contract
- same GPU family

Only the model design and batch size vary.

## Dataset / Curation Contract

- 7 class-I alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- Broad numeric binding contract:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- `measurement_profile=numeric_no_qualitative`
- `qualifier_filter=all`
- no qualitative binding
- no synthetic negatives
- no ranking losses

## Training Contract

- GPU: `H100!`
- fixed epoch count: `5`
- warm start:
  - `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- same seed for all conditions
- same optimizer / LR as current focused broad benchmark

## Designs

- `A03`
- `A05`
- `A06`
- `A07`

## Batch Sizes

- `64`
- `128`
- `192`
- `256`

## Conditions

16 total:
- `A03 x {64,128,192,256}`
- `A05 x {64,128,192,256}`
- `A06 x {64,128,192,256}`
- `A07 x {64,128,192,256}`

## Metrics to Record

- run success / failure
- setup wallclock
- epoch wallclock
- GPU utilization
- GPU peak allocated / reserved memory
- validation loss
- probes:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`

## Deliverables

- timestamped experiment directory under `experiments/`
- `manifest.json`
- `variants.md`
- `options_vs_perf.md`
- plots for:
  - epoch time vs batch size
  - val loss vs batch size
  - probe ratios vs batch size
- update `experiments/experiment_log.md`

## Decision Rule

Prefer the largest batch size that:
- trains successfully on `H100!`
- preserves or improves broad-contract probe behavior
- gives a real epoch-time / setup-time advantage

Do not pick a larger batch solely for speed if probe behavior or val loss degrades materially.
