# 2026-03-12 Hardware Generalization Bakeoff

## Question
Does the apparent optimization advantage of `H100!` over `A100` generalize to the winning configurations from the broad LR/schedule sweep, once memory pressure is equalized by reducing batch size to `128`?

## Motivation
Prior hardware bakeoff results showed:
- `H100!` is much faster than `A100`
- `A100` OOMs on some larger broad-contract models at larger batch sizes
- hardware appears to affect optimization trajectory, not just throughput

The current LR/schedule sweep may produce winners whose behavior on `H100!` could be confounded by memory headroom and kernel choice. To answer whether the optimization advantage generalizes, hardware must be compared on the same winning configs with a batch size that fits on all candidate GPUs.

## Fixed Contract
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
- Qualifiers: `all`
- Warm start from MHC pretrain checkpoint
- No synthetics
- No ranking / contrastive
- Same seed across hardware for each config
- Batch size: `128`
- Epochs: `20`

## Config Selection
This plan is now bound to the completed LR/schedule winners from:
- `experiments/2026-03-12_1643_codex_broad-lr-schedule-round1`

Selected configs:
1. `A03_log10_100k`
   - learning rate: `1e-4`
   - schedule: `warmup_cosine`
   - chosen because it had the strongest overall broad-contract probe behavior
2. `A07_mhcflurry_100k`
   - learning rate: `2.8e-4`
   - schedule: `warmup_cosine`
   - chosen because it had the best validation-loss profile within the `A07` family

These are now fixed. The hardware comparison should vary only:
- hardware

Not:
- architecture
- target space
- LR
- schedule
- batch size

## Hardware Matrix
For each winning config:
- `A100`
- `H100!`
- `H200`

## Metrics
Per run:
- success / failure
- divergence flag and divergence epoch/reason
- setup time
- per-epoch wallclock
- per-epoch current LR
- per-epoch train/val loss
- per-epoch probe metrics for:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- GPU memory:
  - peak allocated GiB
  - peak reserved GiB
- GPU utilization proxy if available

## Main Questions
1. Does `H100!` still converge faster than `A100` at batch `128`?
2. Does `H100!` still reach better probe behavior than `A100` at matched batch size?
3. Is the previous `H100!` advantage mostly a memory-headroom effect, or does it persist when all hardware fits?
4. Is `H200` a meaningful upgrade over `H100!` for these winning configs?

## Expected Outcome
- If `H100!` remains better on both speed and probe behavior at batch `128`, the optimization advantage likely generalizes beyond simple memory pressure.
- If `A100` closes the gap substantially at batch `128`, the earlier difference was at least partly due to batch-size / memory effects.
- If `H200` is not clearly superior to `H100!`, keep `H100!` as the default experiment hardware.
