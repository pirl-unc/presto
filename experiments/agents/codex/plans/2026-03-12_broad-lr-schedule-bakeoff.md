# Broad LR / Schedule Bakeoff

## Goal

Find the fastest stable path to strong broad-contract class-I binding performance on a fixed `20`-epoch budget by sweeping a small set of initial learning rates and scheduler shapes over the two leading canonical Presto architecture hypotheses.

This bakeoff is intended to answer:
1. whether `A03` or `A07` is more optimization-friendly once target space is fixed to each architecture's strongest recent broad-contract variant
2. which scheduler shape gives the best probe behavior by epoch under a fixed wall-clock budget
3. whether any of the higher-LR conditions diverge or become unstable early enough to reject

## Fixed Contract

- Source:
  - `data/merged_deduped.tsv`
- Allele panel:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- Measurement profile:
  - `numeric_no_qualitative`
- Included assay families:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- Qualifier policy:
  - `qualifier_filter=all`
  - censor-aware loss
- Train/val split:
  - expected `32,855 / 8,194`
- Pretraining:
  - `mhc-pretrain-20260308b`
- Training:
  - `20` epochs
  - batch size `256`
  - GPU `H100!`
  - no synthetics
  - no ranking

## Architecture Conditions

### `A03-log10-100k`
- positional base: `P04`
  - `concat(start,end,nterm_frac,cterm_frac)` for peptide, groove1, groove2
- assay head:
  - `shared_base_segment_residual`
- KD grouping:
  - `split_kd_proxy`
- target space:
  - `log10`
  - `max_affinity_nM=100000`

### `A07-mhcflurry-100k`
- positional base: `P04`
  - `concat(start,end,nterm_frac,cterm_frac)` for peptide, groove1, groove2
- assay head:
  - `shared_base_factorized_context_plus_segment_residual`
- KD grouping:
  - `split_kd_proxy`
- target space:
  - `mhcflurry`
  - `max_affinity_nM=100000`

## LR / Schedule Grid

### Initial learning rates
- `1e-4`
- `2.8e-4`
- `8e-4`

### Schedules
- `constant`
- `warmup_cosine`
- `onecycle`

Total:
- `2 architectures x 3 LRs x 3 schedules = 18 runs`

## Logging Requirements

Every epoch must record:
- `train_loss`
- `val_loss`
- current LR
- epoch wall-clock
- forward/loss time
- backward time
- data wait time
- optimizer step time
- GPU util / memory metrics
- probe outputs for:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`

Instability capture:
- detect non-finite train loss
- detect non-finite val loss
- detect non-finite gradients before optimizer step
- record `diverged=true` and `divergence_epoch`
- stop the run early if a divergence condition is hit

## Deliverables

- experiment dir under `experiments/`
- `README.md`
- `manifest.json`
- `variants.md`
- parsed per-epoch metrics CSV/JSON
- comparison table
- at least:
  - loss-curve plot grid
  - LR-curve plot grid
  - probe-ratio-over-epochs plot
- canonical update in `experiments/experiment_log.md`

## Decision Rule

Primary ranking:
1. full probe panel correctness and margin by epoch 20
2. terminal validation loss
3. time-to-good-performance across epochs
4. stability / absence of divergence

Secondary ranking:
- epoch-5 and epoch-10 probe behavior, to identify faster-converging conditions
