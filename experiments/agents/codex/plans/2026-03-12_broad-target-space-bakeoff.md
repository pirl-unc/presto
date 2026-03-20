# Broad Target-Space Bakeoff

## Goal

Compare the two leading canonical broad-contract Presto hypotheses:
- `A03`
- `A07`

across four affinity target spaces on a fixed broad numeric class-I contract.

This bakeoff is intended to answer:
1. whether weak-tail calibration improves with a `100k nM` cap
2. whether MHCflurry-style bounded targets outperform plain `log10(nM)` on the broad contract
3. whether `A03` or `A07` is the better canonical Presto base once target space is controlled

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
- Excluded:
  - qualitative binding
- Qualifier policy:
  - `qualifier_filter=all`
  - use censor-aware loss
- Train/val split:
  - expected `32,855 / 8,194`
- Pretraining:
  - `mhc-pretrain-20260308b`
- Training:
  - `5` epochs
  - batch size `256`
  - GPU `H100!`
  - no synthetics
  - no ranking

## Structural Conditions

### `A03`
- positional base: `P04`
  - `concat(start,end,nterm_frac,cterm_frac)` for peptide, groove1, groove2
- assay head:
  - `shared_base_segment_residual`
- KD grouping:
  - `split_kd_proxy`

### `A07`
- positional base: `P04`
  - `concat(start,end,nterm_frac,cterm_frac)` for peptide, groove1, groove2
- assay head:
  - `shared_base_factorized_context_plus_segment_residual`
- KD grouping:
  - `split_kd_proxy`

## Target-Space Conditions

1. `log10_50k`
- `affinity_target_encoding=log10`
- `max_affinity_nM=50000`

2. `log10_100k`
- `affinity_target_encoding=log10`
- `max_affinity_nM=100000`

3. `mhcflurry_50k`
- `affinity_target_encoding=mhcflurry`
- `max_affinity_nM=50000`

4. `mhcflurry_100k`
- `affinity_target_encoding=mhcflurry`
- `max_affinity_nM=100000`

## Modeling Requirement

For `mhcflurry` conditions:
- residuals must be applied in encoded target/logit space
- not added directly to the post-clamped `[0,1]` bounded target
- outputs should still be reported canonically in log10(nM) / nM after inverting the encoded target

This is the key fairness fix for the target-space comparison.

## Evaluation

For each run record:
- startup/setup wall-clock
- per-epoch wall-clock
- GPU utilization / memory metrics if available
- best and final validation loss
- probe outputs:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- probe ratios:
  - `A24/A02` for `SLLQHLIGL`
  - `A24/A02` for `FLRYLLFGI`
  - `A02/A24` for `NFLIKFLLI`

## Deliverables

- experiment dir under `experiments/`
- `README.md`
- `manifest.json`
- `variants.md`
- `options_vs_perf.md`
- parsed metrics CSV/JSON
- plots if useful
- canonical update in `experiments/experiment_log.md`

## Decision Rule

Primary ranking:
1. probe panel correctness and margin
2. validation loss
3. runtime

If one of the MHCflurry variants clearly wins, carry that target space forward into the next broad-contract canonical Presto sweep.
