# Positional Composition Bakeoff

## Goal

Evaluate whether the useful signal in current positional encodings comes from:
- start-index alone
- end-distance alone
- additive two-sided coordinates
- concatenative coordinate composition
- low-capacity learned composition over the same raw coordinates

Do this on the same broad 7-allele class-I binding contract for:
- canonical Presto (`P03`-style)
- groove-transformer control (`G1`-style)

The point is to answer a focused question:
- are the current `triple` / `triple_plus_abs` gains real biologic coordinate gains
- or just overparameterized additive duplication over the same index

## Matched Dataset Contract

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
  - exact and censored rows are both included
- Expected split:
  - `32,855` train
  - `8,194` val
- Probe peptides:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`

## Output / Loss Contract

All conditions should keep the same assay/output contract.

Outputs:
- `KD_nM`
- `IC50_nM`
- `EC50_nM`

Assay label -> output mapping:
- direct `KD` and `KD (~IC50)` / `KD (~EC50)` rows supervise `KD_nM`
- `IC50` rows supervise `IC50_nM`
- `EC50` rows supervise `EC50_nM`

Qualifier handling:
- censor-aware loss
- exact rows use squared error in target space
- `>` rows only penalize predictions that are too strong

Unless otherwise stated:
- no synthetics
- no ranking losses
- warm start enabled

## Pretraining / Training Contract

Warm start:
- MHC-only class/species pretrain checkpoint:
  - `mhc-pretrain-20260308b`

Training:
- `3` epochs for the first bakeoff
- same batch size for all compared conditions within a model family
- same optimizer / LR / weight decay as the current broad-contract sweep
- same seed per family

## Positional Composition Conditions

Each condition should be implemented as a composition over raw scalar coordinates,
not by increasing token width arbitrarily or by duplicating the same additive table
without a hypothesis.

Raw coordinates:
- `start_idx`
- `end_dist`
- `nterm_frac`
- `cterm_frac`

Conditions:

1. `start_only`
- learned embedding / projection of `start_idx` only

2. `end_only`
- learned embedding / projection of `end_dist` only

3. `start_plus_end`
- additive composition of independent start and end embeddings

4. `concat_start_end`
- concatenate learned start and end embeddings, then project back to `d_model`

5. `concat_start_end_frac`
- concatenate learned start/end embeddings plus scalar fractions, then project back to `d_model`

6. `mlp_start_end`
- MLP over concatenated start and end embeddings

7. `mlp_start_end_frac`
- MLP over concatenated start/end embeddings plus scalar fractions

8. `triple_baseline`
- current triple-style baseline
- for peptide:
  - start + end + frac
- for groove:
  - start + end + frac

## Model Families

### Family A: Canonical Presto (`P03`-style)

Keep fixed:
- `shared_base_segment_residual`
- `binding_core_lengths=8,9,10,11`
- broad numeric contract

Change only:
- positional-composition mode used for peptide / groove coordinates

Recommended first matrix:
- peptide and groove both use the same composition condition

Optional phase 2:
- hold peptide fixed at best setting
- sweep groove only
- then hold groove fixed and sweep peptide only

### Family B: Groove Transformer (`G1`-style)

Keep fixed:
- three segment encoders
- same data contract
- same outputs / loss contract as closely as feasible

Change only:
- positional-composition mode used inside each segment encoder

## Runtime / Evaluation to Record

For every condition:
- model family
- positional-composition condition
- parameter count
- startup / setup wall-clock
- mean epoch wall-clock
- forward/loss time
- backward time
- data wait time
- GPU utilization / occupancy proxy if available
- best validation loss
- probe IC50s:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- key ratios:
  - `A24 / A02` for `SLLQHLIGL`
  - `A24 / A02` for `FLRYLLFGI`
  - `A02 / A24` for `NFLIKFLLI`

## Success Criteria

Primary:
- find whether `triple` really beats simpler coordinate compositions
- identify whether additive duplication was masking a better composition rule

Secondary:
- find a composition that improves broad-contract probe behavior without hurting validation loss badly

## Deliverables

- one top-level experiment dir under `experiments/`
- `README.md` with:
  - dataset contract
  - training contract
  - output / assay mapping
  - comparison table
  - takeaways
- `options_vs_perf.md`
- plots of probe trajectories if practical

## Notes

- `abs_only` is effectively a `start_only` family member if implemented as start-index-only absolute position.
- The experiment should make dimensionality fair:
  - every condition must project back to the same token width / `d_model`
  - no condition should win simply by increasing activation width.
