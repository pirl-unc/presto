# EXP-21 Honest No-Assay Repeat

## Goal

Repeat the old EXP-21 benchmark family under an honest sequence-only input contract to test whether the apparent groove advantage survives once assay-selector inputs are removed.

This plan is intentionally narrower than a full canonical-output rebuild:

- it preserves the legacy distributional benchmark head family
- it disables assay-selector inputs completely
- it does **not** claim that the legacy benchmark output contract is the final Presto modeling contract

So this experiment answers:

- did the old `groove c02` win survive once the forbidden assay input leak is removed?

It does **not** by itself answer:

- is the legacy distributional benchmark the right final output contract for Presto?

That second question is already better approximated by the honest full-output PF07 runs.

## Fixed Contract

- source: `data/merged_deduped.tsv`
- alleles: `HLA-A*02:01`, `HLA-A*24:02`
- measurement profile: `numeric_no_qualitative`
- assay families present in the dataset slice:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- qualifier filter: `all`
- batch size: `256`
- optimizer: `AdamW`
- lr: `1e-3`
- weight decay: `0.01`
- epochs: `50`
- requested Modal GPU: `H100!`

## Model/Input Contract

- runner: `presto.scripts.distributional_ba.train`
- config version: `v6`
- assay input mode: `none`
- content conditioning: `false`
- sequence inputs only:
  - `peptide`
  - `mhc_a`
  - `mhc_b`
- note:
  - this benchmark path does not use the canonical full Presto multi-output assay head family
  - the purpose here is apples-to-apples leak removal for the old EXP-21 family

## Conditions

Run the same 50-epoch / seed-43 point for the three EXP-21 model families:

1. `groove c02`
2. `groove c01`
3. `historical c02`

## Required Outcome

- new experiment directory with reproducibility bundle
- locally fetched raw run artifacts under `results/runs/`
- aggregated summary tables and plots
- explicit comparison against:
  - old cheating EXP-21 results
  - current honest PF07 full-output result
- clear conclusion:
  - whether `groove c02` still wins under honest inputs
  - whether any honest legacy-benchmark run beats the current honest PF07 baseline
