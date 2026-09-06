# PR #45 lineage and funnel closure preflight

## Question

Does the reviewed code preserve standalone DQ/DP beta alleles as resolved
model-facing lineage through the default `mhcseqs` path, and does the real
Hitlist HLA-A*02:01 flank-enabled load expose every unresolved-flank removal in
the canonical funnel JSON and CSV?

## Fixed contract

- Code commit: `15003dacf310de3534325cb2db7fd6f3e2e89481`
- Hardware: local CPU; no GPU
- MHC source: default `mhcseqs`; no index CSV and no fallback source
- Hitlist snapshot: `HITLIST_DATA_DIR=/Users/iskander/.hitlist`
- Hitlist selection: HLA-A*02:01, masking policy, flanks enabled
- Curation seed / split seed: 42 / 42
- Caps: 96 binding, 2 kinetics, 24 stability, 96 elution
- Synthetic data and MHC augmentation: disabled
- Output mode: data preflight only; no training or predictive evaluation

## Acceptance gates

- Real `mhcseqs` records for `HLA-DQB1*06:02` and `HLA-DPB1*02:01` report
  `chain=beta`, export only `groove2`, populate `mhc_b`, and appear in both
  sample and collated resolved lineage.
- `data_funnel.json` contains
  `drop_reasons.unresolved_flank={"binding":4,"ms":235}`.
- `data_funnel.csv` contains the same two rows.
- Split integrity, traceable-lineage, and fake-null gates pass.
- The experiment bundle records the exact command, commit, dataset contract,
  and output hashes. No held-out predictions are expected because this is a
  deterministic data preflight, not a predictive evaluation.
