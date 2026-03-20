# 2026-03-15 EXP-16 Main-Path Baseline Rebuild

## Goal

Rebuild EXP-16 as a trustworthy shared-code benchmark after identifying that the current markdown description and the current shared implementation have both drifted away from the raw winning artifact.

This work must do two things at once:

1. restore the historical EXP-16 winner as a first-class shared-code positive control
2. establish a new shared-path baseline by rerunning the same v6 factorial with a modern backbone on the same executable contract

## Historical contract to preserve

The raw winner artifact is:

- `experiments/2026-03-13_1600_claude_v6-factorial-32/data/distributional_ba_v6_c02_cc0_20260313T202238Z/summary.json`

The executable winner contract from that artifact is:

- shared `scripts/distributional_ba` path
- config version `v6`
- condition `2`
- `content_conditioned = false`
- backbone family: historical `AblationEncoder`
- alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
- assay measurement profile: broad numeric (`IC50`, `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`)
- qualifier filter: `all`
- split: peptide-group `80/10/10`, seed `42`
- train / val / test rows: `15530 / 1919 / 1915`
- epochs: `50`
- batch size: `256`
- optimizer: `AdamW`
- lr: `1e-3`
- weight decay: `0.01`
- warm start: none
- winner metrics:
  - test Spearman `0.8435`
  - AUROC `0.9412`
  - RMSE_log10 `0.8304`

## Main inconsistencies to fix

1. The shared code currently uses `GrooveTransformerModel.encode()` in place of the historical `AblationEncoder`.
2. The markdown summaries currently overstate the contract as 7-allele exact-IC50 with warm-start.
3. The current shared path is runnable but no longer parameter-identical to the historical winner.

## Code changes

### 1. Restore explicit encoder backend selection

Add an explicit backbone selector to the shared distributional BA codepath.

Preferred backend names:

- `historical_ablation`
- `groove`

Implementation requirements:

- move encoder construction behind a small shared abstraction
- reintroduce the historical `AblationEncoder` implementation as real code, not an alias
- keep `GrooveTransformerModel.encode()` available as the modern shared-path backend
- ensure `build_model(...)` can instantiate either backend without changing head behavior

### 2. Preserve backward compatibility carefully

- v6 historical positive-control runs must use `historical_ablation`
- new comparison launcher must be able to select either backend explicitly
- avoid silently changing self-contained experiment-local packages; this rebuild is about the shared main path

### 3. Freeze the positive control with tests

Add regression tests covering:

- `v6 cond_id=2` with `historical_ablation` builds the expected label and parameter count
- `historical_ablation` and `groove` both produce valid forward passes for the same condition
- the historical build matches the raw winner contract at least in:
  - encoder family
  - embed dim
  - parameter count

## Experiment design

Create a new experiment family that reruns the v6 factorial on the **actual executable EXP-16 contract** with an added implementation axis.

### Fixed contract

- source: `data/merged_deduped.tsv`
- alleles:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
- assay families:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- qualifier policy: `all`
- split: peptide-group `80/10/10`, seed `42`
- epochs: `50`
- batch size: `256`
- optimizer: `AdamW(weight_decay=0.01)`
- lr: `1e-3`
- warm start: none
- GPU: `H100!`

### Sweep axes

- implementation backend:
  - `historical_ablation` (positive control)
  - `groove` (candidate baseline)
- `content_conditioned`: `{false, true}`
- `embed_dim`: `{32, 64, 128, 256}`
- `head_type`: `{mhcflurry, hlgauss}`
- `max_nM`: `{50000, 100000}`

Total runs:

- `2 × 2 × 4 × 2 × 2 = 64`

## Verification plan

1. Local dry-run for historical positive-control winner:
   - `v6`, `cond_id=2`, `historical_ablation`, `cc0`
2. Local dry-run for groove candidate winner path:
   - `v6`, `cond_id=2`, `groove`, `cc0`
3. Modal smoke run for one condition from each backend.
4. Launch the full 64-run sweep on `H100!`.
5. Collect all raw artifacts locally.
6. Generate condition tables, backend comparison tables, plots, and held-out metrics.
7. Update the canonical experiment log and explicitly correct the older EXP-16 contract description.

## Questions to answer

1. Does the restored historical-ablation backend reproduce the EXP-16 winner neighborhood in the current shared path?
2. Does the groove backend beat the historical positive control on the same 2-allele mixed-assay contract?
3. Is the best new shared-code baseline still `d=32 + mhcflurry + 100k + cc0`, or does the backend change the winner?
4. How much of the previous “EXP-16 winner” reputation was due to the historical ablation backbone rather than the v6 head/context matrix?
