# Assay Head And Target-Space Bakeoff

## Goal

Test whether the best broad-contract canonical Presto should use:
- one pooled affinity output for all numeric binding assays
- or a shared base affinity with assay-specific residuals / biases

Also test whether a different target space improves weak-tail calibration:
- canonical `log10(nM)`
- bounded MHCflurry-style target
- `50k` vs `100k` max affinity cap

Also test whether the current merged-KD family is too coarse by comparing:
- one merged KD family
- split KD-family supervision:
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`

and whether a factorized assay context works better than the current flatter assay-method encoding.

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
  - exact and censored rows included
- Expected split:
  - `32,855` train
  - `8,194` val

## Source Assay Semantics

These must stay explicit in every writeup because the five numeric families are not interchangeable.

Raw source families in the current merged corpus:
- `half maximal inhibitory concentration (IC50)`
- `dissociation constant KD`
- `dissociation constant KD (~IC50)`
- `dissociation constant KD (~EC50)`
- `half maximal effective concentration (EC50)`

Typical method axes recoverable from source fields:
- preparation:
  - `purified`
  - `cellular`
  - `lysate`
  - `binding_assay`
- geometry:
  - `competitive`
  - `direct`
  - `t_cell_inhibition`
  - `unknown`
- readout:
  - `radioactivity`
  - `fluorescence`
  - `unknown`

## Current Assay Mapping

This must be made explicit in every writeup because the broad contract is not one homogeneous assay.

Current grouped supervision:
- `IC50` rows -> `binding_ic50` -> `assays.IC50_nM`
- `EC50` rows -> `binding_ec50` -> `assays.EC50_nM`
- direct `KD` rows -> `binding_kd` -> `assays.KD_nM`
- `KD (~IC50)` rows -> `binding_kd` -> `assays.KD_nM`
- `KD (~EC50)` rows -> `binding_kd` -> `assays.KD_nM`

Current censor behavior:
- exact rows: squared error in target space
- `>` rows: only penalize over-strong predictions

This is one of the assumptions under test. The planned bakeoff should explicitly compare:
- merged KD supervision
- split KD-family supervision
- pooled numeric supervision

## Current Canonical Structure

For the better broad-contract Presto variants, the current structure is:
- one shared KD/base latent
- `KD_nM` from that base
- `IC50_nM` = base + learned residual
- `EC50_nM` = base + learned residual

In `shared_base_segment_residual` mode, the residual input is:
- segment summary vector
- assay context vector
- scalar affinity score

## Conditions To Compare

### Axis A: Assay Head Structure

1. `pooled_single_output`
- one shared affinity output
- all numeric assay families supervise the same scalar
- no assay-specific residual heads

2. `shared_base_residual`
- current idea:
  - one shared base logit
  - assay-specific residuals added to it

3. `shared_base_context_residual`
- shared base logit
- assay-specific residuals conditioned on assay context only

4. `shared_base_segment_residual`
- shared base logit
- assay-specific residuals conditioned on:
  - segment summary
  - assay context
  - scalar affinity score

5. `shared_base_pooled_segment_residual`
- same as above, but residuals read only mean-pooled peptide and mean-pooled MHC summaries
- intended to test whether simpler pooled residual conditioning is enough

6. `shared_base_factorized_context_residual`
- one shared base scalar
- residual conditioned on factorized assay context embeddings:
  - assay family embedding
  - prep embedding
  - geometry embedding
  - readout embedding
- this is the closest test of:
  - `pmhc -> shared affinity latent -> base scalar`
  - assay-factor embeddings -> residual/bias on top

7. `shared_base_factorized_context_plus_segment_residual`
- same factorized context embeddings as above
- plus pooled peptide and pooled MHC summaries
- intended to test whether segment summaries and assay factorization are complementary

### Axis A2: KD Family Grouping

1. `merged_kd`
- current behavior:
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  all supervise `KD_nM`

2. `split_kd_proxy`
- direct `KD` supervises `KD_nM`
- `KD (~IC50)` supervises a distinct `KD_proxy_ic50_nM`
- `KD (~EC50)` supervises a distinct `KD_proxy_ec50_nM`
- `IC50` and `EC50` stay separate as before

3. `factorized_residual_split_kd`
- same split as above
- but direct and proxy-KD families share a common base scalar and use family-specific residuals
- intended to test whether full head splitting is too sparse while still respecting assay differences

### Axis B: Target Space

1. `log10_50k`
2. `log10_100k`
3. `mhcflurry_50k`
4. `mhcflurry_100k`

## Pretraining / Training Contract

- Warm start:
  - `mhc-pretrain-20260308b`
- First bakeoff:
  - `3` epochs
- Same optimizer / batch size / seed across all conditions
- No synthetics in the first bakeoff
- No ranking in the first bakeoff

Phase-2 extension after the clean bakeoff:
- keep the winning assay-head/target-space design
- add back:
  - peptide ranking only
  - then safe synthetic mode(s) one at a time if still justified
- do not mix ranking and synthetics in the first pass

### Phase 1b: Distributional Bin-Count Sweep

This sweep should happen only after the clean rerun of the distributional-vs-regression benchmark has identified a sane distributional family. Based on the completed diagnostic sweep in `2026-03-13_0728_claude_distributional-ba-heads`, the working assumption is:

- keep:
  - `mhcflurry`
  - `log_mse`
  - `twohot_d2_logit`
  - `hlgauss_d2_logit`
- drop for now:
  - `D1-affine`

Goal:
- test whether coarse discrete affinity support is actually better calibrated / easier to optimize on the mixed numeric+censored binding contract
- specifically evaluate whether fewer bins improve:
  - censor-aware survival likelihood stability
  - weak-tail calibration
  - overall runtime

Matched contract:
- same broad 7-allele class-I numeric binding contract
- same warm start
- same optimizer / LR / schedule
- same batch size
- same epoch budget
- no synthetics
- no ranking

Conditions:
- regression baselines:
  - `mhcflurry_{50k,200k}`
  - `log_mse_{50k,200k}`
- distributional D2-logit:
  - `twohot_d2_logit_{50k,200k}_K{8,16,32,64,128}`
  - `hlgauss_d2_logit_{50k,200k}_K{8,16,32,64,128}`

Metrics to compare:
- aggregate:
  - validation and test loss
  - Spearman
  - Pearson
  - RMSE / MSE in `log1p` or target space
  - `<=500 nM` accuracy
  - balanced accuracy
  - AUROC
  - AUPRC
- probe behavior:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- distributional only:
  - entropy
  - interval coverage
  - PIT / calibration summaries

Primary question:
- do coarse bin counts like `8/16/32` outperform `64/128` for censored BA prediction on mixed numeric assay data?

## Proposed Experiment Phases

### Phase 1: Structure-only clean bakeoff

Goal:
- isolate the assay/output structure question without synthetic or ranking confounds

Conditions:
- head structure:
  - `pooled_single_output`
  - `shared_base_segment_residual`
  - `shared_base_factorized_context_residual`
  - `shared_base_factorized_context_plus_segment_residual`
- KD grouping:
  - `merged_kd`
  - `split_kd_proxy`
- target spaces:
  - `log10_50k`
  - `log10_100k`

Recommended matrix:
- start with 8 conditions:
  - 4 head structures x 2 KD grouping choices
  - all in `log10_50k`
- then carry the best 2 structures into:
  - `log10_100k`
  - `mhcflurry_50k`
  - `mhcflurry_100k`

### Phase 2: Assay-factorization refinement

Goal:
- determine whether factorized assay context actually buys something over the current flat method/category encoding

Conditions:
- flat assay-context embedding
- factorized embeddings:
  - family + prep + geometry + readout
- factorized embeddings + pooled segment residual

### Phase 3: Training-contract additions

Goal:
- only after a clean structural winner is chosen, test whether:
  - peptide ranking
  - censored rows
  - safe synthetics
  help or hurt on top of it

Conditions:
- winner + no extras
- winner + peptide ranking
- winner + one safe synthetic mode
- winner + peptide ranking + one safe synthetic mode (last, only if needed)

## Evaluation

For each condition:
- parameter count
- startup/setup wall-clock
- mean epoch wall-clock
- validation loss
- probe IC50s:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- probe ratios
- explicit assay-family metrics where possible:
  - per-family loss for:
    - `IC50`
    - direct `KD`
    - `KD (~IC50)`
    - `KD (~EC50)`
    - `EC50`
- report whether direct `KD` and the proxy-KD families move together or diverge
- note clearly when validation loss is not directly comparable because target space differs

## Success Criteria

- Determine whether separate assay heads are actually helping beyond a pooled output
- Determine whether `100k` improves weak-tail calibration without destroying the rest of the probe panel
- Determine whether MHCflurry-style bounded targets improve broad-contract performance or only change calibration
- Determine whether direct `KD` should stay merged with proxy-KD families
- Determine whether factorized assay context is worth the extra complexity

## Deliverables

- experiment directory under `experiments/`
- README with explicit assay-family -> output mapping
- explicit source-method factorization description:
  - family
  - prep
  - geometry
  - readout
- comparison tables by condition
- takeaway about the best assay head structure and target space for broad IEDB binding
