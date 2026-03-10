# Affinity Follow-Up Plan (2026-03-09)

## Goal

Push the focused affinity path from "correct ordering" to "credible weak-binder calibration", then decide how to incorporate qualitative binding and eventually class-II without breaking class-I.

## Current facts

- Warm start means:
  - 1 epoch MHC-only pretraining on groove sequences
  - targets: MHC class + species category
  - checkpoint reuse into focused affinity training
- Focused affinity outputs are trained in `log10(nM)` space.
- The current affinity cap is:
  - `DEFAULT_MAX_AFFINITY_NM = 50000`
- The focused trainer already supports censor-aware losses on `<`, `=`, `>` measurements.
- The cleanest strong results so far are:
  - `E004`: 2-allele exact-IC50 + warm start + synthetic negatives
  - `E006`: broader class-I exact-IC50 + warm start + no synthetic negatives
  - `M1`: best architecture benchmark winner for class-I preservation

## Synthetic negative taxonomy

Keep these concepts separate:

1. Anchor-position change
- ensure `P1/P2/PΩ` differ from the original peptide
- weak guarantee: changed residues may still be valid anchors

2. Anchor-opposite perturbation
- only for strong class-I binders
- force `P2/PΩ` into opposite biophysical classes
- stronger guarantee against false-negative synthetics
- larger risk of over-regularization

3. MHC-context corruption
- scramble or randomize MHC
- or remove one MHC chain
- teaches "this peptide is not generally binding absent the right groove context"

## Qualitative binding facts

Current merged-corpus audit for `value_type = qualitative binding`:

- total rows: `88739`
- source: `iedb` only
- label vocabulary:
  - `Positive`: `44454`
  - `Negative`: `35619`
  - `Positive-High`: `3350`
  - `Positive-Low`: `3260`
  - `Positive-Intermediate`: `2056`

Interpretation:

- this is not purely binary
- but it is not a fully consistent ordinal scale either
- safe partial order:
  - `Positive-High > Positive-Intermediate > Positive-Low > Negative`
  - `Positive > Negative`
- unsafe without more policy:
  - exact placement of plain `Positive` relative to the graded positive labels

## Experiments

### A. Combined class-I panel run

Purpose:
- test the user-requested combined recipe directly

Contract:
- warm start on
- broader class-I panel
- exact `IC50`
- peptide ranking on
- allele ranking off
- synthetic negatives on
- anchor-aware strategy `property_opposite`

Run:
- `class1-panel-ic50-exact-warmstart-synth-peprank-20260309a`

Success criteria:
- preserve correct signs on:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- improve weak-binder calibration over `E006`
- avoid clear regression in validation loss versus `E004/E006`

### B. Synthetic ablation on the broader class-I panel

Purpose:
- isolate whether the gain is from synthetics, peptide ranking, or their interaction

Matrix:
- `B1`: warm start + broader panel + no synthetics + no peptide ranking
  - existing comparator: `E006`
- `B2`: warm start + broader panel + synthetics only
- `B3`: warm start + broader panel + peptide ranking only
- `B4`: warm start + broader panel + synthetics + peptide ranking
  - active comparator: `class1-panel-ic50-exact-warmstart-synth-peprank-20260309a`

Evaluation:
- best checkpoint probe values
- weak-binder tail calibration
- validation loss

### C. Anchor-strategy ablation

Purpose:
- determine whether `property_opposite` is actually helping or just making negatives artificially easy

Matrix:
- `C1`: `class_i_anchor_strategy=none`
- `C2`: `class_i_anchor_strategy=property_opposite`

Constant recipe:
- broader class-I panel
- warm start
- synthetics on
- peptide ranking on
- allele ranking off

Evaluation:
- `SLLQHLIGL` / `FLRYLLFGI` / `NFLIKFLLI`
- especially weak-binder calibration for held-out `A*24:02`

### D. Censor-aware broad quantitative run

Purpose:
- test whether exact-only filtering is leaving too much useful signal on the table

Matrix:
- `D1`: exact-only `IC50`
- `D2`: all `IC50` with censor-aware loss
- `D3`: all direct quantitative `IC50/KD/EC50` with assay-specific outputs and censor-aware loss

Evaluation:
- per-assay validation loss
- class-I probes
- whether mixed direct quantitative supervision hurts or helps weak-binder calibration

### E. Qualitative binding as auxiliary supervision

Purpose:
- add qualitative data without corrupting the quantitative affinity target

Designs to test:

1. Ordinal pairwise only
- use partial order pairs:
  - `Positive-High > Positive-Intermediate`
  - `Positive-Intermediate > Positive-Low`
  - `Positive-Low > Negative`
  - `Positive > Negative`
- do not assign synthetic numeric nM values

2. Categorical output head
- 5-class qualitative binding head
- keep quantitative affinity head separate
- useful mainly as auxiliary regularization

3. Weak pseudo-numeric mapping
- only if 1 and 2 are insufficient
- conservative bins:
  - `Positive-High -> <250 nM`
  - `Positive-Intermediate -> <1000 nM`
  - `Positive-Low -> <5000 nM`
  - `Positive -> <5000 nM`
  - `Negative -> >5000 nM`

Recommendation:
- test `ordinal pairwise` first
- categorical head second
- pseudo-numeric last, because it bakes in stronger assumptions

### F. Class-II entry test

Purpose:
- determine whether adding quantitative class-II affinity breaks class-I

Only start after A-D are stable.

Entrants:
- Stage-A architecture winner `M1`
- optional fallback `M2`

Recipe:
- keep class-I probe panel fixed
- add class-II exact `IC50` first
- then add broader class-II quantitative data if class-I survives

Success criteria:
- no material regression on class-I probe ordering
- acceptable class-II validation loss

### G. Direct-only affinity/stability factorization

Purpose:
- remove the current pseudo-kinetic bottleneck from canonical affinity prediction
- replace it with a cleaner decomposition that matches the current evidence:
  - `pmhc_interaction_vec -> affinity/stability latent states`
  - assay outputs derived from those latent states
  - qualitative binding/stability supervised as inequality losses on latent scalar scores

Proposed factorization:

- inputs:
  - `{peptide, mhc_groove1, mhc_groove2}`
- shared trunk:
  - `pmhc_interaction_vec`
- latent states:
  - `binding_affinity_vec = f_aff(pmhc_interaction_vec, pep_vec, mhc_a_vec, mhc_b_vec, groove_vec)`
  - `binding_stability_vec = f_stab(pmhc_interaction_vec, pep_vec, mhc_a_vec, mhc_b_vec, groove_vec)`
- latent scalar scores:
  - `binding_affinity_score = g_aff(binding_affinity_vec)`
  - `binding_stability_score = g_stab(binding_stability_vec)`
- quantitative outputs:
  - `{pmhc_interaction_vec, binding_affinity_vec, assay_context} -> {KD_nM, IC50_nM, EC50_nM}`
  - `{pmhc_interaction_vec, binding_stability_vec} -> {t_half, Tm}`
- qualitative outputs:
  - qualitative binding/stability use inequality losses on `binding_affinity_score` / `binding_stability_score`

Explicit removals from the canonical path:
- `BindingModule`
- kinetic latents:
  - `log_koff`
  - `log_kon_intrinsic`
  - `log_kon_chaperone`
- probe/core KD mixing as the main affinity contract

Compatibility plan:
- keep compatibility output aliases during migration
- if `kon/koff` are still needed for legacy checkpoints or old tests, expose them only as optional auxiliary heads, not as the canonical route to `KD_nM`

Why this is attractive:
- better optimization
- fewer biologic assumptions than the current pseudo-kinetic decomposition
- cleaner place to attach qualitative ordinal losses
- more naturally compatible with later class-II register/core work

Main risk:
- removing the structured path may reduce some regularization on sparse quantitative data
- mitigate by testing against:
  - warm start
  - broader class-I panel
  - synthetics / peptide ranking ablations

Evaluation:
- compare against `E004`, `E006`, and `M1`
- require no regression on:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- check weak-binder calibration explicitly

## Evaluation summary

For every run, record:

- dataset slice
- architecture / contract
- synthetic modes
- anchor strategy
- best checkpoint epoch
- best validation loss
- probe values:
  - `SLLQHLIGL`
  - `FLRYLLFGI`
  - `NFLIKFLLI`
- if qualitative data is used:
  - exact label vocabulary and ordering policy

## Immediate order

1. Finish `class1-panel-ic50-exact-warmstart-synth-peprank-20260309a`
2. Run broader-panel synthetics-only ablation
3. Run broader-panel anchor-strategy ablation
4. Run censor-aware direct-quantitative ablation
5. Only then introduce qualitative-binding auxiliaries
6. Only after that, test class-II
