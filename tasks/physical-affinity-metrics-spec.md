# Physical affinity metric reporting — #116

## Observed defects and scope

The immutable 0.1.16 fit reports tiny finite Pearson values for its constant mean
baseline. Four exact 500 nM training labels cross the threshold after float32 log
transformation, promotion to float64 and serialization. Preserve that run,
prospective criteria and the failed quality result. This PR changes reporting,
not training targets, checkpoints, model inputs, splits or optimization.

## Design

- Detect exactly constant input vectors before mean centering in Pearson.
  Preserve the established undefined/omitted result convention; no arbitrary
  tolerance that would erase small but genuine variation.
- Add optional aligned physical nM target values to binding_threshold_metrics.
  When a physical value is available, determine its binding label and usable
  censor bound directly in nM. Otherwise retain the existing transformed-log
  diagnostic and expose how many observations use each basis. Predictions keep
  their existing continuous log10(nM) cutoff; do not invent precision absent
  from an estimated output. Preserve the existing conservative three-valued
  censor-direction interpretation; exact strict/inclusive operators remain
  available in lineage but are not silently reinterpreted here.
- Canonical held-out accumulation already retains raw_target/raw_unit for each
  effective objective. Use those nM values for overall and mapping-stratified
  binding threshold reports. Prefer original source_value when its unit is nM
  and its value agrees with the recorded numeric raw target at its declared
  float32 precision; this retains original physical precision without using an
  unrelated source field. Leave unsupported/missing raw units as explicit
  transformed-only diagnostics. Do not alter nonbinding metrics or losses.
- Persist the reporting policy in held-out summaries so historical reports are
  distinguishable. No changes to old experiment snapshots or public releases.

## Verification and publication

- Constant target and prediction vectors with nonrepresentable means, multiple
  lengths/dtypes/permutations, plus small genuine variation and perfect/inverse
  controls.
- Exact and censored labels at/below/above 500 nM through float32 transformation,
  promotion and CSV serialization; explicit physical values, missing physical
  values, invalid aligned input and unchanged historical no-physical mode.
- Canonical accumulator and mapping-category routes use the same physical basis;
  unrelated units and mismatching source values cannot substitute labels.
- Recompute a new additive numerical audit from the public 2,614-row frozen fit,
  through datacache. Preserve all predictions and raw source values, identify the
  four changed training classification rows and undefined constant correlations,
  and prove validation/test quality and primary regression results unchanged.
- Publish the audit with source/code hashes and no new training. Bump 0.1.18;
  format/lint/focused/full CI, review, merge, clean-main release and public-wheel
  verification. Rebase the isolated feature branch onto merged #113 before PR.
