# Training-only between-curve and temporal residual decomposition

## Question and unchanged scope

The new exactly profiled analytical fit improves real development shape RMSE,
but train shape RMSE remains 1.16435 and predictions have limited spread. While
the matched fragment fit remains live, determine whether the residual training
MSE is mainly a persistent curve-level signal mismatch or within-curve temporal
mismatch. This is an arithmetic diagnostic, not a new predictor, fit, biological
noise ceiling, attribution to MS versus kinetics, or concentration calibration.
Do not inspect running fragment checkpoints or select a new model from them.

## Fixed comparisons and source

Use completed, independently verified prediction dumps: previous 200-step
joint-offset analytical and fragment, new 200-step profiled analytical, older
2,000-full-batch exact and conditional sampled-hybrid. Include the same constant
condition-scale baseline. Hash all input files before calculation. Select only
original training rows, preserve every condition/construct/curve/time/target and
weight, and require exact measurement identity/target equality across files.
Complete source files include development rows; parse them but never consult
those quantities for this diagnostic. Original test remains excluded. No new
source importer or independent biological label.

## Mathematics and retained evidence

For condition-centered residual r_i = prediction_i - target_i and original
hierarchical weights w_i, group by original curve_id, which includes source
biological-label identity. Define W_c=sum_i(w_i) and m_c=sum_i(w_i*r_i)/W_c.

  total MSE = sum_c W_c*m_c^2 + sum_i w_i*(r_i-m_c)^2.

The first term is persistent between-curve error; the second is within-curve
(time-varying) error. Report both MSEs and fractions overall and per construct and
condition, with exact sum/cross-term checks. A curve is not necessarily one unique
molecular form; keep original source ambiguity. Do not label the first component
pure detectability or the second pure biological processing. Time-varying
measurement effects and erroneous persistent kinetics can contribute to either.

Store curve-level weights/means/component sums, input hashes and exact metadata;
original full predictions remain the per-example evidence. No new validation/test
metrics are produced because this diagnostic is strictly training-only and is not
an evaluation experiment. No threshold/nM metrics apply. Compare component changes
between the fixed models, not just the largest fraction in one fit.

## Verification, budget and decision

Producer uses Python groups and stable sums. Independent verifier uses NumPy
integer grouping/bincount from original prediction files, rechecks all source
identities/weights/targets and verifies all components, totals and reported
fractions at <=1e-10 absolute plus1e-9 relative tolerance. Include a nonuniform
weight toy with both components nonzero; verify identity and wrong-unweighted-mean
counterexample. One local CPU, no environment/cloud change, no optimization,
120-second/4GiB diagnostic cap; preserve failure rather than dropping rows.

Use results to rank the next biological/observation hypothesis after the live
paired fit finishes. This diagnostic alone must not promote a more flexible
response layer, add per-peptide offsets to production, or infer absolute amounts.
The full APC availability goal and independent calibration/machinery validation
remain unresolved. Publish even if decomposition does not distinguish next steps.
