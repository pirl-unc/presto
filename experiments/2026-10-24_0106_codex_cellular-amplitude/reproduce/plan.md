# Train-only monotone amplitude calibration of frozen cellular predictions

## Question and limit

Family2026-10-24_0043 transfers purified-digestion cleavage rates to MAPP with a
single learned removal parameter. All three physical fits converge in that scalar
but lose to uniform observed composition. Their outputs vary; wider-context
validation SD.57088 exceeds targetSD.35130. Positive pooled rank correlations in
the simpler controls do not establish accurate relative amounts.

The next bounded diagnostic asks whether a single training-only monotone amplitude
calibration retains useful sequence information. This is a downstream empirical
measurement-link diagnostic, not new biological kinetics or independent MS-response
calibration. It cannot certify the true physical amount distribution.

## Fixed source/model contract

Use only the verified train/validation prediction rows from0043 and their original
canonical Hitlist recipe/source hashes. Keep all72trainparents/144examples and
15historically exposed validationparents/30examples; exclude14historicaltestparents.
No additional source import, curation, labels, filtering or target transformation.
All imported physical rates AND their fitted removal parameters remain frozen.
Do not transfer the purified-study MS-response coefficients.

For each of three physical arms (uniform, adjacent, wider), let p be its original
centeredlog10prediction and y the original centeredlog10MS target. Fit one scalar:

    beta = max(0, sum_train w*p*y / sum_train w*p*p)
    prediction = beta*p

Use the exact existing equal-parent/condition/candidate weights. No intercept is
needed because both vectors are centered within each example. No penalty, parameter
search or validation-derived calibration. If the exact denominator is zero, report
nonidentification and use the existing uniform composition control rather than an
invented floor. Preserve the signed unconstrained coefficient as a diagnostic,
but the predeclared primary prediction uses the nonnegative coefficient. Beta0 is
a legitimate rejection of useful monotone signal, not a reason to force variation.

Compare original uncalibrated arms, the three calibrated arms and uniform observed
composition. The meaningful transfer test is whether an imported sequence prior
beats BOTH uniform composition and the calibrated uniform-cleavage control on
validation; reducing error relative only to its poorly calibrated original is
insufficient. Report magnitude, parents helped/hurt, pooled/weighted distinction,
and lack of fresh blind evidence. Do not select a calibration variant by validation.

## Execution and verification

Create a new registered experiment directory before computing any beta or metric.
Local CPU, no package/environment changes,60seconds/2GiB producer budget. No Adam,
no new pretraining and no sampled fit for this one-dimensional analytic diagnostic.
Independently reconstruct source weights/targets and original frozen predictions;
verify the scalar through weighted least squares plus the boundary KKT condition.
Recompute all final metrics and source-linked per-example predictions; preserve
all raw coefficients, cross moments and exact input hashes. Report no test results
with the explicit historical-exclusion reason. Registry/source bundle closure as
usual. This diagnostic is not the requested final availability model.

## Decision and broader goal

If no frozen sequence arm beats both calibrated controls, stop extending this
transfer bridge on the same MAPP development set. Prioritize canonical native
intracellular peptidomics admission (Hitlist788), broader independent MS-response
support (95/762), and source-backed cellular machinery/precursor context. Do not
relabel captured fragments as free cytosol or drug exposure as gene knockout.
If a sequence arm does pass, preregister independent cellular/source replication
before promoting it, and then a matched exact/sampled training comparison with
those observation assumptions fixed. No claim of APC concentrations either way.

The existing conditional sampled estimator is numerically sound but its per-
condition full-partition implementation exceeds the declared compute budget.
Any future efficiency change should have a new prospective cost/gradient gate;
possible exact marginalization of unqueried bonds or shared draws across matching
conditions changes the sampler implementation and must not be slipped into0043.
A numerically cheaper sampler is only useful after the biological objective has
credible predictive support. Full APC availability/concentration objective remains
active; no success criterion is narrowed to this calibration diagnostic.
