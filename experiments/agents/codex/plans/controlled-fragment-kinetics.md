# Controlled fragment kinetics: smallest matched exact / sampled model

Prospective design, October 23, 2026. Source qualification is pending under
Hitlist #759 (MAPP) and #770 (purified-protein time courses). This is not a new
experiment registration or a claim that either source has usable training labels.

## Hypothesis and scope

Test whether shared sequence-dependent cleavage rates can explain product
appearance and disappearance over time before adding a processive proteasome,
TAP, MHC selection or an APC-specific loss model. Keep the current exact and
REINFORCE references unchanged. A minimal independent-bond model is a deliberately
limited kinetic baseline, not an established description of cellular proteolysis.

For a substrate of length L, assign a positive rate lambda_k to each internal
bond k. Start with a constant-rate control and a small shared function of the
residues surrounding the bond. Introduce enzyme/context variables only when
the admitted training source exposes them; a single enzyme isotype cannot train
isotype effects. Time is supplied in physical units. No peptide ID feature.

For a contiguous product [i,j), the expected copies per initial substrate are:

```
m(i,j,t) = B(i,t) * B(j,t) * exp(-t * sum(lambda_k, i < k < j))
B(k,t) = 1 - exp(-lambda_k*t) for an internal boundary
B(0,t) = B(L,t) = 1
```

The formula describes independent irreversible cuts along the starting molecule.
It retains products as well as their subsequent fragmentation. Sum over all
spans, including single residues, sub-ligand products and the intact substrate:
expected residue mass is L; expected fragment count is
`1 + sum_k(1-exp(-lambda_k*t))`. A detector can observe a restricted length range;
the unobserved pool must remain in the process model. Enumerate spans using
prefix sums of bond rates, O(L^2) per time, with chunked evaluation where needed.

The matched discrete generator samples each internal cut with probability
`1-exp(-lambda_k*t)`, splits the sequence at sampled cuts, and returns the complete
fragment multiset. Monte Carlo expected span counts must converge to the formula.
This establishes the same model/support before comparing gradient estimators.
The first version is a distribution of discrete fragment partitions, not a
trajectory model with independently learned transport or cleavage ordering.

## Observation and objective must precede fitting

1. Replay original source intensities, background correction, selection and
   replicate aggregation through shared Hitlist. Keep contiguous versus spliced
   products separate, with mapping ambiguities and excluded rows retained.
2. Inspect repeated observations for a reproducible temporal signal. Supply
   precursor amount and time where measured; distinguish different enzyme:substrate
   ratios. Raw MS response is not molar copies. Do not transfer the narrow SWATH
   response calibration to another instrument/assay without validation.
3. Prefer within-peptide temporal comparisons, which can reduce a time-invariant
   peptide-response nuisance. State any observed reference-time calibration as an
   input at evaluation. Such a test is conditional trajectory prediction, not
   zero-shot abundance prediction for an unseen peptide. Freeze the precise
   likelihood, nuisance treatment, selection conditioning and splits only after
   the source schema is qualified, before inspecting held-out outcomes.
4. Guard against trivial solutions. Unnormalized cross-multiplied ratio residuals
   can vanish when all predicted product amounts vanish; they are not sufficient
   by themselves. A free per-peptide amplitude can also hide rate/scale ambiguity.
   Include constant-rate, time-only and no-change controls, parameter-profile or
   sensitivity checks, predicted variation and supported parameter exposure.

## Exact and sampled comparison

Use one frozen observation objective and the same data, features and initialized
parameters. For a squared loss on a linear expected measurement, two independent
sample groups allow an unbiased product estimator of the squared mean residual:
`E[(X_A-y)*(X_B-y)] = (E[X]-y)^2`. Differentiate its expectation with the policy
score and handle differentiable observation parameters explicitly. Individual
estimates can be negative; do not clip them into a different objective.

For log/ratio objectives, plugging finite Monte Carlo means into a logarithm or
denominator is generally biased. Either measure that finite-sample bias against
the exact gradient, use a justified estimator, or name an exact-denominator
hybrid explicitly. Do not call ordinary nonlinear plug-in REINFORCE unbiased.

First verify analytical mass conservation, sample-count convergence, meaningful
parameter recovery on a registered synthetic fixture and gradient agreement.
Then run a bounded biological comparison with source-backed held-out groups.
Report gradient error, distribution/trajectory error, score evaluations, runtime,
memory and variation. GRPO-style normalization/clipping/reuse/reference penalties
remain separate declared interventions after this shared objective works.

## Decision boundary

Only expand to processivity, substrate entry/exit, secondary peptidases and
machinery-specific effects if the baseline fails an identified supported test.
Do not infer cellular clearance, APC transport or concentrations from purified
digestion. A successful conditional time-course fit is progress toward the full
availability goal, not its completion.
