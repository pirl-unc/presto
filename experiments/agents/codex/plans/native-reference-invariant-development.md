# Native responses: test reference dependence without adding model capacity

Prepared after boundary-context family0726 failed all four fixed development
gates. Acquisition responses were concordant in0711, but149features did not
predict across components. Keep both results and the original screen0630 outer
failure. This is the next support/objective question, not another feature search.

## Hypothesis

One hash-selected reference peptide gives some endpoints disproportionate
influence on the current per-parent loss. A reproducible reference-specific
response can dominate several contrasts without being a transferable sequence
rule. Test whether eliminating arbitrary anchor choice improves the *same*
feature models. Do not presume this is the cause of failed prediction.

The hypothesis changes only the response loss, not the eligible endpoints or
feature map. It does not create independent observations, fix missing absolute
calibration or turn a response model into a concentration model.

## Fixed data and support audit before fitting

Use only recipe0612's original158training contrasts,214unique endpoint/parent
pairs,56parents and55dependency components, package manifest SHA256
`5f1f388876e2b5b3f65d254ff4939d2559077c2e6af1dffe603b05a1283b836a`.
The endpoint set for each parent is its admitted reference plus all admitted
nonreference endpoints. Do not admit new source peptides, impute zeros, alter
acquisitions or remove low-support/high-response parents. All other source
eligibility and metadata limitations remain as in0726.

Before reading effects, enumerate endpoint counts per parent/component and total
unordered pairs. Verify original reference membership, one sequence/parent per
endpoint, fixed weight hierarchy, original component assignment and complete
six-acquisition source quantities. Keep all repeated contributor IDs explicit.
For each pair a,b order IDs lexicographically, define features f(a)-f(b), and
compute target directly from raw intensity log10 ratios across the original
threeKD and threecontrol acquisitions. Independently verify equivalence to the
difference of original reference-anchored targets (reference target=0).

Retain only one orientation of every unordered pair. A parent with n endpoints
has n(n-1)/2 dependent pairs, *not* that many biological samples. No pair crosses
parent, component or original partition. Source-wide dependency groups remain
the splitting and bootstrap unit.

Before fitting, report the weighted fractions of parents/endpoints with n=2,
n=3 and n>=4, the pair count and the number of independent within-parent response
degrees (sum(n-1)=158 before cross-parent dependencies). For n=2 the original
reference loss and all-pair loss are exactly the same, so this part of the cohort
cannot distinguish the hypotheses. Do not set a posthoc exclusion rule.

## Two objectives, fixed features

The original objective uses one reference per parent, equal parent weight and
equal weight over n-1 reference contrasts. The proposed objective uses equal
parent weight and equal weight over all n(n-1)/2 unordered pairs. Component
weights, and parent weights within each component, remain uniform as before.
Fit weights normalize to1 in every fold. Each objective computes its own
training-only weighted RMS feature scales, as specified by that objective.

Use the exact frozen zero/length/61feature/149feature conditions from0726. No
new representation, feature pruning, nonlinear head, identity term or pretrained
encoder. The comparison retains both objectives as fixed references; do not
replace0726 or compare mismatched target spaces as if they were identical.

For endpoint residuals e_i within a parent, explicitly verify:

`mean_{i<j}(e_i-e_j)^2 = 2n/(n-1) * mean_i(e_i-mean(e))^2`.

For a specific anchor a, verify:

`mean_{i!=a}(e_i-e_a)^2 = n/(n-1) * [Var(e)+(mean(e)-e_a)^2]`.

Thus all-pair loss is invariant to anchor and common parent shifts. Its n-dependent
factor must not be silently dropped or treated as equal endpoint weighting.
The same identities must hold for model-minus-target residuals and for the
feature/target quadratic objective and gradients. Verify with synthetic data and
the real training rows before fitting.

## Development fitting and matched evaluation

Reuse the exact five/three component fold IDs, alpha grid[.01,.1,1,10,100], tie
rule, weighted uncentered scaling, no-intercept ridge objective and resource
limits specified in native-boundary-context-development.md. All fits use only
original training components. Do not read original validation/test examples or
predictions. These folds are already development data, not fresh blind outcomes.

Reuse0726's original-objective fitted models and selections by exact artifact
hash; do not rerun completed training solely to recreate those controls. For
the all-pair objective and each nonzero condition, perform nested selection on
its fitting rows. Predict both reference contrasts and all pairs in the omitted
training components using each objective's resulting same fitted coefficients.
Report a two-objective × four-representation × two-evaluation-space table, with
all prediction IDs/targets/weights, all model/scale/coefficient records, inner
selections, stationarity, rank and runtime. Zero requires no fit.

Primary evaluation is all-pair weighted OOF RMSE, Pearson, Spearman and prediction
spread, with component/parent/pair weights. Original-reference evaluation is
secondary and remains visibly distinct. Bootstrap55whole dependency components,
2,000PCG64draws, seed202610250726, sorted groups, preserving within-component
weights, linear percentiles. Do not bootstrap pairs as independent.

The reused original-objective coefficients must reproduce0726's original-reference
predictions within frozen numerical tolerances. An independent feature/SVD replay
must verify the new fits, both prediction-space transformations and every bootstrap
draw. Original fit verification is already closed in0726 and need not be duplicated.
Record finite-precision differences;
do not relax tolerances or modify original models to force agreement.

## Decision and stopping rule

Changing the objective receives priority only if the all-pair-trained149feature
model improves all-pair weighted OOF RMSE at least5%over its three all-pair-trained
controls *and* over the original-objective149feature model scored on the same
all-pair space, has predictionSD>=25%of targetSD, improves at least4of5folds over
the original-objective149feature model, and has a positive paired-bootstrap95%
lower bound for that matched-model gain. Preserve every condition even if a
different one looks more favorable. These are development materiality criteria,
not biological validation or a proof of causality.

If this gate fails, stop extending this native response-only cohort's feature or
optimizer search. Its reproducible response can remain a constraint on a later
joint model, but it has not qualified as a stand-alone sequence-generalizing
endpoint. The next scientific move must address native baseline product amounts
and their MS/precursor observation contract using the shared sources, with a
preregistered direct baseline before a matched exact/discrete generative fit.
Do not substitute conditional observed-set composition for full availability.

If the gate passes, still require a new independently confirmed biological cohort
before an exact/discrete comparison is called useful. Sampling or GRPO is not
expected to rescue a failed deterministic supervised representation; eventual
comparisons must match representation, objective, information and compute/score
accounting.

## Execution and closure

Register a timestamped family before diagnostics; retain the source/feature/plan
hashes and exact commands. LocalCPU/oneBLASthread, no installs,20min/stage,
3GiB RSS,1GiB diskreserve+32MiBstartup margin,50ms monitoring plusOSpeak. Capture
environment at stage start. Close all summaries, prediction dumps, plots,
independent checks and canonical registry. No original held-out or500nMmetrics
apply; explicitly explain the training-only development scope.

The full goal still requires precursor-relative concentrations/distributions in
APCs, undetected and short products, machinery-conditioned processing/clearance,
MS observation calibration and independent biological transfer. Neither a
successful response loss nor a numerical exact/RL comparison alone meets it.
