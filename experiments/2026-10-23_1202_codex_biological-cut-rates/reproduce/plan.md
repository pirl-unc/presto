# First biological sequence-to-cut-rate comparison

Prospective model specification after training-only signal family1138. No
biological kinetic fit launched. Full APC concentration goal remains intact.

## Data and evaluation boundary
Use canonical digestion_contiguous_timecourses_v1 reviewed2.0.9 generation,
verified numerically identical to family1123's admitted examples. Freeze exact
source/recipe/code hashes at launch. Six train, three validation, two test
constructs; preserve all fixed splits/conditions/biological labels and original
selection. Training-only signal passes five/six constructs; do not drop LEDGF.
All source-positive trajectories and missingness remain conditional observations.
Upstream review has already inspected descriptive validation/test repeatability;
record this exposure. No model performance on either split exists yet. Treat the
first fit as bounded same-study transfer, with later independent-source validation
required. Do not tune after test results or resplit.

## Parsimonious ordered model
Input: literal full substrate sequence, physical digestion time, measured protocol
enzyme/substrate concentration context, and the required observed reference-time
MS signal for each peptide/biological label. No peptide ID embedding, parent-specific
learned parameter, MHC/TCR inputs or fitted nuisance head. All products are defined
by the parent process, including short nonbinders, irrespective of observation.

At each internal peptide bond, use the ordered residues immediately left and right.
Use20 canonical amino-acid categories with one reference category per position:
38 one-hot contrasts plus an intercept. Inspect training matrix rank and exposure
before optimization; if rank-deficient, freeze a deterministic SVD basis of the
training design and expose its exact mapping rather than allocating unsupported
coefficients. No training data outside these six constructs for pretraining.

log(k_b/hour) = intercept + ordered_sequence_features_b @ weights.
Begin with this context-independent effective-rate model; record enzyme ratios
as supplied covariates but do not silently impose proportional kinetics across
substrates. Add a separate declared context arm with one trainable log enzyme
concentration slope only if the training design has independent support; matched
Ffh conditions supply the only within-construct ratio contrast. Context is a model
arm, not a post-hoc correction to labels. No inferred APC components.

Each bond undergoes one independent irreversible cut. Cut probability at time t:
1-exp(-k_b*t). For a contiguous fragment [i,j), expected copies per initial parent:
product of required boundary-cut probabilities times exp(-t*sum(internal rates)).
Protein termini have boundary factor1. This is a deliberately restricted kinetic
hypothesis; it omits accessibility, sequential proteasome engagement, rebinding,
transport and clearance. Exact expectation conserves expected residue mass when
all spans are included. Do not normalize over observed products or enumerate2^(L-1)
partitions in the exact path: use boundary factors and internal-rate prefix sums.

## Observation and objective
Reference time per peptide/condition is the earliest common positive time across
admitted biological labels, as in1138. Supplied measured reference signal is needed
at inference. Predict signal(t)=signal(ref)*m(t)/m(ref), under a constant response
factor assumption. Train on log10 ratio error using stable analytic log(m), never
finite-sample log/count plug-ins silently substituted for it. Weight squared error
within curve, then equally within condition and parent, matching the diagnostic.
Original arbitrary-unit signal predictions and ratios both retained. No forced
zero target, missing-as-zero label, estimated absolute MS response or molar claim.

Controls: no temporal change, shared-time response fitted on training only, and
one effective cut rate shared across all bonds. Main exact model learns sequence
contrasts. Prespecify optimizer/regularization/budget/checkpoint rule before launch,
using only training mathematical and numerical probes. Every allocated coordinate
must be exposed, finite-gradient and updated; separately report weak identifiability,
rate profiles and Jacobian singular values. Gradient coverage is not biological
rate identification. Preserve all prediction dumps and per-parent metrics.

## Matched sampled qualification and fit
First validate analytic derivatives on small exact cut-state enumerations and
finite differences. At fixed initial and fitted-training states, measure full
partition score-estimator variance for groups32/128/512 against exact gradients,
count every evaluated bond/span/partition, and time actual local CPU work.

For log-ratio loss, a gradient using exact m(t),m(ref) and exact residuals with
sampled occurrence/score expectations is an unbiased hybrid gradient estimator:
derivative(log m)=E[1_span_present*score]/m. Name this exact-objective/denominator
hybrid explicitly. Independent conditional-event sampling or integrating irrelevant
bonds can reduce variance; label those additional analytic contributions explicitly.
A plug-in Monte Carlo log-ratio has finite-sample bias and zero-count issues; measure
both rather than calling it an unbiased replacement. No RL-only or score-saving
claim for hybrid variants using exact expectations.

Only launch an iterative sampled comparison after the gradient probe establishes
its defined objective and feasible variance/budget. Keep the exact fitted reference
fixed. If unconditional samples cannot resolve rare spans, record that failure;
use the conditional/Rao-Blackwellized hybrid as a distinct arm rather than changing
the sampling procedure under an old label. Synthetic pure sampled kinetics remain
a mathematical reference, not evidence of real biological transfer.

## Decision and full scope
Report train/validation/test RMSE, Pearson, Spearman, spread, per-construct losses,
rate/support coverage and work/runtime for all fixed arms. Require improvement over
controls in both validation and test and nonconstant predictions before promotion;
otherwise preserve the negative result and replan from training diagnostics.
No nM/binder thresholds for this endpoint. This bounded purified-enzyme fit cannot
complete APC availability: species/cell machinery effects, precursor context,
transport/clearance and independent non-MHC MS response calibration still require
additional shared data and interventions. Continue upstream requests770/775/757/762.


## Frozen execution contract for this family

Three exact fitted variants: constant1 coefficient; sequence39 coefficients;
sequence_context40 coefficients when the train design has full rank. Reference
amino acid A at each side, amino-acid order ACDEFGHIKLMNPQRSTVWY. Context is
log((substrate_um/enzyme_substrate_denominator)/0.2uM), a supplied protocol-derived
concentration proxy; a learned slope, never assumed linear activity. Train-support
rank gate is evaluated before fitting. For rank deficiency preserve a deterministic
train-only SVD subspace; all fits still report original-space coefficients and
held-out feature residuals. All source labels and original splits stay unchanged.

Adam lr0.03, betas0.9/0.999, eps1e-8,2,000 full-batch updates, float64, one CPU
thread. Initialize intercept log0.01/hr and every other weight0. L2 penalty
0.01/2 times squared nonintercept weights (including context); constant model
unpenalized. Use final checkpoint; no validation-driven selection. Preserve
checkpoints0/100/500/1000/2000 and every100-step training loss/gradient statistics.
No dropout, pretraining, clipping, sample reuse or reference-policy KL. Fit once
per deterministic condition, no seed search. Shared-time control: weighted least
squares on [log(t/ref), log(t/ref)^2], no intercept, no regularization. No-change0.
Weight time observations per curve, curves per condition, conditions per parent,
parents equally in each split; only common positive times, earliest as reference.

Before fitting: tiny six-residue exhaustive cut-state/score-gradient verification,
finite differences, conservation, positive-rate and train-feature rank/coverage.
Hard numerical failure stops the fit with its checkpoint; no automatic retries.
Report all train/val/test ratios and original-unit signal predictions, RMSE,
Pearson/Spearman/spread plus per-parent and objective metrics. Biological threshold
metrics are inapplicable. Final held-out evaluation occurs once per fixed arm.

After exact fits: training-only complete-partition hybrid gradient probes for
sequence_context if supported (otherwise sequence), at initial and final states.
32 independent gradient estimates per K32/128/512, generator seed20261023+index;
LOO reward baseline, exact residuals and denominators. Report mean error, RMS
error, standard errors, full vector dumps, zero-occurrence/rare-product diagnostics,
network/partition/span operations and walltime. Also report plug-in log-ratio
zero-count failure rates without replacing zeros by arbitrary pseudocounts. No
iterative sampled fit in this family until a separate budget/schedule is frozen
from these training-only runtime/variance results. Conditional/Rao-Blackwellized
analytic comparison is explicitly hybrid. This staged qualification does not
claim an RL-trained biological model or a score-computation advantage.
