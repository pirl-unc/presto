# Intact kinetics: optimization and monotonicity explain little of the training miss

Codex / GPT-6, October 25, 2026. Training-only diagnostic registered before new
calculations at Presto `3ab625edce9ceb6aed9d94ee104306dcf84e248c`, with recorded
task/report dirt and experiment-local code. [Frozen plan](reproduce/plan.md),
[commands, environment, input hashes and git state](reproduce/launch.json).

**Neither registered explanation passes its materiality threshold.** Reoptimizing
the unchanged fragmentation model improves training MSE only **2.29–2.32%**
(required 5%). The extra descriptive error forced by increasing, concave curves
is about **7.98–7.99%** of original physical-model training MSE (required 10%).
Both stop rules remain unchanged. These are diagnostic decisions, not proof that
optimization or curve shape never matters.

The larger remaining gap is between a shared-rate model and separately fitted
curves. This motivates investigating rate sharing, feature support and regularization;
it does not establish a unique biological cause. The separately fitted curves have
far more flexibility and are not generalizing predictors. **No new held-out score
or APC concentration result is claimed.**

## Data and read scope

Exactly the 0046 canonical PXD058121 training set: 626 positive product/intact
substrate log10 intensity ratios, 96 alpha-synuclein peptides, times 0–80 minutes,
purified mouse-liver 20S. Original equal peptide/time/replicate weights, source
ambiguity/modification/normalizer policy, and all labels are unchanged. One source
animal, unresolved replicate independence, nominal-zero products and unknown MS
response limit interpretation. No new source curation or synthetic biological labels.

Only the previous family's training covariates/outcomes, exact-model checkpoint
and summary files, and production model code are allowlisted. Paths and SHA256
values are in launch.json. Its `model.data()` loader is never called. The copied
model module supplies mathematical functions; a new explicit loader reads only
training rows. No validation/test covariates, predictions or outcomes are read.
No held-out prediction dumps exist because the purpose is diagnosing already-fitted
training data; the 100/120-minute outcomes were inspected in 0046 and remain closed
to this diagnostic. All new per-example **training** predictions are retained.

Only the intensity-ratio endpoint is fitted. No affinity, stability, MHC, TCR,
species, APC knockout, transport or MS-response head trains. There is no pretraining,
absolute concentration calibration, imputation, missing-negative inference or
censoring. IC50 and 500 nM binding metrics are inapplicable.

## Hypothesis 1: forced curve shape

For independent constant bond hazards and exposure `u = t + tau`, log product/parent
ratio is an outside-interior hazard sum times `u`, plus boundary terms
`log(1 - exp(-rate*u))`. Its first derivative is positive and second derivative is
nonpositive. A fixed peptide offset changes neither. Actual production derivatives
were checked against analytic derivatives at original initial/final and fixed mixed
parameters for both models, not just against a separate reference implementation.

For every peptide, original-weight replicate means are fitted by three descriptive
classes: free timepoint values, nondecreasing values, and nondecreasing concave
values. At most five times permit complete active-set enumeration of the weighted
convex quadratic problem. Every fitted value, multiplier and KKT residual is saved.
Independent weighted PAVA and SLSQP agree with the actual shape solvers; constant,
increasing, decreasing and peak fixtures qualify them before observed-data fitting.

| Training curve class | Original-weight MSE |
|---|---:|
| Free timepoint means | 0.009991 |
| Nondecreasing | 0.012096 |
| Nondecreasing and concave | 0.013600 |
| Original shared uniform fragmentation | 0.045230 |
| Original residue-conditioned fragmentation | 0.045186 |

There are **47 negative changes among 208 adjacent replicate-mean comparisons**
and **16 negative unpenalized slopes among 96 peptides**. These descriptive counts
do not distinguish biological loss from measurement variation. Free-curve residuals
describe within-time replicate dispersion, not pure measurement noise. Shape adds
0.003609 MSE above the free fit, below the predeclared 10%-of-physical-MSE threshold.
All independent curve predictions agree within 6.67e-16; maximum KKT residual is
9.72e-17. The shared physical model's much larger error is not explained simply by
forbidding decreases.

## Hypothesis 2: incomplete optimization

Six L-BFGS-B conditions reuse the exact production objective: weighted MSE with
training-only profiled peptide offsets, plus 0.01 mean squared centered log rates
for the residue model. Uniform has two coordinates; residue has 18. Each starts
from the archived Adam endpoint, original initializer, or all rates1e-5/min and
tau60min. Physical bounds remain rates1e-7–1/min and tau0.001–120min; numerical
raw-coordinate bounds[-30,30] are an explicitly small restriction of the original
sigmoid domain. Maxiter1500, maxfun3000, maxls40, gtol1e-9, ftol1e-13. No adaptive
restart, new seeds or revised loss. These are exact-gradient optimizer diagnostics;
no additional RL variant is run or claimed.

| Condition | Training MSE | Gain vs original Adam MSE | Iterations | Function evaluations | Core seconds |
|---|---:|---:|---:|---:|---:|
| Uniform, Adam start | 0.044194775 | 2.2893% | 25 | 27 | 0.052 |
| Uniform, original start | 0.044194775 | 2.2893% | 31 | 33 | 0.063 |
| Uniform, low-rate start | 0.044194775 | 2.2893% | 26 | 31 | 0.057 |
| Residue, Adam start | 0.044136468 | 2.3219% | 41 | 48 | 0.088 |
| Residue, original start | 0.044136468 | 2.3219% | 48 | 54 | 0.099 |
| Residue, low-rate start | 0.044136469 | 2.3219% | 42 | 53 | 0.097 |

All solvers report convergence and all allocated coordinates change. Every
coordinate has nonzero accumulated gradients. All runs move effective pre-exposure
to approximately its **120-minute upper bound**; none reaches the raw coordinate
bound. This is a model-fitting diagnostic, not evidence that the material actually
experienced 120 minutes of unrecorded processing. Three starts agree closely,
without proving a global optimum. No optimizer result is promoted as a new
biological baseline; the 5% materiality stop rule rejects further optimizer tuning
on this cohort as the immediate priority.

## Independent verification and artifacts

Independent NumPy reconstruction verifies all 3,756 optimized training predictions
and all 246 recorded function evaluations. Maximum prediction error3.55e-15;
objective error1.71e-13. Independently evaluated final finite differences agree with
the production gradients to6.22e-11 (tolerance1e-6). Curve means, per-example shape
predictions, KKT optimality, aggregate MSE and decision thresholds are replayed.
L-BFGS's internal line-search/update implementation is not independently replayed.

[Shape predictions](results/shape-predictions.parquet),
[per-curve fits and certificates](results/shape-curves.json),
[summary](results/summary.json), and [independent replay](results/independent-replay.json)
are retained. Each condition's results directory contains training predictions,
all objective evaluations/gradients, accepted iterates, final parameters, offsets
and convergence/resource details. Executed source snapshots are under
`reproduce/source/<stage>/`; original training/checkpoint inputs remain in the
explicit prior-family paths, with immutable hashes rather than duplicate datasets.

Diagnosis takes2.04s/344,981,504B peak polled RSS; independent replay0.87s/139,100,160B.
Local macOS arm64 float64 CPU, one numerical thread; no cloud/GPU or installation.
NumPy2.5.4, SciPy1.18.1, Torch2.14.1. Frozen overrides set OMP/OPENBLAS/MKL threads
to1. Caps remain3GiBRSS,20minutes/stage,32MiBnew logical artifacts,1GiBfree disk plus
32MiBstartup margin. Run `sh reproduce/launch.sh diagnose`, then `replay`, then
`closure` in a fresh output directory; existing stage receipts prevent overwrite.
No guards or numerical tolerances were changed.

## Consequence for the availability goal

The preceding exact/conditional-score biological failure remains intact. This
diagnostic reduces support for “just train longer” and quantifies, rather than
assumes, the effect of monotonicity. Next compare parsimonious representations of
peptide-specific temporal changes using training-only development and a separately
registered fresh biological evaluation. Distinguish feature/regularization/coupling
effects before adding processing states. An initial-mixture or MS-response model
must be identifiable from actual supporting measurements; it cannot be introduced
solely to improve an inspected test. The full precursor-relative APC availability
and concentration goal remains active and unachieved.
