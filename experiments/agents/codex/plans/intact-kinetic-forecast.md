# Measured kinetic forecast: direct controls, exact fragmentation and sampled scores

Codex / GPT-6, October 25, 2026. Prospective model specification after canonical
source family0021; no fitting or predictive outcomes have been inspected.

## Question and scope

Does a minimal sequence-conditioned fragmentation model predict subsequent
measured product/substrate changes better than persistence and direct temporal
regression? Does a conditional trajectory score estimator recover the exact
gradient/result, and at what variance/runtime cost?

This is a narrow falsification test of a processing module: purified mouse-liver
20S acting on one human alpha-synuclein substrate. It is not an APC concentration,
new-protein, unseen-specimen, MHC-binding or TCR benchmark. Fragmentation rates
are effective model parameters, not certified biochemical constants. Independent
bond cleavage is a deliberately simple hypothesis; the assay does not establish
that it models proteasomal engagement/processivity correctly.

## Shared admission before fitting

Implement a separate registered Hitlist kinetic recipe using the frozen source
package, with all source assignment dispositions and metadata preserved. The
source-family descriptive census found1,278candidate cells/237sequences, an intact
substrate measurement in all21kinetic acquisitions, and80sequences with at least
two early times and at least one late time. These are support counts, not training
results. Do not weaken ambiguity/quantity gates to increase support.

Use deposited F2 once, not reused F1 or publisher-selected S4. Require positive
finite intensity, one reported and replayed mass assignment, a valid source span,
absolute mass error<=10ppm, explicit unmodified source fields, and a matched
input-mass intensity. Omit all ambiguous multiple-mass sequence/acquisition groups;
never choose by intensity or sum unqualified masses. Preserve source confidence
limitations and missingness. Exclude intact parent from the predicted product
targets while retaining its measured denominator and exact source lineage.

Target is `y = log10(product intensity / intact substrate intensity)` within the
same acquisition. Both quantities must be admitted. This can cancel a common
multiplicative run factor; unknown product/substrate MS response remains a
peptide-specific nuisance. It does not convert intensity into molarity.

Use times0/20/40/60/80minutes for fitting,100for validation,120for a once-opened
test. Eligibility uses at least two distinct fitting times, never presence in a
late outcome. Missing later products remain missing; evaluation is conditional
on observed positives. Keep all three reported replicate labels; do not call
them independent biological replicates. Require at least20eligible product
sequences and40observations in each held-out set before predicting/scoring.
If support fails, report failure without changing the split or threshold.

Weights give equal mass to peptides, then times within peptide, then observed
replicate cells within peptide/time. Each split has its own normalized weights.
Train-time per-peptide nuisance estimates remain fixed for both held-outs.

## Exact generative model and representation

Input is the full140-aa substrate, an interval `(i,j)`, and exposure time `t`.
There are139candidate peptide bonds. A bond has a positive constant hazard
`lambda_k`, initially sharing one rate, or sharing a rate by the residue directly
before the bond. Use only residue categories present in the substrate; no dormant
encoder or unsupported feature coordinates. Species and enzyme preparation are
supplied fixed context, not predicted or learned in this single-condition fit.

Every bond independently breaks by exposure `u=t+tau` with probability
`1-exp(-lambda_k*u)`. A complete sampled cut set yields all contiguous fragments,
including1–3mers, long fragments and intact substrate. No4–20mer restriction or
binding screen removes material. Positive `tau` permits products at nominal zero;
it is an effective initial-processing assumption, not a measured preincubation.

The exact expected count of interval `[i,j]` per original substrate molecule is:

```
N_ij(u) = exp(-u * sum(lambda_k, k=i..j-1))
          * (1-exp(-lambda_(i-1)*u), if i>1, else 1)
          * (1-exp(-lambda_j*u), if j<L, else 1)
```

Enumerate all `L*(L+1)/2=9,870` spans when reporting full distributions; use
prefix sums for internal hazards and direct indexed intervals for fitting.
This integrates all2^139possible cut configurations without listing histories.
It must conserve residue mass: `sum(length_ij*N_ij)=L`. Expected fragment count
must equal `1+sum(1-exp(-lambda_k*u))`. Products can be born and destroyed by
further cleavage even though their ratio to intact parent may rise monotonically.
That monotonicity is a falsifiable restriction, not a guaranteed property of data.

The predicted assay value is
`mu_s(t)=a_s+log10(N_s(t+tau)/N_parent(t+tau))`.
Profile `a_s` as the weighted mean training residual for that peptide, recomputing
it differentiably during fitting and freezing it at the chosen checkpoint.
It absorbs unknown fixed MS response and cannot identify that response separately
from other peptide-specific scale factors. New peptides with no fitting anchor
cannot be predicted under this endpoint; broader sequence/MS constraints remain
necessary for the full APC availability objective.

Parameterize each rate between1e-7and1/min andtau between1e-3and120minutes with
smooth bounded transforms. Initialize every rate at.001/min andtau at10minutes.
Compute log probabilities with stable `expm1` expressions, not clipped outputs.
Report any near-boundary estimates and lack of curvature/identifiability.

## Controls and frozen optimization

Mandatory controls: per-peptide training mean; most recent training-time mean;
shared linear time slope with per-peptide intercepts; and per-peptide linear
slopes shrunk toward the shared slope. The latter uses ridge strengths.001/.01/.1/1,
selected only on100minute validation. Time is scaled by80minutes; record exact
loss scaling and closed-form/intercept treatment in the numerical contract.

Physical models: uniform hazard+tau; residue-conditioned hazards+tau. Minimize
the same weighted squared error in product/substrate log10ratio, profiling all
peptide nuisance intercepts. Residue model adds.01 times mean squared centered
log-rate to limit unsupported between-residue spread; uniform model has no such
term. Float64 CPU, one numerical thread, Adam lr.03,800updates, fixed final
checkpoint, no restarts or test-driven changes. Every allocated rate/tau parameter
must receive gradients and change; report rank/curvature separately from gradient
activity. Boundary solutions do not count as physically identified rates.

## Discrete comparison and honest algorithm labels

Use the same cut process, parameterization, inputs, weights, nuisance profiling
and loss for exact autograd and conditional score gradients. For each queried
fragment event, force its bounding cuts and unbroken internal bonds, then sample
unconstrained outside bonds. This samples complete cut configurations conditional
on the queried fragment. `E[grad log P(cut set) | fragment]` is the gradient of
the event log probability. Parent survival fixes all bonds unbroken.

Keep exact forward event probabilities and loss residuals. Replace only the
event-log-probability gradient with the conditional Monte Carlo score estimate;
use16and64draws, seeds42/43/44. This is **hybrid conditional REINFORCE**, not
simulator-only RL, GRPO, or an estimated log-likelihood from rare raw counts.
Independent-bond structure makes the conditional forced-bond contribution exact;
outside-bond scores add zero-mean noise. Therefore sampling is expected to cost
variance here. Do not advertise a biological RL advantage from matching the exact
reference. Clipping, group standardization, reuse and reference KL are absent so
the objective stays matched; their prior synthetic studies remain separate.

Also sample unconditional complete cut sets at fixed predeclared parameter
points. Check exact span probabilities, residue-mass conservation, event zero-hit
rates and Monte Carlo errors versus sample budget. This diagnoses why naive
log-frequency fitting can fail on rare fragments; it is not another biological
fit or a replacement evaluation target. If a future processive/state-dependent
model loses exact tractability, this reference anchors the next discrete method.

## Verification, evaluation and decision

Before fitting: independent NumPy span formulas; short-substrate enumeration of
all cut sets; full-substrate mass conservation; finite-difference gradients;
conditional sampler event inclusion and score-mean uncertainty; independent
source/recipe/target replay through public queries. Freeze source, invocation,
environment, exact numerical contract and seeds in a new timestamped experiment.
Refuse to fit if the source/support/numerical gates fail.

Save every model/checkpoint/optimizer state, training curves, parameter/gradient
coverage, and covariate-only held-out predictions before opening corresponding
outcomes. Validation selects the direct comparator and preferred exact physical
variant. Compare sampled conditions descriptively under the same model contract;
do not select a sampled seed using test. Test metrics: weighted MSE/RMSE/MAE/R²,
Pearson/Spearman, prediction/target SD, per-peptide/time/replicate strata and all
per-example predictions. No IC50/500nM metrics apply. Bootstrap peptides, clearly
not biological specimens, for descriptive paired RMSE differences.

Primary gate: preferred physical model improves RMSE>=5% against the validation-
selected direct comparator on both validation and test, and has positive held-out
Pearson/Spearman and prediction SD>=.1target SD. Report all failures without
retuning inspected sets. Sampling comparison reports gradient error, estimator
variance, objective recovery, actual runtime and sampled events, including exact
forward work. Even a pass demonstrates only within-source temporal prediction
of measured ratios, not absolute availability, APC transfer or physical rate truth.

Resource contract for the next family: local CPU/one thread,3GiBRSS,
20minutes/stage,128MiBnew physical artifacts,1GiBfree-space floor plus32MiBmargin.
Register before any model fit. If full checkpoints exceed the cap, revise storage
prospectively before launching; never drop failed attempts or weaken a guard.
