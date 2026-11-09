# Prespecified conditional inhibitor-response fit

Frozen after the training-only signal diagnostic, before held-out response
statistics or model predictions. This is a small assay-response experiment,
not an intracellular concentration or cleavage-trajectory model.

## Question and data

Can four sequence features predict the repeatable peptide redistribution under
epoxomicin, conditional on an observed untreated captured-product distribution?
Does sampled score-function training recover the same objective as exact
enumeration at this scale? A failure to predict held-out responses must remain a
negative result even when training loss and optimization diagnostics improve.

Use the immutable `mapp_controlled_run_intensity_v1` shared Hitlist export from
`2026-10-23_0645_codex_controlled-mapp-signal`. Source and recipe manifest hashes
must be copied into the new prospective experiment before launch. No new curation
adapter or alternate treatment interpretation. Keep the existing sequence/parent
component split, alpha6 capture, nonflagged peptide PEP <=0.01, one source group,
three positive runs per arm, and at least one direct-MS/MS record in each arm.
Qualified support: train59 peptides/40 components; validation14/5; test14/9.
These are same-study acquisition observations, not independent specimens.

The complete-case restriction uses treatment-arm detectability and retains only
2.6% of training peptides with any qualifying alpha6 signal. This is a selected
observed support experiment; it cannot evaluate missing products, detection
probability, all possible spans, absolute intracellular yield, or prospective
discovery. Do not label excluded or unobserved peptides negative. No binding or
MHC-length filter. Time-course and capture-control acquisitions remain curated
but are not training labels in this comparison. No IC50, TCR or MHC heads here.

## Smallest conditional model

Within each fixed split support, form arithmetic mean measured intensity across
the three untreated runs, `u_i`, and independently across treated runs, `e_i`.
Let `U=sum(u)`, `E=sum(e)`, `q0=u/U`, `q1=e/E`. The untreated observations are
required model inputs even on validation/test. Treated observations are labels.
Never use held-out treated totals or composition as model inputs.

Each peptide has four features: natural log length, fraction AVILMFYW, fraction
KR, fraction DE. Standardize each feature using training peptides only, with
population standard deviation and a fixed floor1e-8. No sequence embedding,
flank selection, parent embedding, allele, species or inferred residual activity.
The context is one known inhibitor condition; it cannot generalize to other
machinery states or KO without new data.

`p_i(w) = softmax(log(q0_i) + x_i @ w)` has four fitted coefficients.
One separately fitted scalar `beta = log(E_train/U_train)` supplies the detected
signal scale. Prediction is `e_hat_i = U * exp(beta) * p_i(w)`. This factors
composition and measured total signal without calling either molar abundance.
Every parameter is trained: four composition coefficients plus one assay-scale
coefficient. The scale has a closed-form training-only fit; no unused modules.

## Matched objective and execution

Composition objective: `KL(p(w)||q1) + 0.1*||w||^2/2`, natural logarithms.
The scale objective is squared error of `beta` versus the training log total
ratio; its closed-form fit is shared by all variants and reported separately.
Do not conflate feature ridge regularization with a reference-policy KL penalty.

- Exact: enumerate every observed peptide in the training support, differentiate
  the full objective. Initialize w=0; Adam lr0.03, betas(0.9,0.999), eps1e-8,
  2000updates, float64, CPU, one thread. Retain the final checkpoint, no early
  stopping or validation-driven hyperparameter choice.
- Sampled references: same policy/objective/optimizer/initialization; sample
  K=32 or128 peptide identities from the current categorical policy per update.
  Use REINFORCE with leave-one-out mean cost baseline and detached
  `log(p_i)-log(q1_i)` costs. No group standardization, clipping, sample reuse or
  reference KL in this biological control comparison; the earlier synthetic
  factorial already isolated those changes. The cost's direct-gradient term has
  zero expected score; document and check the exact gradient identity.
  Seeds20261023/24/25 for each K; no seed selection.
- References: unchanged untreated composition and total (w=0,beta=0); common
  measured total shift (w=0, fitted beta); a separate ridge regression of
  `log(e_i/u_i)` on the same four standardized features plus unpenalized intercept,
  penalty0.1 in a mean-squared-error objective. Ridge has a different objective
  and is a predictive baseline, not a matched gradient comparison.

All proposed configurations are fixed now. Do not add features, change support,
retune penalties or lengthen training after viewing validation/test responses.
Before real fitting, verify the implementation's exact gradient against finite
differences and verify unbiased Monte Carlo mean gradients on a fixed tiny
categorical fixture with a recorded tolerance derived from Monte Carlo error.
This fixture is a mathematical check, not an extra biological outcome.

## Measurements and interpretation

Preserve every checkpoint, invocation, environment and per-example train,
validation and test prediction, including observed untreated inputs, treated
targets, component IDs and both intensity/composition predictions. Report
objective plus penalty separately, reverse/forward KL, total variation, log10
intensity and fold-change RMSE/Pearson/Spearman, and prediction spread. Explicitly
mark constant fold-change correlations undefined. IC50/binder metrics do not
apply. Bootstrap test parent components1000draws/seed20261023 for paired MSE
differences, conditional on the same small study and its acquisitions.

At updates0/100/500/2000 measure gradient bias/error against the exact gradient
using64 fresh groups, with a diagnostic RNG independent of training. Report
training versus diagnostic score evaluations and wall time separately, including
actual enumeration/normalization cost: sampling here still constructs the
categorical probabilities over the full observed support. Do not claim a
scalability win or a simulation of processing trajectories from this small fit.

Preferred predictive condition requires improved validation and test log10
fold-change RMSE over the common-shift baseline, with nontrivial finite spread.
Report any trade-off with distribution metrics. A positive result is still a
conditional assay-response result on87 selected peptides. A negative one moves
the biology effort to the known-substrate kinetics request Hitlist#770; do not
rescue it by test-directed tuning. The separately specified exact span-mass and
discrete cut-partition model needs known substrates/time, not cytokine exposure
time or these selected mixtures. Full availability goal remains active.
