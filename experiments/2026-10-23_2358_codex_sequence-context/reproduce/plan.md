# Sequence context versus observation response: nested biological comparison

## Question and decision

The completed profiled-scale comparison leaves weak development prediction
(RMSE1.26272, Spearman.24265), and the fixed-model residual diagnostic locates
roughly77% of analytical training MSE in persistent curve means. Test whether
ordered sequence context omitted by the current linear models accounts for any
transferable signal. This tests predictive representation, not a causal separation
of processing and MS response or absolute concentration identification.

Run a 2x2 factorial with existing analytical independent-cut kinetics: adjacent
residues versus a six-residue bond window; existing instrument-specific peptide
length/composition response versus adding the first/last peptide residues. Keep
fragment-length dependence off. Retain all four outcomes, then run one separately
gated conditional sampled-gradient hybrid for the full context/full response
arm, selected prospectively rather than from development outcomes. No additional
feature, seed, penalty or optimization sweep is authorized by this plan.

## Fixed biological contract

Use unchanged canonical Hitlist digestion manifest
5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615,
MSV000092631/PXD044463 purified human standard20S. Same90,630training observations
from six constructs/seven conditions,14,822development observations/three parents,
positive corrected log10MS signals, source contributors/forms/instruments,
parent/condition/curve/time weights. No new source curator, filtering, negatives,
censor conversion or synthetic biological targets. Original test parents stay
excluded; already inspected development is not a new blind test. Source parsing
may include excluded rows but they never enter feature fitting or loss.

Inputs: full parent sequence, span, time, enzyme context and instrument. Outputs:
expected fragment copies per precursor plus relative MS response and one exact
profiled condition scale. All lengths1..L remain possible, including nonbinders.
Development scales use development measurements for the existing relative-shape
endpoint. No new APC species/KO/transport/clearance, pMHC or TCR labels train.
Independent MS calibration, precursor abundance and cellular validation remain
requirements of the full goal; fitting this endpoint does not meet them.

## Nested representation and support

Preserve the old40kinetic and38response coordinates exactly, including the old
ratio-trained40coefficient initializer. For a bond between positionsi andi+1,
append amino-acid contrasts at i-2,i-1,i+2,i+3; each has19non-A residue indicators
and an explicit out-of-parent boundary indicator. Adjacent positions already
exist. Total80raw extra rate features. No peptide identity or construct lookup.

For observation response append instrument-specific indicators for first and
last peptide residue (19non-A categories each). No residue outside the measured
peptide enters response; these are shared sequence effects, not per-peptide
fitted offsets. Existing instrument/length/composition features remain.

Use training sequence/metadata only to residualize extra rate features against
old rate columns under equal-parent/condition/bond weights. For response,
residualize within-condition-centered extra features against the existing38
features with the original hierarchical weights. Normalize residual columns by
training RMS, retain the eigenbasis with eigenvalue>max(largest,1)*1e-10, fix signs.
Drop unsupported/dependent directions before allocating parameters. Preserve
transform matrices, names, eigenvalues and per-condition held-out residuals
outside the training span. Held-out unsupported directions are projected away
and explicitly reported; no broad serving-support claim. No labels define bases.
Zero extra coefficients must reproduce the old model exactly in every group.

## Training and comparison

Four exact arms: base, rate_context, response_termini, both. Each starts at the
same old rate initializer, zero old/new response and zero extra rate effects.
2,000full-batch Adam updates, lr.03, betas.9/.999, eps1e-8; no dropout, clipping,
weight decay or schedule. Objective: original weighted condition-centered MSE
plus .005sum squared coefficients excluding the kinetic intercept. Each added
supported coordinate has the same penalty. Final checkpoint only. No development
selection, repeated starts or outcome-driven extension. No new pretraining.

The fifth arm uses the both representation and the same objective/initialization/
schedule; replace only kinetic data gradients by the established conditional
complete-partition importance estimator. K512, seed42, fresh samples each update,
90%gradient-norm proposal plus10%uniform, exact probabilities/residuals/conditional
scores and exact response gradients retained. This is a sampled hybrid, not a
fully simulator-only RL model or a compute-saving claim. No GRPO changes here.

Before fitting, check old-objective equality and zero-extension equality,
independent NumPy full values/Jacobians at initialization and a predetermined
.02*sin(coordinate+1) perturbation for each arm. Check toy full enumeration of
conditional mixture gradients. For the hybrid,16K512 gradient groups at both
fixed states must have mean relative error<=.2 and RMS relative error<=.5, all
finite. Preserve any failure; do not increase samples or choose another proposal.
Qualification failure blocks that arm, not the four exact fits if their gates pass.

## Evidence, limits and resources

Local CPU float64, one thread, shared environment unchanged. Cap each exact fit
at600s and hybrid at1800s, peak4GiB; zero-update cost preflight must support the
cap before launching. Preserve a cap failure, no rescue checkpoint. Log progress
every100updates, all gradients/pre/post optimizer states, fixed RNG states at
steps1/100/1000/2000, material gradient/update counts and final checkpoints.

Save all train/development predictions with source identities/targets/weights,
physical amounts,response,scales. Report weighted MSE/RMSE, pooled Pearson/
Spearman/spread, temporal-ratio error, persistent/temporal decomposition and
per-development-parent results. Separate training gains from development gains.
Compare with constant1.28060, old exact1.26510 and old hybrid1.26279; no nM/binder
metrics apply. Do not promote a model from a one-seed, three-parent development
change or force target-like spread. Meaningful next progress requires improvement
that survives independent proteins and eventually external study validation.

Independent final verification rebuilds raw features from sequences, checks
source identities/weights, NumPy analytical physical counts and all final metrics;
replays all Adam arithmetic, selected exact gradients and selected hybrid RNG/
gradients. Preserve exact execution/source/data hashes and failed attempts. All
results belong in one registered experiment family, with report, plots, archives,
reproduction commands and canonical log. Full availability goal stays active.
