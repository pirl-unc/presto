# Transfer frozen cleavage preferences to cellular captured-peptide composition

## Hypothesis and launch dependency

The verified sequence-context factorial improves purified-digestion development
shape error, primarily through wider cleavage context, not extra terminal MS
features. Test whether those learned rates carry transferable information to the
existing cellular MAPP endpoint where direct training failed to beat uniform.
Close and publish family2026-10-23_2358 before executing this plan. Do not change
its fits or choose a sampled seed based on its outcomes.

This is a different-study transfer diagnostic with historically exposed MAPP
partitions, not a new blind cellular validation or a concentration claim. A
separate native intracellular source admission is owned by Hitlist788; those
unqualified workbooks must not silently enter this experiment.

## Canonical inputs and exposure audit

Use MAPP min2 recipe manifest
df78d02f8200809c659e43adfae29866c48b712122b740379106994622f4ac80,
PXD009092 human HEK293 proteasome-associated fragments. Preserve original parent/
component splits, peptide occurrences, source quantities, qualifiers, source
reference and specimen limitations. Training72parents/144conditions and existing
validation15parents/30conditions; exclude14historical test parents from target
construction, fitting and evaluation. Prior test results remain published but are
not revisited. No new consumer curation, controls-as-labels, missingness imputation
or binding-length filtering. Short/nonbinding product support stays present.

Before target arithmetic, verify manifest/file hashes and perform a sequence-only
exposure audit: exact whole-parent identity, containment of a complete digestion
construct within a MAPP parent (and reverse containment), shared measured peptide
sequences, source component identity and new-feature support. Retain overlap flags
and report strata; do not invent homology/specimen independence or silently remove
rows. If overlap makes an evaluation stratum too small, report that limitation.
Original source/run graphs remain the grouping authority.

Import only verified base and rate_context kinetic coefficients/feature transforms
from2358. Freeze their hashes. Their MS-response coefficients are not transferred:
MAPP has a different capture/measurement domain. Preserve the original adjacent
and wider-context rate models; all purified-digestion parameters remain frozen.

## Minimal cellular bridge: input and removal

A cell is not a known-time purified substrate pulse. Use a deliberately minimal
continuous precursor-input plus common fragment-removal assumption. The same
independent-cut expectations C_s(t) used for digestion define residence-weighted
amount per unit precursor influx:

    F_s(delta) = integral_0^infinity C_s(t) exp(-delta*t) dt.

Delta>0 is one global effective residence parameter, shared across both MAPP
conditions. Imported enzyme context is fixed to its documented reference0; the
relative rate/removal scale is an assumption, not measured intracellular time or
residual proteasome activity. No condition-specific parameter or inferred KO is
added. Predicting treatment change is intentionally outside this first transfer
comparison; the model gives zero condition change for identical parent sequences.

For span[a,b), let S be the sum of internal bond hazards and u=S+delta. With two
required boundary hazards l,r:

    F = l*r*(2*u+l+r)/(u*(u+l)*(u+r)*(u+l+r)).

With one boundary k: F=k/(u*(u+k)); with no boundary: F=1/u. Compute positive
log forms, never subtract nearly equal reciprocals or floor values. This covers
all spans1..L; no observed-set mass normalization is part of physical F. With
unit full-parent influx, sum_s length(s)*F_s must equal L/delta. Absolute cell
amount would additionally require influx/precursor abundance, removal calibration
and compartment volume. The current data do not supply those anchors.

For each MAPP candidate, sum F over original allowed occurrences (never duplicate
its label), then center log10F within its parent/condition. Compare with the
unchanged original centered log10intensity. This observation bridge assumes equal
unknown peptide-specific capture/MS response within the observed set; it is an
explicit uncalibrated assumption. Failure cannot uniquely falsify cleavage rates.
Do not call the measured pool free cytosol or infer total unobserved product yield.

## Fixed comparisons and training

Retain exact uniform observed composition as a zero-parameter baseline. Fit three
one-scalar models: uniform cut rates (all.01 per original hour), frozen adjacent
rates, frozen wider-context rates. Initialize delta=.01, optimize log(delta) with
2,000full-batch Adam updates, lr.03,betas.9/.999,eps1e-8; no penalty, clipping,
bounds, schedule, pretraining on MAPP, checkpoint selection or repeated starts.
Loss is equal-parent/condition mean centered-log10 squared error, with equal
candidate weight inside each example. This aligns training to the measurement
endpoint; earlier KL fits are contextual different-objective controls. Final
checkpoint only. Preserve divergent/nonfinite or boundary-like solutions rather
than rescuing them with floors or outcome-driven restarts.

Run one prospectively named sampled hybrid for frozen wider-context rates with
the identical scalar objective, initialization and2,000updates, K512,seed42. Exact
forward F, residuals and conditional derivatives remain; this is not a simulator-
only RL model or a claim of score savings. Use conditional complete partitions
with a sampled age and an explicit target-mixture importance correction. Derive
and verify the estimator before launching; do not add GRPO modifications here.

Conditional age for a chosen span can be sampled as sums of independent waiting
times. Two-boundary spans: first wait Exp(u+l+r), then Exp(u+r) for left-first or
Exp(u+l) for right-first, then Exp(u). Left-first mixture weight is
(u+l)/(2*u+l+r). One boundary: Exp(u+k)+Exp(u); whole parent: Exp(u).
At sampled age T, force required boundaries cut and internal bonds uncut; other
bonds follow their original independent cut probabilities. For prior age
T~Exp(delta), target-span probability is delta*F. Mixture density q/p sums
w_j*I_j/(delta*F_j) over admitted spans. Preserve complete partitions, including
unobserved/short products, for score checks.

Use the Rao-Blackwellized derivative d_s=dlogF_s/dlogdelta=-delta*E[T|s] in the
importance-weighted kinetic gradient. Subtract a shared within-example derivative
baseline from d_s, adding its exact coefficient-sum term back; this keeps the
objective unchanged and avoids avoidable cancellation variance. Use the existing
90%gradient-norm/10%uniform target proposal with this centered derivative. Record
explicitly that exact conditional derivatives are retained, as in the prior hybrid.

## Qualification, resources and decision

Before biological updates: compare all spans in small parents with an independent
branching-generator resolvent, numerical integration and finite differences at
fixed delta values1e-4/.01/1/100; verify the residue identity. Check conditional
age means, full-partition support and importance expectation on enumerated small
systems. Check actual initial and predetermined logdelta±1 states without
validation labels:16K512 groups, mean relative gradient error<=.2 and RMS<=.5,
finite values. If a reference gradient is effectively zero, report that condition
and use a prospectively specified absolute-error criterion before executing
probes; do not invent a denominator after results. A failed sampler gate blocks
only the sampled fit; preserve exact biological comparisons and the failure.

Freeze the exact qualification implementation, numerical tolerances and absolute
near-zero criterion in the experiment registration before running it. Bound
qualification and fit runtime from a target-independent cost preflight; local
CPU/one thread/4GiB, no Modal/environment changes. Register actual caps before any
fit and preserve failures without extensions.

Save all train/validation predictions and full optimizer trajectories; report
MSE/RMSE, Pearson/Spearman, KL as secondary descriptive metric, spread, per-parent
and overlap strata, learned delta and runtime/coverage. No test metrics because
its parents are explicitly excluded. Independently reconstruct source targets,
physical amounts, feature transforms, all final metrics and optimizer arithmetic,
plus fixed exact/sampled gradients and RNG draws. One scalar training success
cannot qualify the full architecture.

Proceed toward independent cellular datasets only if the frozen sequence prior
improves meaningful biological controls; otherwise reconsider the observation
bridge or source support. Preserve production/removal/MS/precursor uncertainty.
Shared admission of Hitlist788, direct cellular quantities, machinery contrasts,
transport and downstream pMHC selection remain required for the full goal.
