# Full-parent processing versus direct native-response regression

Prospective after the fixed 0314 baseline result and before physical-model code or
fitting. The full objective remains precursor-relative peptide availability in
APCs. This stage tests a transferable processing representation on real native
cell response measurements; a steady-state proxy is not completion of that goal.

## Hypothesis and fixed source contract

Direct sequence regressions overfit held-out parent components: validation loss
0.12919 / 0.15318 versus condition-only0.07349. Test whether enforcing whole-parent
fragment production/removal gives better transfer with fewer degrees of freedom.
Do not tune more variants of the same direct regression on this validation set.

Use the unchanged canonical v3 native-response recipe and component split. For
physical models require its `full_parent_model_supported` flag. There are1,855
training and245validation groups; only six training groups are omitted for missing
or unsupported full-parent reference support. Count exact/bounded observations and
all omitted source IDs before launch. Retain canonical weights on this subset and
renormalize once within each split. Do not recompute or select components by outcome.
Refit the same six frozen baselines on this identical subset/weight contract for
the final comparison. All631testgroups have full-parent support; reserve their
metrics until every model/setting is frozen and all required fits finish.

Keep all exact and bounded targets, candidate origin alternatives, form identities,
contributor IDs, source metadata and censor directions. No target clipping,
generated negatives, MHC length filter, chosen parent or per-peptide free offset.
Specimen, control-denominator, homology and external-study independence remain
unresolved. This is a same-study, seen-treatment, held-out-parent-component task.

## Minimal generative model and inputs

Inputs: complete candidate parent protein sequences; candidate peptide spans;
supplied condition identity (cell and literal chemical exposure, with dose/time
provenance retained); no MHC/TCR sequence or assay label as an input feature.
Every full protein can generate every contiguous span1..L, including short products
and nonbinders. Compute only the observed queries for the training loss; this is
an implicit all-span model, not a distribution normalized over observed positives.

Continuous unit precursor input, independent irreversible bond cleavage, common
fragment removal. Fix removal rate delta=1 to choose an unidentifiable time unit.
Control bond rates r_i=exp(rho + w·phi_i). Treatment multiplies every bond rate by
exp(a_condition). The condition offset b_condition represents a shared input or
observation-ratio shift; it cannot absorb per-peptide abundance differences.

For a span with internal cut-rate sum R, set A=delta+R. Its steady-state amount
per unit precursor input is:

- whole parent: F=1/A;
- one internal boundary of rate l: F=l/[A(A+l)];
- two boundaries l,r: F=lr(2A+l+r)/[A(A+l)(A+r)(A+l+r)].

Use the corresponding log formulas for numerical stability. The model is equivalent
to integrating survival times cut/no-cut probabilities over precursor age. It does
not require enumerating every span in memory. Across candidate origins, collapse
duplicate full reference sequences, give each distinct candidate sequence equal
unit input, and sum all exact occurrences within that parent before averaging
across parents. This explicit origin prior is not measured precursor abundance.

Prediction: b_condition + log10(F_treated/F_control). No within-parent normalization,
MHC selection or MS-detectability head is fitted here. Same-form detector cancellation
and unchanged relative precursor input except the shared condition offset are
assumptions. Modified forms keep distinct observations but share sequence-defined
processing amounts; report modified/unmodified strata and do not claim a PTM model.

Two physical representations:
1. Uniform bonds: w absent; train rho, a and b.
2. Adjacent sequence bonds: left/right residue indicators phi, centered using only
   supported training-parent bonds; remove linear null directions with a frozen
   training-only SVD, retain its mapping. Train rho, w, a and b together. No imported
   digestion fit or additional pretraining; initialize uniform bonds.

Condition indicators use a training-only centered orthogonal basis with a global
intercept for b and for a. Allocate no unseen condition or unsupported feature
coefficient. Record all training support and parameter changes. Initialization:
rho=0, w=0, a=0, b=0. At a=0 the treatment/control prediction is identically zero;
paired sampling must retain that equality for shared-rate derivatives.

## Matched differentiable and sampled arms

For both representations fit an exact-gradient arm and a conditional-score hybrid.
Same physical forward model, features, initial parameters, data, reductions,
regularizer and optimizer. The hybrid uses exact forward amounts, exact residuals,
exact origin probabilities and exact b gradients; it is explicitly not simulator-only
RL or a GRPO claim. It samples the kinetic score that gives the log-amount derivative.

For an observed span, sample its origin with probability proportional to its
contribution to F. Conditional precursor age has a simple event representation:
whole parent Exp(A); one boundary Exp(A+l)+Exp(A); two boundaries choose left-first
with probability (A+l)/(2A+l+r), then draw Exp(A+l+r)+Exp(A+r)+Exp(A), or swap l/r
for right-first. These are conditional event waiting times, not biological labels.
Internal bonds are uncut, boundaries are cut; outside bonds are analytically
integrated out. Their conditional mean score is zero, reducing variance without
changing the model. The sampled kinetic score uses log(exp(-delta*t)*P(span|t));
differentiate rates with sampled choices/times detached. Preserve per-query origin
mixture scores and use shared uniform random draws for treated/control pairs.

Use K=512 conditional samples per response group, seed42, float64, one CPU thread.
No clipping, group-standardization, sample reuse or KL penalty in this first native
comparison: earlier synthetic factorials remain references; changing the biological
objective here would confound representation with estimator effects.

Before fitting: verify stable formula against small-parent full-span integration,
probability/length-weighted mass accounting, conditional-age density/moments and
finite-difference gradients. At initialization and one predetermined small parameter
perturbation, 16 independent K512 groups must have mean relative gradient error<=.10
and RMS<=.25 (record absolute error for zero exact-gradient blocks separately).
Do not tune draws or thresholds after the check. If it fails, preserve failure and
run the admitted exact biological arm; no unqualified sampled fit.

## Fit, evaluation and resource contract

All arms:200full-batch Adam updates, lr.02, beta(.9,.999), eps1e-8, no dropout,
gradient clipping, schedule, restarts or validation selection. Exact/censored loss
matches0314; penalty .01 on supported coordinates except b's global intercept.
Declare the full coordinate penalty mapping before code execution. Final checkpoint
only, save training curves and all parameter changes. These are bounded model
comparisons, not a convergence claim; report stationarity and remaining train loss.

Local caps:300seconds per exact arm,1800seconds per sampled arm,4GiB per process.
The first20updates of each sampled arm are part of the registered run; retain them
and project runtime before continuing. If projection exceeds the cap, stop that
arm with its partial checkpoint and report the failure, without silently reducing
data, changing sample count or lengthening the cap.

Primary comparison: canonical-weight exact/censored validation and then test loss
against all matched baselines. Also exact-value RMSE/MAE/R2, pooled Pearson/Spearman,
prediction spread, nonzero response sign accuracy, bound satisfaction/violation,
condition/component/form/origin-ambiguity strata, parameter support, gradient error,
sample counts, score evaluations and actual runtime. Use one exact prediction per
distinct input to preserve analytic ties. Source records and all final per-observation
train/validation/test predictions are required. No500nMbinding metrics apply.

Promote no physical model that fails to beat condition-only and no-change on the
same primary held-out contract. Report exact versus sampled accuracy and cost even
if both fail. Freeze all choices before opening test metrics; no test-driven retry.

## Relation to the full goal

The steady-state conditional model is a deliberately falsifiable first processing
representation. Chemical labels are not measured machinery activity. Exposure
timing, changing precursor input, compartment exchange and state-dependent processing
can invalidate this proxy. Do not translate delta/rho to biological half-lives or
these ratios to concentrations. A passing result would motivate time-resolved
perturbation dynamics and typed APC machinery conditioning using the existing
source timing/handling fields. Actual APC validation, precursor support, independent
MS response, transport, pMHC selection and concentration calibration remain required
before the persistent objective can be complete.
