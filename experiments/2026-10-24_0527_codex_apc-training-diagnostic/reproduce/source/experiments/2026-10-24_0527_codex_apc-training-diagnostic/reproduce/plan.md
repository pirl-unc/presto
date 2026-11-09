# APC training diagnostic: precursor calibration, penalty and condition-specific rates

Previous goal turn is progress: 0449 completed nine fixed biological models,
independent closure and publication34568ca. It failed: condition-only testRMSE
.120011 versus processing+precursor.139278; withoutprecursor.120008 with almost
constant within-condition predictions. Preserve that failure and its inspected test.
The full APC availability/concentration goal remains unchanged and unachieved.

## Question and endpoint discipline

Separate three explanations before seeking a new independent population:
1. Does forcing proportional transfer of the digest-derived precursor signal hurt?
2. Does the .01 kinetic penalty prevent fitting sequence-dependent response?
3. Does scaling all control cleavage preferences together prevent condition-specific
   response even when a useful sequence signal exists?

Use ONLY original0449training observations:492peptides,984condition groups,
361canonical dependency components,2,952source channel responses. No new access to
originalvalidation/test outcomes or predictions. Hash/filter the existing canonical
cohort by its original split before feature fitting. Preserve all parent/precursor
alternatives, quantities, contributors and hierarchical weights; no new source curator.
No affinity, machinery, TCR, detectability or absolute-concentration labels are added.
Same MHC-bound response endpoint and study limitations remain. No synthetic labels.

This is a posthoc training/development diagnostic, not a new untouched biological
test. Fivefold component-grouped inner CV within originaltraining uses fixed
sha256("pxd057588_train_diagnostic_v1:"+component_id) modulo5. Do not reseed or
rebalance from outcomes. Refit all feature scaling/projections and precursor
calibration inside each training fold. Retain original row weights, renormalized
within each fitting/evaluation subset. Also fit each fixed arm on all originaltrain
for optimization/identifiability diagnosis. No checkpoint selection or outer-test use.

## Controls and precursor calibration

Four controls: zero response; independent condition means; condition means plus
the unscaled precursor log-ratio (gamma1); condition means plus a learned single
nonnegative gamma. Here z is the existing equal-candidate-parent mean linear fold
ratio expressed in log10. Fit gamma=max(0,sum w z_centered y_centered /
sum w z_centered²), with condition-wise weighted centering. No upper bound or
outcome-selected penalty. Fit two condition offsets analytically and unpenalized.
Save unconstrained gamma, nonnegative gamma, variance/covariance and offsets.

For every physical arm, freeze that fitting-subset gamma and use parent input
10^(gamma * parent_log10_ratio). The calibration control and full origin mixture
are not algebraically identical with multiple different parents; preserve both.
Gamma is an empirical transfer/calibration coefficient, not a measured biological
elasticity. It cannot resolve normalization, measurement noise or selection effects.

## Fixed processing factorial

Use the qualified independent-cut continuous-input/common-removal model, delta1,
all1..Lfragments implicit, complete parent sequences and all query occurrences.
Control log-rate =rho+phi·w. Two treatment options:
- shared preference: log-rate_control + x_condition·a (0449kinetic structure);
- condition-specific preference: log-rate_control + x_condition·a +
  phi·(x_condition·V), where V has2×rank coordinates.

Phi uses the same40adjacent-residue indicators, fitting-subset equal-parent bond
weighting, centering/scaling and SVDrelative1e-10. Condition x is the fitting-subset
global/contrast basis. Allocate only supported directions and fail rather than
invent a held-out direction. Shared rates allocate41kinetic parameters, flexible
rates117when rank38; two offsets are profiled exactly at every objective call.
Each arm additionally uses the one fixed estimated gamma. No peptide-specific offset.

Factorial: shared/flexible treatment preferences × kinetic penalty .01/0.
All arms zero-initialize kinetic coordinates, use400full-batch float64Adamsteps,
lr.02,betas.9/.999,eps1e-8,seed42. Penalty applies to all kinetic coordinates;
observation offsets are unpenalized. No pretraining, warmstart, clipping, schedule,
restart or selected epoch. Report this different nuisance-fit contract explicitly;
do not call it a controlled repeat of the old joint-offset fit.

Primary objective weighted MSE in log10response plus the registered penalty.
For each update, profile the two condition offsets as weighted residual means.
Differentiate through the profiled objective. Predict inner held-out rows with
training-fitted offsets only; never profile offsets against evaluation targets.

## Numerical and biological decisions

Before fitting, check3finite-difference directions for shared/flexible zero and
fixed NumPy703normal*.03states on the originaltraining cohort: absdifference<1e-7.
Save parameter-block gradient norms and ranks/singular values of the condition-
centered weighted prediction Jacobian at initialization and each full-data final fit.
These diagnose local identifiability, not causal interpretation or global convergence.
Decompose training MSE into within-response-group channel variation and error of
group means; a group-mean oracle is a descriptive label-derived lower bound only.

First complete all4×6exact physical fits and all4controls×6fit sets. Preserve
all final checkpoints/trajectories/transforms and source-linked inner-CV predictions.
Inspect all predetermined conditions together, with no winner retuning.

Run a matched conditional-score hybrid ONLY for the predesignated flexible/zero-
penalty arm if it improves pooled inner-CV RMSE by at least2% over calibrated-
precursor control and has within-condition predicted SD>1e-3 in both conditions.
This gate prioritizes biological signal over more sampler work; preserve a failed
gate and do not substitute another arm. If eligible, qualify its zero/fixedrandom
full-training states with16independent draw groups,K512,meanrelative<=.10 and
RMS<=.25 full unregularized gradient error. Use paired treated/control uniforms,
precursor-weighted exact origin probabilities, exact forward/residuals and profiled
offsets; sampled conditional event-score kinetic derivatives. No GRPO clipping,
standardization,reuse orKL. If qualified, run all5folds+fulltrain with identical
400updates,seed42,and compare originaltraining/CV losses,variation,time and scores.
This is a hybrid, not simulator-only RL. No success claim from passing gradients.

LocalCPU only,one requested numerical thread,no shared-environment changes.
Per exact fit300s,per hybrid1800s,4GiB. Project hybrid total from its first20real
steps and stop if over cap. No outcome-driven retry. Preserve failed/capped fits.
Do not claim regularization/structure is causal merely from one optimizer endpoint.

## Required closure and next step

Register timestamped family before arithmetic; freeze originalsource/filteredinput
hashes,code,primitives,invocations,versions and all nondefaultenv. Reuse qualified
source admission; do not repeat broad workbook/reference audits. Independently
reconstruct calibration/profiled offsets, scalar physical predictions and metric
reductions from recorded final coefficients/source rows. Save all per-row predictions,
train/inner-CV RMSE,MAE,R2,Pearson,Spearman,sign,spread,condition/component strata,
loss decomposition,parameters/gradients/ranks,sample counts/runtime and plots.
No originalheld-out test predictions by design: this is diagnostic work within the
originaltraining cohort; nM/binder metrics do not apply.

UpdateREADME,canonicalexperiment_log andtasks. No production baseline promotion
from this diagnostic. If calibration or flexibility fits training but worsens CV,
record overfit and stop expanding this source's features. If exact and sampled
agree again, preserve exact as practical method. The next predictive experiment
needs an independent prospective population or independently constrained precursor/
machinery/selection/observation information. Full concentration goal stays active.
