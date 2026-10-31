# HAP1 shared TAP-route response: exact versus conditional sampled gradients

Codex / GPT-6, 2026-10-24. Prospective specification before new fits or subgroup
outcome arithmetic. Previous goal turn made progress: 2041 was independently
verified and published at59a8db6; Hitlist2.0.28 release bytes were verified.

## Question and connection to the complete availability goal

Does a shared peptide-dependent reliance on a TAP-mediated route explain both
TAP1 and TAP2 knockout responses better than direct regressions with the same
inputs? This tests one pre-MHC delivery component. It does not identify the
baseline intracellular pool, full processing trajectories or concentrations.
The complete goal remains unchanged. Failure rejects this bounded response
parameterization; numerical agreement alone is not success.

The primary HAP1 paper (PMID40113210 / PMC12090245, Results and Fig4) reports
different TAP1/TAP2 effects, N-terminal source bias and allele-dependent changes.
TAP also supports the loading complex. Thus the fitted quantities below are
effective conditional response parameters, not experimentally measured TAP
activity, transport specificity, signal-peptide annotation or causal routing.
No pathogen sequences, interventions or optimization are part of this experiment.

## Data contract

Unchanged canonical source88cad10d... and recipea1eec392... at
data/hap1_response_recipes/2026-10-24_0730_v1. Load only original outer-training,
complete-precursor rows, then supplied condition in {hap1_tap1_ko,hap1_tap2_ko}.
This is a prospective mechanism-defined subgroup, motivated by primary biology
and the disclosed2041condition findings, not a new independent cohort.
Keep every eligible positive response, replicate, candidate parent and occurrence.
No imputation, score filter, missing-as-negative conversion or new assay labels.
Target remains log10 positive KO intensity minus mean log10 positive WT intensity.
Precursor is the same condition-level MaxLFQ surrogate; no aliquot/molar claim.

Reuse original component SHA256 fivefold assignments. Require >=300 peptides
and >=100 components per condition overall, and >=20 components per evaluation
fold/condition before fitting. Recompute equal condition/component/peptide/replicate
training weights inside each fit; use full-subgroup weights for pooled CV.
Outer validation/test remain closed. This is reused-development evaluation.
Other assay families and IC50/binder thresholds do not apply.

## Inputs and mathematical model

For each peptide/source-parent key, retain the original61length/composition/P1/C
features, add20P2 and20P3 one-hot features, and the four qualified2018geometry
features averaged over all parents/occurrences: log1p(start), log1p(C distance),
N-terminal fraction and C-terminal fraction. Total105 availability features.
No MHC scores or allele identities enter this route function. Queries in this
experiment have length8..14; shorter products remain unsupported extrapolation.
Standardize using fitting rows and weights only; omit zero-variance coordinates.

Let rho(p)=sigmoid(theta0+z(p)theta) be the modeled fraction of baseline-delivered
peptide relying on the affected route. Let epsilon_c=sigmoid(eta_c) be an effective
remaining contribution in the supplied KO condition. There are two conditional
paths with weights w_bypass=1-rho and w_TAP=rho*epsilon_c. Their sum
R_c=1-rho+rho*epsilon_c is the KO/WT relative delivery amount in (0,1].
The complement denotes delivery not retained, not experimentally proven degradation.
This is conditional on baseline delivery; it is not a source-protein-normalized
concentration and is not normalized across the observed ligand list.

Predict y=gamma*precursor + offset_c + log10(R_c) + score_z*beta_c.
Gamma is the nonnegative calibrated-precursor fit on this fit's training rows,
shared across all arms. Offsets are profiled weighted residual means by condition.
The optional three frozen log10 MHCflurry scores enter only the observation layer,
with separate coefficients per condition. They are empirical selection covariates,
not measured Kd; mixed affinity/MS/teacher training and incomplete lineage from2041
remain explicit. Hash-pin all existing scores/geometry; no new neural inference.

Rho and epsilon are not separately identified as causal quantities by these ratios.
Report the training Jacobian spectrum/rank, parameter changes, delivery-fraction
spread and saturation. Do not interpret finite regularized parameters as measured
rates. Unknown observation scaling is absorbed by offsets and score covariates.

## Fixed comparison and training

Five arms, each full training plus five folds (30 fits):
1. sequence_scores: original64-column ridge, refitted on the identical TAP subgroup.
2. expanded_ridge:105 availability +3 score columns, condition-specific ridge.
3. route_only_exact: shared route, no MHC-score observation coefficients.
4. route_scores_exact: shared route plus observation scores; primary model.
5. route_scores_sampled: identical model/objective/initialization/Adam, sampled gradient.

Ridge uses the original fixed0.1condition-normalized penalty. Route models use
weighted MSE +0.01*(sum(theta_all^2)+mean((eta-logit(.05))^2))
+0.1*mean_c(sum(beta_c^2)); theta_all includes its intercept. Initialize theta/beta
to zero and epsilon=.05. Offsets unpenalized, gamma frozen as above. Float64 CPU,
one Torch/BLAS thread;250full-batch Adamsteps,lr.03,seed42; no schedule, clipping,
sample reuse, reference KL, group standardization, early selection or hyperparameter
search. Retain steps0/50/100/150/200/250 and final predictions. Record last50-step
objective change and gradient norm; do not claim convergence if still improving
>1% of max(abs(objective200),1e-8). No automatic extension after failure.

Sampled arm uses exact forward log(R) and64fresh conditional route samples per
training row/update. Sample Z~Bernoulli(w_TAP/R), then use the mean gradient of
log(w_Z) as the unbiased score estimator for grad log(R). Detached posterior/counts,
analytic observation/penalty gradients. This is an exact-forward conditional-score
hybrid, not full simulator-only REINFORCE, GRPO or a new cleavage-trajectory test.
The comparison isolates gradient estimation under an identical response objective.
At final checkpoints evaluate exact forward values for both arms. One stochastic
seed is a bounded diagnostic, not a variance/optimizer superiority claim.

## Gates, independent checks, resources and closure

Before fitting: exact two-path sum versus scalar calculation <=1e-12; autograd
versus analytic gradients <=1e-10 over logits[-8,-3,0,3,8] and KO logits[-6,-3,0,3];
finite differences <=1e-6. Verify epsilon=1 gives logR=0 and monotonic R in epsilon.
For64samples x256replicates, mean score-gradient error must be <=5analytic standard
errors+1e-10 per coordinate. Also verify the objective offset projection and ridge
normal equations independently before launching the family. Gates stop on failure.

Primary biological development gate: >=2% pooled CV RMSE gain versus sequence_scores,
>=1% versus expanded_ridge, and improvement versus expanded_ridge in>=4/5folds.
Report both KO strata, all five folds, weighted loss/RMSE/MAE/R2/SD, Pearson/Spearman,
frozen-predictor exposure strata and model runtime. No outer-test promotion.
Sampled/exact agreement: relative CV RMSE difference<=2%, descriptive, not a win.
Preserve all results even if these gates fail. Stop this family rather than tune.

Independently rebuild weights/features/transforms, source joins and ridge solutions;
reconstruct all route predictions with NumPy scalar formulas, profiled offsets,
penalty/objective and final full-training analytic gradients (1e-9). Replay archived
metrics; qualify exact-input ties separately rather than altering raw ranks.
No large new data copies: source-linked per-example training/CV predictions,
latent outputs, checkpoints, histories and selected-source cohort are retained.
Pin source/model hashes, environment, Git/dirty state, launch commands and executed
source; report failures, full logs, metrics/plot, canonical registry and decisions.
Fit cap1800seconds/4GiB; numerical probes120seconds/2GiB. No cloud, installations,
shared-environment changes or biological claims from untrained model branches.
