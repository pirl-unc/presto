# Conditional nominal-dose response diagnostic

Prepared 2026-10-25 by Codex / GPT-6 after source family0959 passed independent
replay and Hitlist2.0.40 was published from tested main commit
`d76c225a0aa741a1a1b0aee707bff328ca467c7c`. This is a prospective plan; no fit has
run under it. Register a timestamped experiment and freeze this plan, actual
recipe, code and invocation before numerical preparation or evaluation.

## Question and endpoint

Does nominal dose predict the positive aligned MS signal of the intended heavy
standards beyond a constant response for each standard/form/charge/DIA method?
This is an observation-model diagnostic conditional on the deposited library,
positive signal and exact scored-feature linkage. It is not an unconditional
detection model, general sequence encoder, or intracellular concentration model.

The source contains ten intended standards, nine library-supported forms and
90 aligned standard/acquisition slots. Earlier audits found 87 exact feature
links and three missing IDs (one positive signal, two zeros). Recompute the
dispositions through the released public API; do not assume that all 87 links
are eligible without verifying positivity, units and unique joins. The absent
tenth standard remains an explicit panel-membership absence, never ten zeros.

The nominal solution grid is supplied metadata, not certified injected amount.
Retain the twofold-versus-1:10 source conflict and all unknown purity, volume,
recovery and preparation relationships. DIA1 and DIA2 are different acquisition
methods, not exchangeable replicates. The library uses this study's outcomes;
even leave-level-out fits remain conditional development diagnostics.

## Shared data contract and admission

Use the canonical package from family0959 with manifest SHA256
`e81fc00bbe8b952fb37ddd63c0c369b31936bc78f1cbb109c9354254bbf5f486`.
Open it using the clean Hitlist2.0.40 checkout and public EvidenceStore APIs.
Retain raw inputs once under data/; do not parse original workbooks in a new
Presto adapter or duplicate the large package per fit.

Use the existing schema1 declarative materialization API if it can express the
policy. Publicly query the complete standard membership and acquisition/nominal
metadata; freeze these small source-backed sidecars and query specifications.
Request all aligned slots for the intended library members through the feature
link relation with aligned/scored joins. Preserve every candidate and all recipe
decision/reason records, contributors and source coordinates. Retain metadata
for the absent panel member separately because it has no aligned candidate row.

Accept only unique target-standard/form/charge/acquisition joins with exact
scored-feature identity and positive finite aligned quantitative_value in
source_signal_arbitrary_units. Keep literal source intensity, parsed value,
feature IDs, scores, original units/status, nominal level and method. No extra
q-value threshold is inferred. Missing, zero, negative, nonfinite or unmatched
values are not fitted; their distinct dispositions remain in the output.
No LOD, censor bound, pseudocount, missing-value imputation or native background
negative is invented. Generic recipe ambiguous/excluded labels retain their
actual semantics; only accepted records enter the fit.

Require five distinct positive nominal levels with one acquisition per method
at each level, consistent nominal units/status, unique molecular membership,
and source-backed acquisition links. Preserve name-derived link precision. Fold
groups unite both methods at a nominal level; the recipe's acquisition grouping
alone does not establish the final split. Any ambiguity or unsupported essential
metadata stops admission. If the existing public API is insufficient, record the
specific gap on Hitlist#845 and implement it there before fitting.

## Inputs, representations, outputs and four fixed models

For each accepted row define
`x = log10(nominal_concentration / largest_supplied_nominal_concentration)` and
`y = log10(aligned_signal_in_source_arbitrary_units)`. Keep original quantities
beside these explicit derived values. The highest supplied nominal level is
known design metadata, not an intensity-derived normalization.

Let g identify standard, exact molecular form, charge and DIA method. The only
inputs are this nuisance group and x. There is no sequence representation or
pretraining, and no species/APC/TCR/affinity/processing parameter is allocated.

| Arm | Prediction | Fitted parameters |
| --- | --- | --- |
| constant | a_g | one intercept per supported group |
| unit slope | a_g + x | one intercept per group; slope fixed at one |
| shared slope | a_g + b*x | group intercepts and one shared slope |
| method slope | a_g + b_method*x | group intercepts and two method slopes |

Fit weighted least squares without regularization or hyperparameter search.
Each group has equal total training weight; within a group rows share that
weight equally. Minimize the mean of group mean squared log-signal residuals.
All allocated coefficients must be identifiable and solved. Require full design
rank and at least two distinct training levels in every represented group.
Never substitute zeros or a regularizer for a missing group's intercept.

## Fixed development evaluation

Five leave-one-nominal-level-out folds hold both acquisitions at the level
together. Fit all four arms in every fold: 20 fits. Four full-data fits provide
descriptive coefficients only, for 24 total fits. No random row split or changing
folds after results. No external test is claimed: prior source audits inspected
this small selected panel, and there is no independent preparation/biological
replication or unseen-standard representation in these models. Original native
outer partitions remain closed.

Write every OOF prediction with raw/derived target, candidate/source identity,
group, method, nominal level, fold and training-ID digest. Record all fold/full
coefficients, training membership, matrix rank and residual diagnostics.

Report pooled and equal-group RMSE/MAE/MSE, Pearson/Spearman, prediction/target
standard deviations and per-level, per-method and per-standard errors. Report
within-group-centered OOF metrics separately so peptide offsets cannot conceal
flat dose response. Center only for descriptive metric calculation after OOF
predictions exist; evaluation targets must never affect a fitted coefficient.
Undefined constant-vector correlations remain null with a reason. Binding
threshold metrics and log10(nM) error are inapplicable to MS signal.

Primary comparison: shared slope versus constant on identical accepted rows.
Prespecified diagnostic gates are: at least 10% reduction in equal-group OOF
RMSE; positive shared slope in all five training folds; within-group-centered
OOF prediction SD at least 25% of centered target SD; and no worsening of
equal-group RMSE within either method. Unit slope tests proportionality; method
slopes test acquisition dependence. Report all arms even if gates fail.
Do not turn the winning arm into a general observer. No bootstrap significance
claim is warranted from nine selected standards and shared acquisitions.

## Verification, resources and decision

Before fitting, test the actual fitting path on known slopes and unequal group
sizes, held-level exclusion, positivity/missingness and rank failures. Independently
reconcile admitted metadata and targets to source-backed query records. Replay
all fitted coefficients and OOF predictions with scalar grouped means and
within-group covariance formulas, separate from the weighted matrix solver.
Recompute every metric and gate from archived predictions. Treat numerical ties
explicitly; do not change tolerances or round biology to force rank agreement.

Freeze launch code/configuration and use local CPU with one BLAS thread. Each
numerical stage retains the existing 20-minute, 3-GiB process-tree RSS and 1-GiB
disk-reserve limits plus 32-MiB startup margin, 50-ms monitoring and OS peak check.
Launch only after 3 GiB free-plus-speculative memory is available; inactive RAM
is excluded. No Modal, shared environment changes or repeated source rebuild.
Preserve failures and prospectively amend before any changed contract is run.

Close the timestamped family with summaries, all predictions, independent
verification, runtime receipts and a canonical experiment_log.md entry. Passing
supports dose sensitivity only under this library/positive-linkage selection.
Failure stops expansion on this panel. Either outcome leaves protocol transfer,
native/APC calibration, precursor-relative yield/lifetime, short/unobserved
products and the next matched exact-versus-discrete biological comparison open.
