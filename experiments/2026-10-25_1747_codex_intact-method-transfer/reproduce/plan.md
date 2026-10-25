# Frozen organ transfer with method-supported unmodified search assignments

2026-10-25, Codex/GPT-6. This is a new comparison; family1655 stays closed at its
strict row-explicit gate. Source-only family1735 independently qualified the
original Methods declaration of no additional search modifications. No F3 model
scores were seen. Original source cells/quality metadata were already audited;
this is not an untouched dataset, independent-animal or independent-protein test.

## Sole cohort-policy change

Use the same independently verified PXD058121 original F3 source package and
metadata-covered44acquisitions. The shared Hitlist endpoint is separately named
`pxd058121_organ_method_baseline_change_v1`. It retains raw nulls and records the
source Methods assertion and its scope. An explicitly unmodified row qualifies;
a blank Unlocalized Mod field qualifies only with reported zero localized and
terminal shifts and the pinned study Methods declaration. This is support for the
search assignment, not biochemical proof of purity. The exact study software
version and export path remain unknown. No other source-quality gate changes.

Original organ views only; exclude recorded F2 acquisition/raw-file reuse, unknown
raw acquisition, ambiguous reported/replayed assignments, invalid spans, >10ppm,
conflicting input-ID/intensity links and all multiple eligible masses per
sequence/acquisition. Require a unique admitted intact normalizer in the same
acquisition. Do not impute missing/zero/negative quantities or infer censoring.

For each Brain/Liver/Spleen peptide curve with eligible nominal-zero measurements,
supply their mean log10(product/intact) ratio. Preserve every baseline and
normalizer ID/value/replicate label. Later targets are observed log ratios minus
that supplied baseline. Future support cannot create baseline eligibility. Primary
times30/60minutes, descriptive extrapolation90/120minutes. Require each fixed organ
to have>=20supported primary peptide sequences and>=40primary cells. If any fails,
stop all scoring and preserve the failure, without substituting organs or times.

## Unchanged frozen model comparison

Freeze the same31checkpoint/selection files and SHA256s as family1655 before any
new-cohort numeric admission. No fitting, organ-specific correction, selection,
hyperparameter change, extra training data or old F2 held-out target access.

The21fixed arms are:

1. Zero change.
2. Shared-slope, length-slope and sequence-slope ensembles from0143; equal mean of
   all5stored outer predictors, retaining saved transforms and inner-selected alpha.
   Time feature is elapsed minutes/80. No full-cohort refit.
3. Uniform, residue-penalty.01 and residue-penalty0 physical ensembles from0143;
   equal mean of the5saved training-objective-selected initializations.
4. The14original0046 final checkpoints: uniform/residue exact2 and conditional-score
   16/64draws × seeds42/43/44 × both parameterizations12. These are exact-forward,
   REINFORCE-gradient hybrids, not simulator-only RL or GRPO. Evaluate physical
   component(t)-component(0), cancelling the historical peptide offsets.

Inputs: peptide/parent sequences and span, time and supplied baseline. Organs are
evaluation strata, not learned inputs. No MHC/TCR/species classifier/processing
machinery/MS-response head is trained. The physical process covers all spans,
but this observed-positive endpoint does not validate missing peptide mass.

## Independent verification and frozen quality gate

Replay source eligibility, the method evidence, raw-null preservation, baseline and
normalizer joins, every disposition, target arithmetic, support and weights without
calling the shared compiler. Pin the full verified shared-code commit before
admission. Verify prediction reconstruction and zero-time cancellation before
reading new outcomes for scoring. Save every per-example prediction/outcome and
the model/source identity support flags; no prediction-based exclusions.

Primary weights: equal organs, peptide curves, observed times, replicate cells.
Report weightedMSE/RMSE/MAE/R², unweightedPearson/Spearman, prediction/targetSD,
organ/time strata and relative gains. No new validation split or model selection:
all models are fixed. Original0143/0046metrics retain their earlier contracts.
IC50/500nM binding metrics are inapplicable. Supply baseline dependence and one
animal/one substrate limitations in all interpretations.

The preselected sequence ensemble must improve primaryRMSE>=5% over both zero and
length, have positive pooled Pearson and Spearman, predictionSD>=.1targetSD and no
organRMSE>10% worse than length. Secondary results cannot rescue primary failure.
Use1000paired peptide-identity bootstrap draws, seed31415, resampling jointly across
organs; do not imply animal-level replication. Report all gates and fixed exact/
score-pair differences. Inference costs are new; prior training costs are not rerun.

If sequence passes, retain it only as a relative temporal comparator. If it fails,
close this evaluation without feature searches, threshold changes or refitting F3.
Either result leaves calibrated precursor-relative APC quantities, machinery
conditioning, short/unobserved products and external biological transfer unproven.

## Execution and closure

LocalCPU/float64, one numerical thread,3GiB free+speculative startup and RSS limits,
20minutes/stage,1GiB disk reserve+32MiB margin,50ms polling plus OS peak. No Modal,
shared-environment edits, raw-MS downloads or unrelated process changes. Freeze
each stage's source/requests before execution; retain all refusals/failures.
Shared source stays once under data with provenance snapshots. Close report,
summary tables, prediction dumps or exact support-failure reason, canonical log
and source/implementation issues. Full APC availability remains the active goal.
