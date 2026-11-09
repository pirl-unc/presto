# Frozen MHC-score control for HAP1 response

Codex / GPT-6, October 24, 2026. Plan precedes score generation and new fits.
Previous goal turn made progress: 2018 parent-context ablation failed its fixed gate;
source 2.0.27 published, recipe 819 merged, release 40129209653 still running.

## Scientific question and full-goal connection

Do three frozen allele-specific MHC prediction scores explain or improve the HAP1
sequence-response signal, and does protein geometry add information after them?
This controls a missing selection-related feature family before another mechanistic
exact/sampled comparison. It neither replaces the APC concentration goal nor claims
that scores from an affinity-named predictor are isolated biochemical affinity.
MHCflurry's 2.3.0 ensemble trained with measured affinities, reassigned MS ligands,
synthetic negatives and teacher pretraining. Use as empirical derived covariates;
never as measured labels, physical energy, occupancy or calibrated concentrations.
Static affinity alone cancels in a same-peptide KO/WT ratio. Here separate supplied
condition coefficients capture condition-dependent associations; they are not
measured changes in peptide affinity or causal loading/processing rates.

## Data and frozen predictor qualification

Use unchanged HAP1 recipe a1eec392c06485966e131663f85e7599dcd9e5ff621841eb0d44eeaafe779309
at data/hap1_response_recipes/2026-10-24_0730_v1; source88cad10d… from PXD056426.
Parquet-filter split=train, precursor_status=complete: 87,260 observations,
6,317 peptides, 1,798 molecular components, ten known conditions. No new source
curator, eligibility filter, outcome averaging, imputation, negatives or bounds.
Target remains positive replicate log10 KO/WT elution response. Preserve all prior
WT/B2M/precursor/mapping/weight policies. Other assays and 500 nM metrics inapplicable.

Pin installed clean MHCflurry source cd141c8c3c4eb393ab7a53fd4aa40a981cf5d10f and observed
package version 2.3.14; no environment upgrade. Use explicit installed 2.3.0
presentation-bundle affinity_predictor path, with all eleven selected networks.
Read only the affinity submodel; hash its manifest, weights, allele sequences,
training table, export metadata and imported source. No processing/presentation
combiner or percentile input. The published 2.3.0 weights use 2023 affinity data;
new software/model date does not imply new biological observations.

Before inference, inventory available fit/pretraining and selection provenance.
Audit exact peptide and allele-peptide overlap with bundled affinity training rows,
measurement kinds/sources, and any HAP1/Shapiro/PMID40113210/PXD056426 markers.
Record source/sample/teacher lineage that is absent; do not infer disjointness from
missing fields or dates. This is a development feature comparison, so prior exposure
to peptide identities alone does not exclude rows or change the fixed cohort. If
actual HAP1 response/source training is identified, stop before fitting. Missing
full lineage explicitly prevents independent/generalization claims.

Generate positive finite raw nM API scores for A*02:01, B*40:01, C*03:04 for the
6,317 training peptides only, then log10-transform. All three scores remain inputs;
no best-allele assignment or binding-based selection. Fail unsupported sequences,
alleles or nonfinite outputs rather than impute/exclude. Scores and ensemble-member
outputs are derived predictions and not the paper's original MixMHCpred results.
Verify API aggregate against independent geometric mean of all member nM predictions
at 1e-10 log10; verify deterministic reversed/rebatched 64-peptide probe at 1e-5
log10 (float32 network arithmetic). CPU backend, one Torch/BLAS thread, batch1024,
no network compilation or parameter updates. Inference cap300s/6GiB. Pin all input
and model hashes before and after scoring; stop if source/weights changed.

## Fixed fitting comparison

Reuse original five component SHA256 folds, weighting, nonnegative fitting-only
precursor gamma, condition offsets and global weighted standardization / within-
condition centering. Fixed condition-normalized MSE +0.1 sum(beta²); closed-form
condition-specific ridge. Same frozen 2018 regression core, verified by hashes and
original reference prediction replay. Six arms × full+five folds =36 fits:

1. calibrated_precursor (no sequence features).
2. sequence (original length/composition/P1/C,61 columns).
3. length_affinity (length + three log10 nM scores,4 columns).
4. sequence_affinity (64 columns), primary.
5. sequence_shuffled_affinity (64 columns): preserve the complete score triplet,
   use the 2018 outcome-independent within-length cyclic donor permutation within
   each fitting/evaluation partition separately; no cross-fold donors.
6. sequence_affinity_geometry (68 columns): append the four qualified source-parent
   geometry values from 2018 with all candidate/occurrence averaging unchanged.

Primary development gate: >=2% pooled CV RMSE improvement over sequence, >=1% over
shuffled scores, improvement over sequence in >=4/5 folds. Report standalone score
and geometry increments descriptively; do not select a new test winner. No extra
penalty/features/seeds after failure. Exact fixed full/fold fits, no neural finetuning,
pretraining, RL, GRPO, optimizer schedules or auxiliary Presto heads. 300s/3GiB fit cap.

Outer validation/test were inspected in 1503 and remain closed: no new scoring or
predictions there. Reused inner folds are development only. Preserve all per-example
full/inner-CV predictions and exact source joins. No external, unseen-condition or
concentration validation. Strong performance would motivate a separate selection /
processing model and a fresh biological population, not validate mechanism alone.

## Verification and closure

Before new arms, replay original control/sequence full/CV predictions within1e-10.
After fitting, independently reconstruct weights, input features and transformations,
SVD solutions, all per-example predictions (1e-10), archived-value metrics (1e-10),
continuous replay metrics and exact-input-tie rank qualification if needed. Retain
all fit/source errors, never silently loosen a gate. Preserve RMSE/loss/R2/MAE/SD,
Pearson/Spearman, condition/fold and fixed exposure-stratum metrics, support and
coefficient counts, timing/memory, scores/member outputs, model/teacher/source limits,
launch/source/environment and dependency receipts, comparison plot, report and
canonical experiment-log entry. Raw logs stay local unless fully inspected before
publication; publish only reviewed scientific output or retain explicit raw paths.
