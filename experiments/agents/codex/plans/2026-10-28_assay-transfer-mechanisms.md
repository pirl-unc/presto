# Explain asymmetric MHC-I / MHC-II auxiliary benefit

Owner: Codex / GPT-6. Baseline b5628eff. User authorizes continuing experiments
and asks for exact data composition, admission losses and mechanisms. Use the
same common architecture/trainer and at most two explicit H100! workers.
Family: `2026-10-28_0250_codex_assay-transfer-mechanisms`.

## Questions and limits

Previous six-output data was 82.1% class I versus 66.4% in exact IC50 training.
At matched updates, IC50 exposure fell roughly sixfold. Wider capacity lacked a
wider IC50 control. Endpoint, allele and censoring distributions also differ.
Inventory source/endpoint/class/species/groove/length/qualifier mix, unique peptides,
studies, pMHC contexts and allele concentration. Count observations separately from
unique peptides and optimizer exposures. No arbitrary row cap.

## Evaluation population

Use only the prior family's TRAIN observations. Old validation/test rows and
predictions must not enter new fits or selection. Rebuild connected components
across peptide, original measured-outcome identity, group IDs and group aliases.
Hash each component with fixed salt `2026-10-28-assay-transfer-mechanisms-v1` into
80% train, 10% validation, 10% test. Pin assignments/count support before fitting;
do not search split seeds for accuracy. No pretrained weights. These rows were
previously used in other models' training: newly held-out development comparison
for from-scratch models, not untouched external biology. Studies/homology may overlap.

All arms use identical IC50 exact/bound rows in each partition, physical units,
target policies and train-only scales. Auxiliary endpoints stay distinct. Sparse
class-II outputs remain unqualified. Off-rate excluded.

## Fixed comparison: twelve fits

Two seeds42/43, 6,000updates each:

| Arm | Capacity | Labels/sampler | Batch |
|---|---|---|---:|
| small-ic50 |128/2/4/512|IC50 exact+bounds, natural|64|
| wide-ic50 |192/3/6/768|same IC50 rows, natural|64|
| wide-six-natural |192/3/6/768|six outputs, natural|64|
| wide-six-class-balanced |192/3/6/768|six outputs, existing equal-class sampler|64|
| wide-six-primary-matched |192/3/6/768|50% IC50, remaining50% natural auxiliary proportions|128|
| wide-six-primary-shuffled |192/3/6/768|same inputs/sampler; auxiliary targets permuted within endpoint/class/groove/qualifier|128|

The last two preserve expected IC50 label counts per update, not identical gradients:
batch size, total examples and loss normalization differ from the64-example control.
Report actual endpoint/class exposures. Their paired real-versus-shuffled contrast
holds inputs, batch, compute, target marginals and schedule fixed, testing auxiliary
label information conditional on groove/qualifier. Permute only new TRAIN rows with
fixed seed73028; preserve original values and donor identities. Report unchanged
singleton/constant strata and actual changed-label fraction. This is an explicitly
synthetic negative control, never a promotable model.

Put IC50 first in endpoint configuration; verify initial common-core/IC50-head
hashes match between same-width/seed conditions. No class heads or new branches.
AdamW3e-4, decay1e-4, dropout.1; mean standardized one-sided squared assay loss,
endpoint weights1, no identity/pretraining/schedule/AMP/TF32. Validation every500/
final, checkpoint100, no early stop. Select lowest pooled exact IC50 validationRMSE,
then test selected checkpoint once. Plot validation by updates and IC50 exposures.

## Minimal common-core change

Extend metadata sampler with optional `stratum_weights`, only with `balance_by`.
Records contain `values` in field order and positive finite relative `weight`;
name all observed training strata exactly once. No targets/heldouts determine
sampling. Weighted epoch quotas cover each stratum once, cycle/shuffle within
strata, then shuffle the weighted epoch. Preserve original RNG/order byte-for-byte
when weights are absent. Persist order/RNG through existing resume and sampler
implementation fingerprints. Validate quotas before constructing an impossible
order. Document sampling versus loss weights and repeated-label exposures.
No second trainer or experiment-only optimizer.

Test invalid/unknown/missing strata, default compatibility, quotas/cycling,
determinism and exact interrupted/resumed weighted training. Run focused trainer/
resume tests and package checks for this substantive sampler change.

## Interpretation and promotion

Contrasts: wide/small IC50 (capacity); wide-six-natural/wide-IC50 (auxiliaries plus
dilution); class-balanced/natural (class exposure); primary-matched/natural (target
exposure with batch/compute change); real/shuffled primary-matched (auxiliary label
information beyond allele means). Secondary within-groove rank, calibration,
censoring, class/output counts, parameter coverage and rare/unseen-groove/length
strata. Report failure to reproduce the original asymmetry without adding arms.

Seal validation-only decision before opening tests. A real-label six-output arm
must gain>=.01 macro class rank vs wide-ic50, lose no class>.005 rank or>2%RMSE,
improve macro rank in both seeds and each class in some seed. Ties within.005 favor
lower compute. Small/wide capacity contrast cannot change the comparator posthoc.
Keep prior full-cohort recipe unless support is demonstrated on a comparable
contract; any winner here gets a named recipe for this new development cohort.
Do not tune on old/new tests or add adaptive hyperparameter searches in this family.

## Reproduction and closure

Freeze manifests, source/data hashes and all package files including inference
before launch; retain exact invocation/env/dirty state. Independent source audit
conserves rows, excludes all old heldouts, verifies components/permutation donors
and checks inventory by a separate aggregation. Live side-by-side validation and
exposures; tests hidden until decision. After all fits terminate, independently
recompute loss, Spearman/Pearson/RMSE and applicable500nM metrics/bounds; preserve
predictions, paired peptide-bootstrap intervals, plots, failures and parameter
coverage. Replay64savedTRAIN probes/checkpoint through evaluation/public predictor
on freshH100. Preserve failures; no silent fit retries. Close README, canonical log,
task review, supported common recipes and publication before final handoff.
# Descriptive checkpoint-selection audit added during execution

After observing seed42's IC50-matched validation trajectory, add an explicitly
posthoc diagnostic of the fixed pooled-IC50-RMSE selection rule. From saved
validation histories only, report the actual selected checkpoint and the steps
that would minimize each class's RMSE, maximize each class's Spearman, or minimize
the unweighted mean of the two class RMSEs. Verify the original selected step
against the original pooled-RMSE minimum and report class counts/weights.
This diagnostic does not select, refit, promote or test any alternate checkpoint;
all twelve original selected artifacts and promotion rules remain unchanged.
It assesses whether pooled selection sacrifices one class while improving the
more numerous class, separately from auxiliary-data interference and exposure.
