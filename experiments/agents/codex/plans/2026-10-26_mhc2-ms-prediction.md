# First qualified MHC-II MS detection-enrichment comparison

Codex / GPT-6, 2026-10-26. Prospective family:
`experiments/2026-10-26_2030_codex_mhc2-ms-prediction/`.
This advances the MS part of the full objective after the verified1954 inventory.
It does not qualify isolated processing pathways or replace later condition work.

## Question and data

Does a native Presto MS output, initialized from fixed IC50 features, discriminate
reported class-II MS detections from a declared generated background better than
binding-only and simple sequence controls? Does that signal transfer from the
qualified DP source to a separate DQ8 study? Performance is conditional on the
background distribution, not natural prevalence or measured nondetection.

Use only unmodified, source-positive admitted observations from DP PMID35251023
and DQ8 PMID34433824, with the prior source hashes and all contributor identities.
Require exactly one reported complete paired molecule. Preserve all exclusions.
No new allele assignment, default alpha partner, outcome-imputed quantity, native
cell-expression claim or pathway inference. The DQ8 paper includes spliced-peptide
analyses; retained source detections are not asserted to be exclusively linear.
The exact pair comes from the deposited source, separately from capture evidence.

Collapse identical study/peptide/groove-pair units into one modeling observation,
keeping every original ID and source offset in a lineage table. Choose the
lexicographically first original ID as the representative; do not choose by score.
The original full JSONL and contributor records remain immutable.

The primary model inputs are peptide and complete MHC pair. Existing generic
condition annotations are preserved in original source/lineage, not supplied as
asserted unengineered conditions. This first comparison has no contrasted
processing-state supervision; leave pathway/cell/intervention vocabularies empty.

## Prospective partition and exposure policy

Connect DP peptide units that share any contiguous nine-residue sequence. Assign
each connected component by SHA256 of `presto-ms-dp-v1|` plus its smallest peptide:
first eight digest bytes as an unsigned fraction, train<.70, validation<.85, else
test. All molecule units for a peptide share one component/split.

Remove entire validation/test components sharing any nine-residue segment with
any input peptide from the fixed IC50 parent's original train/validation/test
partitions. Training components may retain those overlaps, explicitly counted.
Holdout outcome history of the IC50 family is development; the new MS split is
frozen here before MS predictions. A shared segment exclusion does not establish
protein-homology, donor, specimen or search independence.

Use DQ8 only for external testing. Exclude any DQ8 peptide sharing a nine-residue
segment with any DP peptide (including excluded DP) or any original IC50 input.
Build DQ8 connected components for uncertainty reduction. Require at least
200 positive peptide units and20 components in each DP validation, DP test and
DQ8 external set; otherwise stop preparation and preserve the failed contract,
without rehashing/reassigning to achieve counts.

## Generated-background contract

Generate one deterministic amino-acid permutation per retained positive peptide,
shared across that peptide's molecule units. Keep exact length/composition and
the same pair/source/split. Seed a local PRNG from SHA256 of
`presto-ms-background-v1|` plus the positive sequence. Try at most1000 successive
shuffles; reject unchanged sequences, any known positive MS sequence in the full
hash-pinned native export, previously generated sequences, and any sequence with
a nine-residue segment occurring in another split's retained positives or controls.
For validation/test/external controls also reject initializer-exposed segments.
Generate in sorted split/peptide order with a precomputed mapping of all positive
segments to their partitions. No predictor scores participate.

Fail preparation if any retained peptide lacks a valid control under this policy;
do not silently drop hard examples. Controls are source-labeled generated sequence
background, never measured negatives. Preserve generation seed/attempt/parent IDs.
Each positive/control pair gets equal row mass; report counts by molecule and
component and paired/cluster uncertainty rather than treating rows as specimens.
Output1 means observed detection,0 means generated control; no censor bounds apply.

## Fixed comparison and initialization

Freeze the verified old seed42 IC50 selected checkpoint, SHA256
`db2585e63489cb5aac8c7a32fd7e522bcbc51035846c05f8cd2794c4449e3dc5`.
Do not select a censor-refined initializer after seeing its results.

- Binding-only logistic control: one parent IC50 log10(nM) score, standardized
  using DP training inputs only; intercept and coefficient fitted on training.
- Sequence logistic control: length,20amino-acid fractions,20N-terminal and
 20C-terminal indicators; training-only standardization.
- Combined logistic control: sequence features plus fixed parent binding score.
- Native MS seeds101/102: canonical `fit_index`, retain IC50 endpoint and its
  full frozen parameter dependency set; append `elution_observation`.
  Empty new context vocabularies. All newly allocated MS/context parameters train.
- Native training-label permutation diagnostic, seed101: same cohort/config/budget.
  For each positive/control pair, swap training labels using a deterministic
  hash coin `presto-ms-label-null-v1|` plus peptide. Original evaluation labels
  remain unchanged. It is one diagnostic permutation, not a calibrated null test.

Logistic controls minimize mean binary cross-entropy plus.01times squared slope
norm, unpenalized intercept, float64 L-BFGS-B, max300iterations/ftol1e-12/gtol1e-8;
fail on nonconvergence. No hyperparameter sweep or held-out fitting.

Native budget is exactly three complete epochs over the same DP training rows:
batch64, AdamW3e-4, weight decay1e-4, dropout.1, identity weight0, seeds above,
checkpoint every50steps. Freeze exact update counts after cohort construction,
before predictions. No early stopping within these three epochs. Primary is the
fixed-final checkpoint; canonical validation-selected output is secondary.
Primary native result averages logits from the two true-label final checkpoints.
Never choose the better seed or select a replacement based on either test set.

The elution output jointly combines learned cellular and detector-related scores.
Only their total is supervised; improvement does not identify the individual
components or establish independent antigen-processing prediction.

## Evaluation and gates

Save source-linked predictions for DP validation, DP test and external DQ8 test
for every arm, with labels, split/component/molecule/source IDs, logits,
probabilities, controls and exposure disposition. Report BCE, Brier score,
accuracy/balanced accuracy/precision/recall/F1 at.5, AUROC and AUPRC. IC50 nM
regression/500nM binding metrics do not apply to MS labels; instead verify exact
preservation of all parent IC50 parameter tensors/scalers and fresh no-grad
prediction probes to2e-5log10(nM). Independent metric reduction must agree.

Primary success requires native ensemble BCE at least2% lower than each of
binding-only, sequence and combined controls on both DP test and DQ8 external,
with AUROC no more than.01 lower, plus native AUROC at least.02 above the
permuted-label diagnostic on both. Also report validation and both individual seeds.
Use2000 paired component-bootstrap draws, seed20261026, for relative BCE gain
and AUROC difference against combined control; if either test's BCE-gain95% CI
includes zero, call improvement inconclusive even if point gates pass.
Report all failed gates. Generated-background discrimination cannot qualify
measured-ligand prevalence, independent binding or processing mechanisms.

## Resources, verification and closure

Local CPU only; no GPU/cloud/install. Preparation/verification1thread,300s/2GiB RSS;
fit/inference2threads/1interop,2h/action/4GiB RSS. Keep the original10GiB available
RAM and disk floors, start only at13GiB RAM/15GiB free disk. Do not overlap a new
native MS fit with censor recovery; preserve older controllers. Stop on failure,
no automatic retries or reserve reductions. Raw data/checkpoints under
`/private/tmp/presto-mhc2-20261026/ms-prediction/`; compact published summaries,
lineage and prediction dumps under the family, maximum250MiB.

Before predictive execution, verify original input hashes, disjoint components
and controls, exact length/composition matching, source-positive exclusion,
initializer exposure, source joins and manifest acceptance. Use artificial tests
that deliberately violate leakage/control/lineage invariants. Independently
replay cohort membership/generation from original source bytes. Freeze invocation,
code/runtime hashes, git status and resource receipts before fitting. Preserve
all failures. Close through all metrics, independent verification, useful plots,
README and canonical log; do not stop at launch.

- [ ] Implement and verify immutable source-linked cohort.
- [ ] Freeze manifests and exact update budget.
- [ ] Run fixed controls and all native arms under resource limits.
- [ ] Independently verify predictions/metrics/retained IC50 behavior.
- [ ] Publish complete validation/test outcomes and limitations.
