# Assay-only real-data memorization control

Prospective plan, 2026-10-21 UTC; Codex / GPT-6. This instantiates the supplied-
metadata control in the completion addendum. It does not replace the auxiliary,
full-cohort, APC, MS or TCR requirements, and cannot qualify generalization.

## Question and source contract

Can the small direct IC50 network fit 32 real, consistent observations without
auxiliary classification, target clipping or competing assay losses?

Reuse the pinned Hitlist `ci-corpus-v4/binding.parquet` (SHA-256
`2b3f6ea2dd647e7fc54d214eef420feaebc4e81fab97ffc0565a2de1f8695ba0`).
Use only IEDB direct exact IC50, nM, purified-MHC assays, unmodified canonical
peptides of length 8–25 for I and 10–25 for II, and explicit human HLA restrictions.
Class II must name both alpha and beta alleles; no default partner is inserted.
Resolve named alleles from the actual pinned local mhcseqs catalog (SHA-256
`ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1`),
requiring human, functional, non-null/non-questionable/non-pseudogene `ok`
groove exports. Verify selected grooves against the installed parser and pin its
version/source. Do not mutate the shared environment or rebuild a mutable catalog.

The earlier corrected-human cohort's 976 exact class-II training observations
name only beta chains. The full v4 export contains explicit pairs elsewhere, so
this is a new diagnostic cohort, not a replay of the old 32/full-fit membership.
Exclude every old validation/test peptide using the pinned historical membership
table (`8959e60c0d9b8049aec25095b9c354d456b48f97ba8fb31d7b786e2861113e99`).
All selected observations become training-only inputs; their IDs must also be
excluded from future independent evaluations that seek to avoid diagnostic exposure.

Deduplicate original assay IDs, reject conflicting copies, and exclude all
identical canonical input tuples with disagreeing exact labels from this
diagnostic. Retain the conflict report for full-data noise analysis. Select one
assay per consistent tuple, then the first 16 per class by SHA-256 of seed 42
and the canonical tuple, independent of target value. Require >=1 log10(nM)
range in each class before fitting; failure stops this recipe. Write per-row
source dispositions, curation counts, exact selected inputs/labels/source IDs,
catalog records and hashes before optimization.

## Training and fixed gate

- Shared encoder: width 128, 2 layers, 4 heads, current modular implementation;
  direct IC50 head and explicit class-II registers. Supplied human molecular
  species/class; no auxiliary tasks or replay, no APC/MS/TCR branches.
- AdamW, constant learning rate 0.0003, weight decay 0, dropout 0, seed 42.
  Full batch 32, all allocated parameters trainable, no pretraining, no generated
  labels or synthetic examples. Raw log10(nM), training-only target standardization,
  one squared loss per observation.
- Maximum 2,000 updates or 20 minutes of optimization/evaluation wall time,
  whichever arrives first. Local CPU, one Torch/BLAS thread; no GPU/Modal.
  Check the fixed gate every 25 steps, plus at the budget limit; stop on first
  overall AND per-class pass, or preserve a failed/budget-limited result.
- Gate, overall and each class: exact training RMSE <=0.10 log10(nM), Spearman
  >=0.95, and prediction SD / target SD in [0.8, 1.2]. Require real gradient-
  driven updates for every allocated parameter and observed feature, including
  register pooling. Save initial/intermediate/final predictions and coverage.
- No validation/test partitions: this deliberately tests capacity/optimization
  on training labels and consumes no generalization estimate. Do not promote a
  checkpoint, claim an accurate register, or compare these metrics numerically
  with the historical held-out collapsed-model scores.

## Closure

Freeze the launcher and invocation under a registered timestamped directory,
record actual git commit/dirty state, package versions, runtime/RSS, input pins,
conditions, all observed metrics and a fresh checkpoint prediction round trip.
Record mean-target training comparison for scale context, without treating it
as held-out benchmarking. Write results and the unchanged gate decision to the
family README and canonical experiment log even on failure. A failed gate stops
expansion and requires a new registered one-factor diagnosis.
