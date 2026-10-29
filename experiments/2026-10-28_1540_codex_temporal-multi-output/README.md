# Shared multi-output training and 2026 study holdouts

Status: **Cohort independently verified; two H100! fit calls launched on 2026-10-29 at 03:00 UTC. No validation results yet.**
Agent/model: Codex / GPT-6. Date: 2026-10-28 UTC.

[Prospective specification](../agents/codex/plans/2026-10-28_temporal-multi-output.md)
· [live report](http://127.0.0.1:54502/dashboard.html)
· [canonical registry](../experiment_log.md).

## Source and admission contract

Fresh canonical IEDB and CEDAR full exports were downloaded with Hitlist 2.0.40,
10,066,712,903 and 9,188,295,774 bytes respectively; see
[acquisition receipt](results/acquisition.json). The 5,208 source reference entries
include 45 distinct papers dated 2026, with PubMed publication-date XML retained.
A metadata-only inventory finds **no direct IC50, KD or EC50 measurements in those
2026 exports**. The user therefore authorized older whole-assay-study holdouts.
These are new from-scratch study splits, not claims that historical data were never
used in earlier architecture development.

Use canonical Hitlist indexing/export and the shared typed Presto adapters.
Admit unmodified peptides with resolved human MHC groove sequences from the pinned
mhcseqs catalog. Preserve every reported candidate MHC; do not assign a cognate
allele to a multi-allelic sample. Ordinary explicitly allele-resolved human DR may
use a declared DRA*01:01 reference, preserving original typing. Explicit partners,
construct exceptions, DP/DQ and unresolved serotypes retain stricter rules.
Molecular MHC species, antigen/source organism and experimental host are separate.
This is the compatible human-MHC cohort, not a claim to train on every raw export row.

Physical measurements remain distinct: IC50, KD, EC50, KD-via-IC50, KD-via-EC50,
half-life, off-rate, on-rate and td50. Literal qualitative binding and observed
elution are separate binary outputs. Bounds keep their original inequality;
training scales use exact TRAIN values only. Allocate only supported TRAIN heads.
Shared source implementation is pinned at `c8f032a7a200a1ea87121b016eb9dbb22c90e3db`;
[complete runtime file hashes](reproduce/runtime-pins.json) include 211 files.

## Frozen selection and model recipe

Prespecified 2026 test PMIDs: 41534530 (class-II MS), 42069060 (class-I MS),
42048449 (class-I KD-via-IC50); all three have eligible observations. A deterministic
metadata-count rule adds older assay studies for test and validation. Source-study
alias components remain whole; exact peptides are disjoint, with test > validation
> train priority. Final membership, exclusions and numerical budgets were frozen and independently
verified before any fit; see `results/partition-verification.json` and
`reproduce/training-contract.json`.

One shared-v5 wide 192/3/6/768 model and the canonical observation-manifest trainer,
seeds 42/43, no pretrained weights, dropout .1, AdamW 3e-4/decay1e-4, batch128.
Relative label frequencies: 60% elution,20% IC50,15% other physical outputs,5%
qualitative binding; within families, endpoint/class frequencies use square-root
counts. Canonical weighted quotas cover all unique TRAIN rows. The resulting budget is51,099updates per seed: one complete weighted epoch
of6,540,642examples, batch128 (final batch may be partial). Validate every1,000 updates;
equal-endpoint loss selects the checkpoint. Full validation/test dumps; a fixed
16,384-row cap applies only to post-fit TRAIN diagnostic predictions.

All outputs use the same peptide/groove encoder and full ordered ligand context.
Unsupported source-protein/cellular-context inputs are not allocated. Their source
annotations remain available for stratified reporting. This task does not qualify
native DO:DM, TCR or independently identified processing/detector mechanisms.

## Ligand comparison populations

Shared canonical backgrounds match positive length, MHC candidates, context and
study: one per TRAIN positive, five per evaluation positive, seed20261028, with
replacement. Use exact same-protein mapping when available, otherwise report the
frozen human Ensembl112 proteome population separately. Exclude all known positive
peptide sequences and other-split sequences. These are unobserved comparisons,
not measured non-binders. Report length-only controls and same-protein/fallback
AUCs separately. The complete positive blacklist includes held-out sources for
conservative background exclusion only.

Source-reported MS negatives are retained as **assay-scoped nondetections**, with
study-specific AUCs separate from generated backgrounds. A large historical source
is HLA-B27 transgenic rat/ERAP1 data, so human MHC must not be relabeled as human
host or a universal non-presentation claim. Mixed/unknown conditions remain explicit.

## Execution and verification

Source preparation: CPU8,96GiB RAM. Attempt2 failed at Hitlist's default16GiB
provenance scratch limit; raw downloads survived. Attempt3's explicit256GiB disk
request was rejected before a worker started. Attempt4 asserts128GiB scratch,
16GiB reserve and512GiB requested ephemeral disk. All attempt receipts/logs are
preserved. Dropped local log/controller connections do not restart the remote build.

Predictive resources: requested `H100!`, maximum two workers, CPU8/64GiB RAM each.
Both workers report NVIDIA H100 80GB HBM3,85,009,367,040device bytes, Torch2.14.1+cu130,
CUDA13.0; TF32/AMP/compile disabled. Fit runtime and held-out metrics remain pending.
There is no preferred new checkpoint yet.

Software evidence: the current shared suite passes2,805 tests/four skips;
the literal purified-method and reported-MS-negative admission checks passed79
and89 focused tests, respectively. Thirty-two preparation/report/controller/recovery fixtures passed before launch,
including the real ObservationIndex/common-trainer preflight, independent source/background audit and separation of
reported-negative versus generated-background AUCs. Artificial tests are not
biological performance evidence.

## Reproduction and artifacts

- `code/acquire.py`, `build_index.py`, `admit.py`: canonical source stages.
- `code/partition.py`, `prepare_worker.py`, `prepare.py`: fixed cohort preparation.
- `code/launch.py`, `worker.py`: two-seed common-trainer fits and live validation.
- `reproduce/`: source snapshots, exact invocations, failed-attempt dispositions,
  dependency call IDs and pinned package/reference hashes.
- Modal volume `presto-checkpoints`, `/runs/2026-10-28_1540_codex_temporal-multi-output/`:
  source downloads/index, admission SQLite, prepared manifests and eventual raw fits.
- Local large artifacts: repository `artifacts/2026-10-28_1540_codex_temporal-multi-output/`.

Final closure still requires: all fits, complete validation/test prediction dumps,
independent metric/source checks, selected-checkpoint TRAIN-probe serving replay,
per-output/class/study summaries and uncertainty, canonical registry/model-reference
updates and the shared recipe handoff. PR131 stays draft pending scientific gates.

## Fresh source build and admission progress

The full canonical build completed at 2026-10-28 17:58 UTC, yielding 5,844,451 MS,
821,470 binding and26,694 other-assay rows. The source inventory is large (298.8MB)
and lives at the hashed location in `results/source-inventory-location.json`.
The canonical upstream build excludes class-II peptides shorter than12 residues,
including150,931 MS and3,273 binding rows. This is an upstream curation policy,
not a claim that shorter class-II ligands cannot bind.

Admission attempt4 failed during remote launcher import, before data processing;
local filesystem upload paths are now constructed only locally. All three
launchers passed remote-path import checks. Attempt6 completed against the source build with the verified attempt5 database backup.
Admission retained2,023,956 original observations:263,432 physical/qualitative
measurements and1,760,524MS observations. Direct IC50 I/II counts before partitioning
are19,167/38,543;279,583rows across all outputs use declared reference DRA.
The complete receipt is `results/admission.json`.

The source also contains468,548 high-throughput multiplexed binding records not
covered by the frozen soluble-affinity/qualitative-purified adapter. Of these,
418,890 are the peptide microarray study PMID32903714; its exported labels are
not numerical affinity measurements. Other display-based studies share the broad
method label. See `results/multiplex-assay-inventory.json` and
`results/unsupported-assay-inventory.json`. A future typed microarray/display
readout must preserve its measurement scope instead of converting it to IC50.
Primary source: https://pubmed.ncbi.nlm.nih.gov/32903714/ ; source assay metadata:
https://www.iedb.org/assay/17059219 . These rows are explicitly excluded from the
current frozen admission contract, not silently counted as trained examples.

## Earlier operational evidence (historical)

As of2026-10-28 19:11 UTC, source build and the binding/other-assay passes are
complete. The canonical human MS export contains5,440,009rows. Attempt5's measured
rate of306rows/s would exceed its four-hour limit. A consistent read-only SQLite
backup was committed to the volume before stopping that worker. The2.57GB backup
contains406,381retained rows and1,703,850known-positive sequences; its exact hash
and receipt are in `results/admission-local-recovery-backup.json`.

Attempt6 uses512GiB worker-local disk and a1GiB SQLite page cache, replaying the
same source rows through the unchanged scientific adapters. All406,381snapshot
rows matched byte-for-byte. Counters and per-row dispositions are regenerated;
status counts are never used as a transactional resume ordinal. The metadata-only
vocabulary pass projects14fields, while actual admission retains all161fields.
The recent MS processing rate was about1,460rows/s. At19:11UTC925,696MSrows had
been checked. Active call: `fc-01M61Y051TV4HX19H6F0W50P6N`, app
`ap-i9Xi2zCOm6fs7EnhbbEF1a`. Completed replacement artifacts will preserve attempt5
in a separate directory. The earlier stage-guarded backup attempt created nothing;
its failed receipt remains alongside the successful later backup.

The queued preparation controller depends on attempt6. Local DNS failures are
retried with bounded saved-call polling; genuine remote timeouts, source hash
mismatches and disk failures still fail immediately. Profiling found a costly
full-column Arrow conversion but did not establish its relationship to heartbeat
transport failures. Diagnostic receipts, original snapshots and logs are retained.

`code/advance.py` owns the local continuation lock. It waits for the queued
`code/prepare.py` call, requires its independent audit, freezes the exact contract,
launches at most two H100! fits, collects hash-verified artifacts, freezes seed
selection from validation only, and checks saved predictions with independent
SciPy/sklearn calculations. It does not mark the scientific writeup complete.
The selected checkpoint is also replayed through guarded public serving on TRAIN
probes only. Final scientific interpretation and common-recipe promotion remain open.

Preparation and GPU workers use512GiB requested local disk for indexed reads,
with the same canonical manifest paths and hashes; completed datasets and fit
outputs are durable on the Modal volume. This changes no rows, model code or
numeric precision. `results/ci-2f6f1ca8.json` records passing CI for the pushed
shared-source/experiment commit; PR131 remains draft.

## Verified cohort and active fits

Preparation call `fc-01M627VM7J2SPPR71V2MRSHT08` completed after Modal preempted
its first CPU container and automatically retried the same input. A closed-file
materialization backup was also committed, with hashes in
`results/materialization-backup-completed.json`. The completed cohort passed both
canonical preflight and independent original-target, study-alias, exact-peptide,
background-lineage/blacklist/ratio and TRAIN-scale checks. No duplicate preparation
was submitted. The verified audit removes127,096original rows by peptide embargo.

| Partition | Original observations | Generated comparators | Total rows |
|---|---:|---:|---:|
| TRAIN | 1,875,420 | 1,617,732 | 3,493,152 |
| Validation | 6,534 | 12,050 | 18,584 |
| Test | 14,906 | 59,250 | 74,156 |

TRAIN includes54,736direct IC50 measurements (18,531I/36,205II),17,832KD,212EC50,
72,925KD-via-IC50,38,932KD-via-EC50,8,729half-life,22koff,6kon,503td50,
40,460qualitative binding and1,641,063original MS observations. Every allocated
output has TRAIN support, but koff/kon have no eligible held-out observations;
they are unvalidated outputs, not evidence of generalization. Some other
output/class cells have no validation or test support; retain explicit N/A.
All molecular MHC is human; experimental hosts and peptide organisms vary.
`results/cohort-strata.json` preserves method, host, source organism, peptide
length, candidate count and reported allele counts on original observations.
Reported-allele membership counts can overlap for multi-allelic observations.

[Modal app](https://modal.com/apps/iskandr/main/ap-Du3wtmTBI9AvARUgRQ9w3v):
seed42 call `fc-01M62SMM6TFPHZKNMQJ45MP221`;
seed43 call `fc-01M62SMM9W381JKPPKCKJP0SDK`.
The exact invocation and dirty state are in `reproduce/launch.json`; scientific
runtime remains the211-filec8f032a7 snapshot. Experiment observer state is bound
atomically to common resumable model/optimizer/RNG/order/evaluation checkpoints.
Three interruption tests establish exact restored weights, optimizer, coverage and
evaluation-cache reuse on artificial CPU fixtures. Recovered elapsed/optimizer
seconds exclude rolled-back work; per-attempt GPU peaks are not total paid compute.

The local dashboard poller now has a30-second remote-read deadline and retains
last-known evidence with an explicit poll error. It does not resubmit jobs.
Validation appears every1,000updates; test reporting remains behind the saved
validation-only selection and independent-prediction-verification gates.

## Canonical allele composition audit

The initial raw-metadata table has reported-candidate allele counts for MS but
lacks the analogous physical-assay provenance field. A supplemental read-only
audit now counts canonical catalog alleles for every original observation,
preserving class-II paired candidates separately from their allele union.
`results/cohort-composition-verification.json` reconciles every original
split/endpoint and every total split/endpoint/class count with frozen preparation.
Individual allele memberships may overlap; observation totals never do.

Original TRAIN MS counts are 1,348,309 class I and 292,754 class II. Direct IC50
counts are 18,531 and 36,205. Class-I IC50 includes 5,880 HLA-A*02:01 observations.
Full original support, catalog allele memberships, candidate sets, studies and
qualifiers are in `results/cohort-composition.json`. The metadata audit ran in
existing worker `ta-01M62SMMGKXEEGYE67SFBSRR3R` without model evaluation or new GPU
allocation; its source and invocation are frozen in `reproduce/`.

![Original support versus planned optimizer exposures](results/training-composition.png)

The figure describes the frozen sampling design, not predictive performance.
PNG/PDF and the exact plotting invocation are preserved; actual optimizer
exposures and selected-checkpoint coverage will be reported after fitting.
