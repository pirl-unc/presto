# Active completion progress

## 2026-10-20: training support and measurement preparation

The full objective in `completion-spec.md` remains active. This entry records
progress, not completion of the architecture or scientific qualification.

Implemented:

- Feature-column/category and residue-row exposure/update evidence, checkpoint
  persistence, and inference rejection of unseen effects. Tensor coverage alone
  no longer authorizes a feature. Unobserved numeric slots fail preflight.
- A required coverage closure gate, with artifacts preserved on failure and a
  narrowly declared, justified training-only diagnostic exception.
- Original measured-outcome lineage checks independent of export IDs. A disk-
  backed preparation primitive emits per-row dispositions and pinned per-source
  files, deduplicates copies, defers conflicting exports, retains independent
  replicates and keeps duplicate group aliases for split checks.

Verification: four new regressions failed against the previous implementation
for the expected missing guards. After changes, 56 modular and documentation
checks pass; pinned Ruff lint/format pass for all changed Python files. An older
candidate-ranking software fixture needed training exposure to its second
candidate's PFR feature columns; it can no longer silently use random columns.
No real-data learning result has been claimed or run for this change.

Data inspection:

- Weirdo has `data/swissprot-sequences.csv`, `data/swissprot-8mers.csv` and
  `data/category-splits/` in its existing checkout. The prepared predictor uses
  ten independent category-membership labels and family-disjoint folds. This
  should train a multilabel auxiliary, not a mutually exclusive origin species
  classifier or a substitute for measured recognition.
- Hitlist's release inventory includes `ci-corpus-v4` and
  `human-cohort-corrected-v1`. The GitHub `releases/latest` pointer still returns
  `ci-corpus-v3`, so that endpoint is insufficient for export freshness. Actual
  manifest/hash/schema verification and source preparation remain to do.
- The local filesystem has about 17 GiB free; reuse pinned existing inputs and
  bounded disk-backed preparation rather than duplicate large source corpora.

Still pending at that checkpoint: the early-supervision extensions, cell/cytokine/profile machinery,
quantitative routes, actual source adapters/cohorts, continuation/freezing/resume,
scientific evaluation gates and registered fitting/transfer evidence, plus final
PR/CI validation. TCRvdb access remains unverified/unavailable from prior work.

## Early-supervision implementation and identity/IC50 clarification

The working implementation now adds independent peptide-source membership heads,
MHC/TCR pairing supervision with explicit evidence requirements, per-category
support checks, and optional supplied/override categorical inputs with endpoint
masks. The Weirdo adapter reads hash-pinned prepared split archives and preserves
source fold/group/category semantics with declared label-blind sampling. These
are executable source/model interfaces; no real-data qualification is implied.

The user questioned the rationale for "identity plus IC50." The code trace
confirms that rich sequence representations feed the assay model; annotation
heads are optional auxiliary supervision. Assay updates still change the shared
encoder even when categorical predictions are detached. The docs now describe
that drift explicitly. The completion spec requires a supplied-metadata assay-
only control and separate pretraining/replay factors before claiming transfer
benefit; those comparisons are prospective and have not run.

Verification: 63 modular model/training/documentation tests pass in 5.76 seconds;
pinned Ruff lint/format checks and diff whitespace validation pass. The published
`9e97fd021ace06c80f2269ce91767dd3f08e4573` head has four successful GitHub CI jobs;
that CI evidence does not cover these subsequent working-tree changes. No new
real training experiment, calibrated auxiliary predictor or qualified assay
checkpoint exists. Full-goal source, context, MS, TCR and scientific gates remain
open.

## APC cell-type/cytokine context implementation

Added typed independent cytokine exposures, supplied/override cell type, optional
unit-declared dose and duration, and sourced/versioned machinery reference
profiles. Species-wide and cell-specific defaults remain visibly marked priors;
sample annotations (including unknown) and explicit overrides take precedence.
The current repository does not yet supply a qualified biological profile catalog.

The existing cellular fusion consumes these fields; purified molecular assays,
fixed TCR-pMHC binding, digestion and the shared detector remain invariant. The
same feature-column support guards cover new categorical/numeric columns and
checkpoints preserve their meanings. Preflight requires cell-category exposure,
present/absent cytokine contrasts and varied declared numeric values. These do
not establish matched-cohort causal effects; source attribution and controlled
predictive comparisons remain necessary.

Verification: 75 focused modular and documentation checks pass, including actual
new-column updates, unsupported-state rejection, profile/checkpoint round trips,
reference precedence, concurrent exposures, class/cellular routing and detector
invariance. Pinned Ruff lint and diff checks pass. A lint failure in the new
parameterized validation test was corrected with an explicit error-message
assertion before the successful rerun. No real-data fit or perturbation-response
claim has been made. Quantitative MS/processing endpoints remain the next missing
architecture paths, alongside the wider source and qualification backlog.

The immutable source commit `701db80ebe2da9cb158050908fdf8a98f546d358` passes the
complete local regression: **2,445 passed, four skipped, 712.26 seconds**. The
skips retain their existing explicit qualification/environment conditions.
Repository-wide Ruff lint/format and strict MkDocs pass. Source/test files were
unchanged throughout that run; no process was restarted based on quiet output.

## Source inspection and upstream quantitative requests

The actual current release inventory and cached bytes were inspected. The source
inventory is [source-inputs.json](source-inputs.json): v4 binding/other-assay/bulk
tables and corrected-human manifest/training/context files match GitHub release
digests. Existing datacache generations are reused in place. Weirdo reference and
prepared archives now have actual local SHA-256 pins. This is source inspection,
not a new fitted experiment or admission/independence certificate.

The corrected-human schema contains cell type, reviewed condition/control fields,
independent perturbation/cytokine annotations and lineage fields. The new adapter
still needs to inspect their actual contents, preserve missingness and produce a
complete admission/disposition report; schema presence does not fill those gaps.

Filed [Hitlist #726](https://github.com/pirl-unc/hitlist/issues/726) for quantitative
HAP1 elution/proteomics outcomes and replicate lineage. The inspected packaged
membership CSV contains only peptide, class and restriction, while the primary
publication reports quantitative profiling and deposited processed data. Also
added [requirements to #95](https://github.com/pirl-unc/hitlist/issues/95#issuecomment-6307156975)
to keep raw and normalized bulk intensities separate and preserve actual run/
replicate and censoring evidence. Neither request claims original supplementary
tables have already been downloaded, parsed or admitted here.

## Quantitative observation paths and preparation context

Added explicitly declared quantity domains for bulk/elution observations and
independently measured MS/digestion/cellular component targets. Each domain pins
meaning, canonical unit, normalization, reference and target transform. Raw log
encoding and real censor bounds survive admission, target conversion, prediction
dumps and public inference. Incompatible domains receive separate scalar
readouts; within a domain the MS readout is shared across bulk/elution/anchor
paths. Feature blocks remain shared with binary observations. Quantitative-only
subsets allocate no unused binary readout parameters.

Outputs stay in the declared transformed measurement space, with training-only
residual scaling inside the loss. Bags have one aggregated outcome/loss, with
log-base-correct averaging. Numeric digestion preparation features have declared
units and missingness and affect digestion alone. Allocated preparation channels
require varied training values. Component-target admission requires explicit
independent-measurement evidence; this is an auditable source assertion, not a
certificate of identifiability or calibrated detector response.

Verification: **84 focused tests passed in 5.21 seconds**; repository-wide Ruff
lint and format checks pass (331 files), as does diff whitespace validation. The
first focused quantity run exposed an overly specific test error-message match;
the corrected assertion passes. These checks include an artificial streaming
fit, support closure and checkpoint/inference round trips. No real-data fit or
anchor/transfer experiment ran. All four GitHub checks on preceding published
head `c66caae59eed3a7acef6aaea8cff7ad5d90937e9` succeeded; they do not cover these
subsequent quantitative changes.

Direct biochemical cleavage/transport supervision is still a distinct missing
path: `cellular_quantity` means available pMHC, not an arbitrary processing assay.
Real quantity adapters/cohorts, independent anchors and transfer qualification,
continuation/freezing/resume, broader scientific gates and the other full-goal
requirements remain open. Full regression for this source slice is pending.

Full regression subsequently completed on immutable application/test source
`51b5f61`: **2,454 passed, four skipped, 1,052 warnings, 604.90 seconds**.
Strict MkDocs also passes. During regression only the next experiment's plan,
launcher and preparation artifacts were added; application/test sources stayed
unchanged. The warnings are preserved in the local full regression log.

## Prospective assay-only capacity diagnostic

Registered `2026-10-21_0353_codex_modular-memorization-control` before fitting.
The earlier corrected-human cohort has 976 exact class-II training observations,
all with beta-only reported restrictions. The pinned v4 binding table contains
explicit pairs elsewhere. The diagnostic therefore prepares a new, training-only
cohort from v4 and excludes historical validation/test peptides; it is not a
replay of the earlier membership. No alpha partner is filled in by convention.

Source preparation scanned 814,268 rows, retained 26,619 eligible candidate rows,
identified 841 conflicting exact-input groups (excluded only from this diagnostic)
and selected 16 consistent inputs per class by a label-independent hash. All 32
raw targets exactly replay as log10(nM). Each class spans over one log10 unit;
class-I targets include measurements above the historical clipping ceiling.
Selected catalog groove exports agree with the installed parser. Full source
and candidate dispositions are hashed local artifacts; small cohort and source
receipts are reviewable. The fixed fitting gate has not yet run at this entry.

The diagnostic then ran from frozen `122b611124563b653fbddbbce30f921ea0f14fff`
and passed at 100 updates in 33.05 seconds on one CPU thread. Overall/class-I/
class-II RMSE was 0.03294/0.04333/0.01711 log10(nM), Spearman was
0.99927/1.00000/0.99706 and SD ratios were 0.99978/1.00000/0.99897. All 32
allocated parameter tensors and all exposed feature columns have actual update
evidence; register weights updated on every step. A separate process exactly
reproduced predictions/metrics and passed the public predictor's support guards.

The complete experiment README, source pins, per-example predictions, coverage,
history, register diagnostics, plot and raw-checkpoint locations are recorded in
the registered family and canonical log. No validation/test split was used by
design. This establishes capacity on 32 fit targets with supplied metadata and
no auxiliary tasks; it does not establish generalization, auxiliary utility,
correct register localization or the historical model's collapse cause. The
remaining full-cohort/pretraining/context/MS/TCR/resume/qualification work is
unchanged.

## Subset transfer and frozen dependencies

Added explicit compatible task removal, named projection of task-dependent
feature columns, retained endpoint scalers, and declarations that freeze entire
trained functions. A frozen output includes its shared encoder/fusion/pooling
dependencies; a frozen MS component includes encoder/features/readout. Dropout
stays disabled in the frozen encoder. Parameter flags, dependency declarations,
feature support and optimizer membership are checked and survive checkpoint
round trips. New heads/columns remain unsupported. This does not create a
scientific validation status or a calibrated MS component.

The manifest runner can retain frozen outputs without new labels, exclude their
rows from optimization, and train sibling modules with current-update coverage.
Inherited scalers cannot silently be refit on a different cohort. Checkpoint
selection excludes frozen tasks, so a constant retained IC50 metric cannot
choose a newly trained KD/MS/TCR checkpoint. Shared updates invalidate dependent
validation status. Exact interrupted-run resume is not implemented by this slice.

Verification at this stage: **91 focused checks pass in 3.30 seconds**, plus
repository-wide pinned Ruff lint/format (334 files) and strict MkDocs. A new
annotation-retention assertion initially compared autograd and inference
attention modes and differed by one float32 ULP; comparing both in the actual
inference mode gives exact equality. Tests use artificial observations and do
not alter the prior registered real-data fitting evidence.

Full regression on immutable application/test source `1108838` completed with
**2,461 passed, four skipped, 1,056 warnings in 394.35 seconds** using four pytest
workers. The log is `/private/tmp/presto-continuation-full-regression.log`.
Only planning documentation changed while that test process ran. The next
[resume specification](resume-spec.md) describes durable optimizer/RNG/sampler
state, selection recovery, pinned execution contracts and test-cache reuse;
that implementation and its equivalence checks remain open.

## Interrupted-run resume implementation

The next slice adds one explicit resume path to both training CLIs. Durable
snapshots preserve optimizer membership and state, global/local RNG states,
sampler/replay order and cursors, partial-epoch accounting, feature/parameter
support, frozen dependencies, scalers, patience and selected-checkpoint bytes.
The data/configuration/source/runtime contract must match. Concurrent trainers
in one run directory are rejected; a stopped process releases its lock.

Finalization has a committed end-of-optimization phase, per-split atomic caches
bound to the selected checkpoint, and hashed completion evidence. A reporting
retry does not rescore completed splits. Test consumption interrupted without a
complete cache fails closed; completed coverage failures remain failures.
Registered retries preserve the original recipe, append attempt/failure records
and close the registry once. CPU/CUDA/MPS RNG support is implemented, but only
CPU equivalence has been tested. Time lost after an earlier durable snapshot is
not included in the recovered elapsed-time accounting.

Verification before final broad regression: **128 focused modular/CLI/docs checks
pass in 16.33 seconds**. New tests compare exact weights, optimizer states,
coverage, history, selection and prediction records against uninterrupted runs
in a separate Python process, including dropout, replay and uncommitted work.
They also cover frozen dependencies/scalers, patience/earlier selection,
changed contracts and optimizer order, missing snapshots, concurrent writers,
incomplete test evaluation, completed-result tampering and both registered CLIs.
The first changed-data fixture used an invalid blank JSONL record; it was replaced
with a valid changed numeric measurement so the intended resume contract gate
is exercised. An initially mistyped test filename collected no tests; the recorded
128-test run used actual files. Pinned Ruff lint and format pass (337 files).
Strict MkDocs passes. Final full regression on immutable application/test source
`4b8f4df` completed with **2,476 passed, four skipped, 1,056 warnings in 770.15
seconds**, using four workers. The log is `/private/tmp/presto-resume-full-regression.log`.
The run was observed live through a quiet interval, with two workers actively
using CPU; it was not restarted. Only data inventory/planning documentation changed
while it ran. This is software evidence; real-data qualification, auxiliary
benefit and the full biological/data requirements remain open.

During final regression, a data-only [auxiliary source inventory](auxiliary-source-inventory.json)
reverified actual hashes for the MHC catalog and Weirdo reference/splits. It
records 56,440 raw MHC entries and 569,516 reference proteins. Class-II entries
are individual alpha/beta chains (3,193 alpha grooves and 17,728 beta grooves
present before quality filtering), not observed pairs. Raw species fields also
include unresolved taxa and need curation. Weirdo uses mnemonic-prefix grouping
and 20 hash buckets; these are not demonstrated homology clusters. The existing
table-verification statistic compares another k-mer table and does not replace
selected-row auditing against the pinned reference. The next [admission specification](auxiliary-data-admission-spec.md)
preserves these distinctions before auxiliary pretraining. No fitting ran during
this inventory. Upstream Hitlist #723 and #726 remain open without replies.

## Audited peptide-source pretraining

The first bounded reference family is now closed. The old pinned Weirdo archive
lacks the newer adapter's assumed grouping arrays; its failed preparation is
preserved. An explicit amended path reconstructs memberships, occurrences and
source-family groups without inventing supplied buckets. Independent scanning of
569,516 reference proteins validates all fixed 7,000 selected labels and folds.
There are no selected cross-fold occurrences; mnemonic-prefix grouping still
does not establish homology independence.

Frozen fit source `1d3c96b` trained the peptide-only auxiliary subset using the
canonical trainer and fixed prior/composition controls. The network selected
epoch 5 and stopped at epoch 10. Validation/test mean CE is 0.285826/0.302032,
versus composition 0.286331/0.300864 and prior 0.292560/0.308435. The network beats
prior but its validation difference from composition has a 95% interval crossing
zero; the joint prospective screen fails. All 29 allocated tensors/exposed
features updated, and first-training-batch predictions replay exactly in a fresh
process. Per-category metrics, all predictions, source/recipe/coverage artifacts
and a plot are retained in the registered family and canonical log. Independent
sklearn AP/AUROC/CE recomputation agrees on all 90 cases.

This is actual nonconstant auxiliary fitting with a negative baseline-comparison
result, not demonstrated affinity transfer or complete annotation qualification.
Single-chain annotation admission, real MHC/TCR cohorts, matched affinity controls
and the rest of the full objective remain open. The next
[single-chain specification](single-chain-auxiliary-spec.md) preserves genuine
per-chain evidence while keeping assay/pair-level requirements strict.

Full regression on immutable application/test source `1d3c96b` passes:
**2,484 passed, four skipped, 1,056 warnings in 370.10 seconds**, four workers.
The log is `/private/tmp/presto-reference-full-regression.log`. Application and
test sources remained unchanged; only saved-output reporting and documentation
were added during the run. Repository pinned Ruff and explicit checks on the
three experiment scripts pass, as does strict MkDocs. This is software
verification, not a reversal of the failed auxiliary comparison screen.

## Genuine single-chain annotations and source admission

The optional baseline import correction is published at `07ce448`; all four
exact-head GitHub checks succeeded. It preserves the failed Weirdo comparison
and frozen sources. Single-chain species annotations are now supported for an
actual MHC groove or TCR chain without invented partners. Pair-level targets and
assays retain their complete-input requirements. Across auxiliary partitions,
an exact chain cannot recur under a changed partner or slot. No parameter shapes
were added or changed.

There are 17 new cases among 154 passing focused checks, including canonical
mixed single/full-input training, checkpoint transfer and feature support. A
direct identical-weight comparison gives exact old/new outputs and diagnostics
for nine complete-pair class-I/class-II/inferred-class annotation, IC50 and TCR
binding requests. Full regression on immutable application/test source `e6cb835`
passes: **2,502 passed, four skipped, 1,056 warnings in 323.68 seconds**, four
workers. The log is `/private/tmp/presto-single-chain-full-regression.log`.
Pinned repository Ruff and strict MkDocs pass. Subsequent data preparation did
not alter those application/test sources.

The [registered MHC reference admission](../../experiments/2026-10-21_0618_codex_mhc-reference-admission/README.md)
then executed on clean `1b218e7`. It supplies 13,218 actual single-groove species
annotations across four exactly mapped taxa, with all 56,440 raw-row dispositions
and 33,696 source-domain aliases independently replayed. There are no conflicting
species sequences. The manifest and generated observations are hash-pinned;
source aliases and actual execution/source snapshots are retained. No pretraining,
prediction, class/pairing target or absent partner was generated.

The frozen component split passes its mechanical count minimum, but one human
B/C component accounts for 91.95% of human test annotations. This concentration
is reported without reseeding/rebalancing; exact-domain/original-chain grouping
does not establish homology independence. Any future model comparison must
account for per-species and per-component performance and the restricted test
domain. Source admission is complete for this bounded contract; calibration,
paired MHC/TCR sources, useful auxiliary transfer and all broader biological
qualification requirements remain open.

Publication closure: `42eefec3a5aaba008fb7cc1740a9c6bae1362017` has four successful
exact-head GitHub checks, including both test/distribution jobs. The next
registered work is full eligible purified IC50 admission with raw bounds,
original-outcome deduplication and preserved historical partitions. Its adapter
and recipe add 45 checks; all 199 focused modular/CLI/docs checks pass. This
preparation keeps independent disagreeing measurements and does not carry over
the 32-row capacity diagnostic's consistency filter. No full-cohort fit has run.
