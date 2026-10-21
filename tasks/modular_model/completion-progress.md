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
