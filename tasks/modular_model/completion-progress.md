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

Still pending: the early-supervision extensions, cell/cytokine/profile machinery,
quantitative routes, actual source adapters/cohorts, continuation/freezing/resume,
scientific evaluation gates and registered fitting/transfer evidence, plus final
PR/CI validation. TCRvdb access remains unverified/unavailable from prior work.
