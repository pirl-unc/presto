# First implementation review

Superseded for completion status by the
[plan-conformance audit](plan-conformance-review.md). The neural foundation below
is implemented, but milestone A remains partial: source preparation and feature-
level support/closure guarantees are incomplete. No real-data quality gate has
passed. Exact published-head PR and push CI subsequently succeeded.

- Planning authority: issue #130 and its frozen `issue130.md` copy/receipt.
- Implementation PR: https://github.com/pirl-unc/presto/pull/131
- Base: `ee4ad60f9750996a4a8e0ecda73e84f583085dd1`.
- Initial implementation: `89afccb9aefb539f5efe9916492951c0890e18d4`.
- Final Python source after allocation review:
  `77d9856b15abd25fa1e6f963585cbc42e446c81e`.
- Worktree: isolated branch `codex/modular-presto-model-20261020`.

## Verified software behavior

The model allocates only declared endpoint families, uses explicit class-II
registers, keeps identity supervision separate, preserves roles/overrides,
retains concurrent machinery states and separates digestion from shared MS
response. Raw targets retain units and censor direction. One bag supplies one
loss. Public prediction returns units/support and refuses untrained paths.

Review found two distinct single-core cases worth guarding: a previously allocated
attention layer with no gradient evidence must not silently score longer peptides;
a fresh run on a single-core-only class-II subset should omit that layer entirely.
Both behaviors have regressions. Preflight also rejects unsupported assay classes.

The two canonical CLI entry points use the same normalized manifest pipeline.
Software fixtures exercise preparation, split/hash checks, training, optimizer
updates, checkpoint round trips, source/subset extension, inference, candidate
ranking and experiment publication/closure into a temporary test registry.

## Validation evidence

- Complete parallel local regression: **2,418 passed, four skipped**, 903 warnings,
  291.12 seconds. Command: `TEST_SH_MAX=2 ./test.sh -q` using the existing shared
  Python environment, an isolated import path, and OMP/OPENBLAS/MKL threads set to 1.
- Final allocation amendment: **49 focused tests passed** across
  `test_modular_model.py`, `test_modular_training.py`, and `test_docs_match_code.py`.
- Repository-wide Ruff **0.16.0** lint and format check pass (317 Python files).
- Strict MkDocs build passes.
- Full remote checks for the final published head are recorded on PR #131.
  This review does not assert a remote result before it is available.

These are software checks. The synthetic integration fixtures are not experiments
showing biological predictive quality. No new real-data training job was launched.

## Completion boundary and next work

This is the software foundation for milestone A, not completion of issue #130.
The canonical guide is `docs/modular_model.md`.

Next: pin/qualify the actual source cohort and identity catalog; preregister the
32-example fitting diagnostic; qualify held-out IC50 against matched mean/ridge
baselines; then admit additional measured endpoints under the issue's gates.
Matched APC cohorts, independent MS anchors and transfer evaluations remain data
and scientific qualification tasks. The TCRvdb author file remains unavailable.

Quantitative MS, PTMs/noncanonical or short class-II cores, additional reaction or
stimulus features, exact resume/arbitrary checkpoint migrations and automatic
preparation of every Hitlist source remain explicitly unsupported. The original
model/checkpoint path remains the default. No calibration, causal component
identification, or improved generalization is claimed for the new model yet.
