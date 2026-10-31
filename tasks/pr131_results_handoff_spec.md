# PR #131 results and future-work handoff — 2026-10-30

## Scope and constraints

Commit all pending durable changes in the active shared-training checkout to
the existing PR head `codex/modular-presto-model-20261020`. Include completed
curation and direct-decoder reports, independent metric audits, frozen analysis
source, lessons, ideas and a future-work handoff. This is publication of existing
work, not a new scientific experiment or model promotion. Keep PR #131 draft and
preserve scientific qualification requirements #130/#132.

The older root checkout contains separate historical work; do not transplant its
old architecture over this branch. Local process locks are operational state,
not reproducibility evidence. Retain large raw prediction/checkpoint files in
their documented artifact locations and publish hashes and retrieval references.

## Review and publication steps

1. Inspect every pending file's type, size and purpose; validate machine-readable
   reports and source snapshots. Check that final results supersede live status
   without erasing historical attempts or intermediate observations.
2. Preserve findings about aggregate versus within-assay correlation, poor
   censor satisfaction, weak stability, source coverage, synthetic negatives,
   short training exposure and limitations of external-tool comparisons.
3. Search existing Presto/Hitlist issues. Create narrowly scoped missing follow-ups
   with evidence, acceptance criteria and links; reuse existing source-library
   issues for known upstream semantics rather than duplicating them.
4. Update task state, experiment records and PR description with measured
   outcomes, validation and limits. Do not present a selected checkpoint as
   biologically qualified or unrun proposals as improvements.
5. Run applicable artifact/documentation/lint checks. No runtime changes are
   intended, so reuse the recorded passing full suite unless a new code change
   or unresolved failure justifies rerunning it.
6. Stage the reviewed durable files, inspect the staged diff, commit, push without
   force to the existing PR head and verify the remote commit and draft status.

## Completion evidence

Record the commit, issue URLs, verification results and remaining scientific
limitations in `tasks/todo.md` and the PR. The branch must contain all reviewed
durable results; remaining ignored files must be operational state or documented
raw artifacts, not unpublished source or scientific conclusions.

CSV publication detail: generated CSVs retain standard CRLF line endings. Check
staged whitespace with `cr-at-eol` enabled alongside the normal whitespace rules,
rather than rewriting frozen evidence to satisfy a line-ending assumption.
