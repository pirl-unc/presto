# Complete coverage-report PR #66 — 2026-10-20

## Scope and preservation

Finish the current coverage-report PR, including its existing local review fixes
and verification bundle, then summarize the project from its implementation,
canonical contracts, and recorded results. Preserve the later, uncommitted
target-validity, training, and showcase changes in the original checkout.
Do not install or upgrade shared-environment packages, retrain models, launch
cloud experiments, or treat historical metrics as current-model validation.

## Delivery plan

1. Fetch current remote state and inspect PR checks, reviews, and base changes.
   Record the original working-tree/index state. Create an isolated checkout at
   the PR head without stashing or resetting the original work.
2. Bring across only coverage census implementation/tests, the coverage spec,
   its verification scripts/results, and relevant experiment/task/lesson notes.
   Exclude later target-validity edits from the shared coverage test file.
3. Review cache lifetime, class-count SQL, atomic batch failure, and report
   independence. Verify the frozen evidence against source/script hashes and
   rerun the differential software verifier. No new performance study is needed.
4. Run focused tests, full tests, pinned Ruff, and strict docs in the isolated
   source tree. Confirm imports resolve to that checkout. Fix concrete failures
   within this PR's scope and record environmental limitations accurately.
5. Commit the reviewed scope, push to the existing PR branch, update its
   description, and check final-head CI and mergeability. Merge the PR when
   required checks pass. Preserve the original checkout and unrelated changes.
6. Summarize implemented purpose, inference inputs, training data, output
   semantics, measured model quality, and remaining gaps. Explicitly distinguish
   merged code, later local work, implemented outputs, and target design.

## Acceptance evidence

The PR has no unresolved actionable review findings, final-head CI passes, and
GitHub records the merge. The original work remains recoverable and untouched
apart from these planning/result notes. The summary cites repository files and
experiment records and makes no new biological-quality or real-data-speed claim.

## Replan after remote inspection

The only open PR remains #66, but main advanced from `8b5f60f` to `3418d0f`.
GitHub reports merge conflicts. Integrate main in the isolated PR checkout after
capturing the existing coverage fixes, preserving the newer affinity-target
audit and species/assay provenance additions. Check batch rollback against the
new audit tables too. Resolve shared documentation by preserving both histories.
Compare parity with current main as well as retained historical evidence; older
fixtures cannot establish equality of newly added provenance fields by themselves.
