# Consolidated Presto foundation

Date: 2026-11-18 UTC. Owner: Codex. User authorization: consolidate the shared
model, workflow, current corpus and installed package in one pull request.

## Outcome

A normal editable or wheel installation exposes the same shared class-I/class-II
model used by the recent presentation work. One supported preparation, training,
evaluation and inference workflow replaces the need to import experiment-local
runtimes. Historical results and source snapshots remain reproducible. The new
data contract names the corrected Hitlist corpus separately from the loader
version, preserves evidence roles, and requires identity-based leakage checks.

## Implementation sequence

1. Inventory the dirty root, isolated implementation and current upstream main.
   Preserve existing work before branch integration; do not delete raw data,
   checkpoints or frozen experiment evidence. Integrate on one clean PR branch
   based on current main, resolving actual differences rather than copying an
   old checkout over current upstream work.
2. Promote the shared model/adapter/trainer changes and compact experiment
   evidence not already upstream. Keep historical readers behind explicit
   compatibility boundaries. Document one current architecture and checkpoint
   decision; the provisional processing checkpoint does not become the default.
3. Provide reusable package commands for protected source preparation,
   checkpoint evaluation and label-free inference, using existing core APIs.
   Generalize repeated orchestration only where it removes an actual dependency
   on an experiment. Keep frozen launchers unchanged and mark historical entry
   points clearly. Package resources/configuration and test wheel behavior.
4. Pin corrected corpus v4 by manifest/content hashes and Hitlist 2.1.12 by
   software version. Use public loaders and audit refreshed output provenance,
   conditions, candidate typing and reported/derived/donor_set distinctions.
   Run admission and complete ancestry/held-out identity protection before any
   training membership. TEST outcomes stay sealed. Reuse downloaded corpus files;
   avoid duplicating multi-GB artifacts on the nearly full local disk.
5. Align current README, architecture, workflow, data-generation and checkpoint
   documentation. Make the development install point to the reviewed source.
   Audit shared OpenVax dependencies, preserve editable/local work, and resolve
   compatible metadata/version issues without forced installs or downgrades.
6. Run focused behavioral tests, full required regression/lint gates, clean wheel
   checks and installed CPU train/evaluate/predict smoke checks for both classes.
   Compare retained-checkpoint predictions across packaging changes. Any actual
   benchmark/training experiment gets the required experiment registry record.
7. Review the complete diff, preserve local work with a recovery receipt, publish
   one PR with concrete validation and remaining scientific limits, and follow CI.
   Do not merge the PR without a user instruction.

## Acceptance checks

- One shared model and trainer behind ordinary installed commands; no source-path
  injection or experiment directory is required by the supported workflow.
- Both-class checkpoint loading/prediction, source context/APC routing, resume and
  class-balanced selection retain tested behavior.
- Corpus software version, corpus generation and exported cohort are separately
  recorded; newer data are not silently assigned old split membership.
- Historical evidence and unrelated local edits survive byte-for-byte where
  untouched; no TEST outcomes are read or promoted into training.
- Tests/package checks are recorded with their exact source commit and explicit
  environmental limitations. The PR includes all necessary source changes and
  compact provenance, excluding raw datasets and model tensors.

## Scope limits

This is consolidation and data admission, not a new architecture sweep or a claim
that isolated gene/drug effects are validated. Keep the retained general-purpose
checkpoint as the default. Do not rewrite completed experiments to use new data.
