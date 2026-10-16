# Cancel superseded feature validation while preserving release evidence — #120

## Problem and scope

Feature pushes and pull requests currently retain superseded CI runs, wasting
runner capacity alongside current validation. Releases require the distributions
from successful checks on the exact clean-main commit. An ordinary shared main
concurrency key is unsafe even with cancel-in-progress false: a new pending run
can displace an older pending run. GitHub documents that behavior at
https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency.

## Design

Use workflow-level concurrency with separate PR-number and feature-ref groups.
Include event and workflow identity so feature pushes and PR validation cannot
cancel one another. Give main, immutable tags and other events a unique run-ID
suffix, protecting running and pending exact-source validation. Enable cancellation
only for pull_request and non-main branch push events. Keep existing triggers,
checks, dependencies and release artifacts unchanged. Version 0.1.20.

## Verification and release

Review an event matrix: repeated updates to one PR or one feature branch share a
cancellable group; different PRs/branches differ; PR and push differ; separate main
runs and tag runs always differ and cannot cancel one another. Validate workflow
syntax, format/lint and full CI. After 0.1.19 distribution is verified, merge only
with successful final-head checks, then distribute exact successful clean-main CI
artifacts and independently verify public release hashes and the wheel.

## Reviewed event matrix

| Event | Group identity after workflow/event prefix | Cancel active predecessor |
| --- | --- | --- |
| PR #121, any update | pr-121 | yes |
| PR #122 | pr-122 | yes, only #122 |
| Feature push fix/example | refs/heads/fix/example | yes, same feature push only |
| Different feature push | its own full ref | yes, its own group only |
| Main push, run A | run-A | no |
| Main push, run B | run-B | no |
| Immutable tag, run C | run-C | no |
| Other event, run D | run-D | no |

Final PR and feature-push checks both gate merge. Distribution uses the exact
successful clean-main push artifact after merge; unique protected groups retain
both pending and running release evidence. No trigger or job is removed.
