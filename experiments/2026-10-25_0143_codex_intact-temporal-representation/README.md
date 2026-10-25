# Intact temporal representation development

Registered before execution; not launched because current free space is below the
unchanged1GiB+32MiBstartup reserve. [Full plan](reproduce/plan.md) and
[exact source/git/environment contract](reproduce/launch.json).

This is nested development CV within previously inspected original training data,
not a new untouched biological test or APC-concentration validation.

All nine Python modules now pass isolated Ruff E9/F and AST parsing, with the
four original input hashes and frozen production model bytes unchanged.
[Initial static review](reproduce/static-review.json) and the subsequent
[launcher-provenance review](reproduce/static-review-final.json) record exact code hashes.
No prepare, qualification, fit, evaluation, replay or closure stage has launched;
static checks do not establish numerical or predictive correctness.

The runner includes seven arms, fold-local inner selection, all45physical starts,
per-example inner/outer predictions, independent source/feature/objective/gradient
and metric/bootstrap replay, and automatic scientific report/table/manifest closure.
The initial report above remains the current status until those stages run.
Stage receipts will record the actual execution revision, tracked dirty state and
package versions; the registration revision remains distinct. Versions must match
across stages. These checks add no biological conditions or outcome access.

At02:12UTC, an attempted `prepare` launch was refused by the unchanged startup
guard before creating a stage receipt or starting a subprocess. A preceding free-
space reading exceeded the threshold; capacity fell again before launch.
[Preserved refusal](reproduce/prepare-preflight-refusal-01.json). No model/data
stage ran, and the threshold was not lowered.
