# Corrected human corpus preflight

Agent/model: Codex (GPT-6). Prospective data-only qualification of the corrected
[Hitlist human cohort](https://github.com/pirl-unc/hitlist/releases/tag/human-cohort-corrected-v1),
native manifest `b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372`.
The fixed contract is in [the plan](../agents/codex/plans/corrected-human-preflight.md).

Released Presto **0.1.13** and Hitlist **1.78.0** are the installed producers.
This recipe PR versions the repository as 0.1.14; that is not the experimental
producer. [Producer pins](reproduce/producer.json) bind the public wheel and main
commit. The launcher refuses dirty recipes and records the exact recipe commit.

The source contains 4,071,463 human-source/HLA observations. Both MHC classes are
eligible. The new cohort uses explicit reservoir selection (binding/elution caps
5,000 each, kinetics/stability 1,000 each, seed 42) and relative_deficit_v2 whole
peptide-component 80/10/10 allocation with split seed 42. Original baseline inputs,
membership and results remain immutable; this changes both source and policy, so
it is not a causal comparison of individual corrections. Native assay endpoints,
units, censor bounds, source IDs and contributor lineage remain preserved. Strict
MHC sequence resolution is enabled. Only the exact existing mhcseqs 2.6.13 catalog
is reused through its unchanged datacache recipe and explicit four-file hashes;
no prior cohort or partition assignment is reused. All shared products use datacache.

No pretraining, fitting, model constructor, optimizer, predictions or performance
metrics are allowed in this family. Train/validation/test membership and full output
evidence coverage will be measured. All augmentation/generated-negative ratios and
regularization weights are zero. Positive elution has no observed negative class;
unknown origins, donors, specimens and acquisitions do not establish independence.

[candidate-support.json](reproduce/candidate-support.json) fixes a prospective IC50
requirement before launch: direct binding:IC50 evidence from IEDB/CEDAR, at least
50 observations, 50 distinct peptides and 50 exact observations in every split.
The canonical preflight remains undeclared-output reporting. A later manifest may
bind these fixed claims to actual membership, but it is not an executed gate or a
trained-output claim. Thresholds will not be relaxed after seeing counts.

Reproduce from the recorded clean recipe commit with
`bash experiments/2026-10-16_0750_codex_corrected-human-preflight/reproduce/launch.sh`
from the repository root. The launcher
fetches the released producer through datacache. CPU: eight, RAM: 64 GiB, scratch:
512 GiB, GPU: explicitly none, Torch/BLAS threads: one, loader workers: zero,
function/work limits: 50/45 minutes. Runtime, runner-only RSS, all logs, pip check,
versions, catalog, membership, funnel, coverage and leakage evidence are retained.

A first local prelaunch used the tooling interpreter, which lacked filelock. The
wheel fetch failed; the hash guard then rejected the empty path before a remote
app ran. The unchanged log is [preserved](results/prelaunch-failure.log). #107 fixes
the shell wrapper to propagate a fetch failure before invoking Modal. The actual
retry uses the shared-environment interpreter; no shared packages were modified.

Status: corrected recipe frozen before remote launch; no experiment results yet.
Next Step: execute, recover, independently verify and publish the complete bundle.
Open Questions: whether the fixed IC50 support requirement passes and how much
experimental-origin independence remains unresolved. #678/#680 and #48/#53 stay open.
