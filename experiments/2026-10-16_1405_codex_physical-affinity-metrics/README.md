# Physical affinity metric audit

Planned data-only replay of the immutable effective-target fit, by Codex / GPT-6.
No model fitting, new predictions, checkpoint selection or criterion changes.
The specification is `tasks/physical-affinity-metrics-spec.md`.

Inputs are the public run archive and inventory from
https://github.com/pirl-unc/presto/releases/tag/effective-target-fit-qualification-v1.
Audit the same 2,614 source-linked IC50 observations, every original predictor,
constant-vector correlations, physical 500 nM classes and canonical overall/
mapping threshold reports. Identify any changed observation IDs and preserve
all original raw/effective values, operators, source context and partitions.
Require exact-regression results and the failed held-out quality decision to
remain unchanged. Publish an additive datacache generation and independent hash
verification; do not overwrite the historical run or its public reports.

Run `reproduce/launch.sh` from an isolated environment with this exact checkout
installed. The audit requires a clean source commit and records actual package,
code, dependency, input and output pins. CPU only; no cloud or GPU required.
