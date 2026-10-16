# Actual human-corpus frozen-input preflight

Status: registered; not yet run. Agent/model: Codex (GPT-6).
[Prospective plan](../agents/codex/plans/human-corpus-preflight.md).

This family characterizes the canonical consumer of the published human-source
baseline, including actual frozen membership and support/leakage audits. Source:
[human-cohort-baseline-v1](https://github.com/pirl-unc/hitlist/releases/tag/human-cohort-baseline-v1),
native manifest `039efc7854098951b5c10b3fb5b38294452eeb425e367ed1fafdeebddf34b38d`.
The full input has 4,071,463 observations with source contributors, citable
references, cell types/conditions, MHC sets versus restrictions, endpoints,
units and censoring. It is the Hitlist 1.76.0 baseline, before #694/#696 fixes;
its unresolved donor/specimen/acquisition identities are retained.

The registered configuration caps selected assay families, freezes 80/10/10
membership before conversion and preserves known reuse components. No model,
optimizer, predictions or performance claim: this is a data/support preflight.
No live assay files, synthetic labels, flanks or auxiliary augmentation. Exact
settings are in reproduce/launch.json. CPU-only Modal, eight CPUs/64 GiB maximum,
one Torch/BLAS thread, 45-minute work cap. Shared datacache retains source,
partitions, complete run outputs and failure receipts. Outcomes and public
hashes must be recorded here and in experiment_log.md after the run.
