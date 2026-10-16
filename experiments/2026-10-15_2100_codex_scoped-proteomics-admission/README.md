# Published scoped proteomics admission audit

Agent: Codex / GPT-6. Registered 2026-10-15 America/New_York.
Plan: [admission contract](../agents/codex/plans/2026-10-15_scoped-proteomics-admission.md).

This checks the actual public PXD013455 HeLa Trypsin/P archive through the new
adapter and canonical dataset/collation/support audit. It does not fit a model,
assign partitions or evaluate held-out performance. No validation/test metrics or
prediction dumps are expected; those require the next canonical trainer work.

Input manifest SHA-256:
`441f3c66c6e6bc6a90c7cb8c54d0131676c3cb4e7a04ddbd9346061ad8f9d84b`.
Select 10,000 unique peptides by reservoir seed 42 after complete occurrence
collapse. Direct MS/MS, matching and scoped absence are distinct proxy families.
No excision, biological replicate or depth labels. Multiple protein occurrences
mask flanks/parent while preserving a complete sidecar lookup. Citations,
cell/assay/search/reference conditions and unresolved identities remain pinned.

CPU only; one Torch/OpenMP thread; 15-minute wall limit. No GPU, pretraining,
synthetic data or loss optimization. Source code is committed before launch;
`reproduce/launch.json` captures its commit, paths and environment overrides.
All reusable audit data are published to shared datacache; results here contain
small summaries and hashes/locations, not duplicate data tables.

## Results

Pending. Runtime, peak RSS, full/admitted support and source/candidate linkage
will be recorded in `results/summary.json`. Artifact hashes and paths will be in
`results/artifacts.json`; closure will update this README and the canonical log.
No model baseline promotion or broad training-readiness claim is intended.

Status: planned, not launched.
Next Step: run after the final adapter source-lineage regression passes.
Open Questions: actual mapping ambiguity and label support after one-vote collapse.
