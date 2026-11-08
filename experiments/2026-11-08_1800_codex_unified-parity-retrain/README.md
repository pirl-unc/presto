# Newer-code unified parity-data retrain

Date: 2026-11-08
Agent/model: Codex / GPT-5
Status: common-loader preflight implementation in progress; no optimizer launched

Specification:
[`2026-11-08_unified-parity-retrain.md`](../agents/codex/plans/2026-11-08_unified-parity-retrain.md).

This family restarts two matched Stage B fits from the same frozen Stage A
checkpoint. The reference condition uses the four original
affinity/presentation sources. The parity condition differs only by adding the
verified NetMHCpan 4.2 and NetMHCIIpan 4.3 BA TRAIN projections as an
unspecified-assay IC50 output.

No TEST, locked confirmation, T-cell, PRIME, or model-mechanism change belongs
to this first comparison. The full canonical common-loader preflight must pass
and be independently verified before either requested `H100!` optimizer starts.

Large inputs, checkpoints, and predictions remain on the `presto-checkpoints`
Modal volume. Git contains the exact manifest template, launcher, receipts,
summaries, evaluation code, and hashes only.
