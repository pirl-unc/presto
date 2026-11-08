# Newer-code unified parity-data retrain

Date: 2026-11-08
Agent/model: Codex / GPT-5
Status: primary preflight stopped on BA lineage defect; correction specified;
no optimizer launched

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

## Primary preflight stop

Frozen call `fc-01M6YPYAEWMTAAMWZ6C7XBCJRM` used 8 CPU / 32 GiB, no GPU,
and blocked network access. The reference condition passed the complete common
loader on 1,560,770 rows: 1,509,704 TRAIN and 51,066 development, zero TEST.
Its pass took 1,227.21 seconds. The parity condition then stopped after 350.73
seconds at byte zero of the first BA file because the projected row lacked the
canonical original-measurement identity required for cross-source duplicate
detection. Peak RSS was 4,729,880,576 bytes; no optimizer or prediction ran.

This failed gate is preserved, not bypassed. The correction is preregistered in
[`2026-11-08_ba-projection-lineage-continuation.md`](../agents/codex/plans/2026-11-08_ba-projection-lineage-continuation.md): add deterministic archive-native
measurement lineage, extend the BA embargo to the complete Stage B development
union, write immutable v2 artifacts, and rerun the full preflight before GPU use.

Large inputs, checkpoints, and predictions remain on the `presto-checkpoints`
Modal volume. Git contains the exact manifest template, launcher, receipts,
summaries, evaluation code, and hashes only.
