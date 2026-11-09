# Newer-code unified parity-data retrain

Date: 2026-11-08
Agent/model: Codex / GPT-5
Status: corrected common-loader preflight passed and independently closed;
no optimizer launched yet

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

## Corrected v2 preflight result

The corrected projection adds canonical measurement lineage and complete Stage B
development embargoes. Primary call `fc-01M6YSQBZ1AKY6WW2NN1ZBQD9N` under app
`ap-dBDyzedkYeSHH7SCN0EKLT` hash-copied all six sources and the Stage A parent,
then passed both complete canonical manifests:

| Condition | TRAIN | development | total | loader seconds |
|---|---:|---:|---:|---:|
| reference | 1,509,704 | 51,066 | 1,560,770 | 493.59 |
| parity data | 1,652,917 | 51,066 | 1,703,983 | 507.49 |

The primary used 8 CPU / 32 GiB / 512 GiB scratch, no GPU, 1,010.20 seconds,
and 4,817,805,312 bytes peak RSS. Receipt SHA-256 is
`6efe6b3f64cbff080f9b03b9dce0ca1906a008d487fee3bd7d9eb8532ac6aaa2`.
Returning the full Python object exceeded Modal's inline threshold and attempted
a forbidden blob upload; the already committed receipt was recovered directly
from the volume. No scientific work was rerun for that transport failure.

The first independent call `fc-01M6YTYJTA4Z33FBNACMH886HY` was preempted after
reference completion and parity-manifest creation. Its incomplete directory has
no receipt and remains preserved. Atomic retry-safe call
`fc-01M6YWM3TV4JF1CT6HACAQW0M7` under app
`ap-FMlOBE4ADcMdKxRwETVd70` completed both manifests with the same row counts,
partitions, manifest hashes, source hashes, and condition-level pass status.
Runtime was 1,623.50 seconds and peak RSS 4,823,691,264 bytes.

Its raw in-process comparison marked the receipt failed at `reference/model`
because it compared a live Python structure containing tuple/list distinctions
against the JSON-loaded primary receipt. The persisted model, training, scales,
census, admission, row, partition, and manifest values are canonical-JSON
identical. The failed raw receipt remains immutable at SHA-256
`fdce01321abedf12f0e3aa961f0bd8ab93bade1896a101bc65530d6015178326`.

CPU-only closure call `fc-01M6YYDWEEHT574AQTDNNQCRYT` under app
`ap-KMHe8cOf1BqdxQ7c7acXTL` pinned both full receipts and canonical-JSON compared
every preregistered scientific field plus every source-copy byte count/hash. It
passed in 0.80 seconds; closure receipt SHA-256 is
`b60d3f9343808b23fa67a6f636df0be01699f59475f7af596701c1ac00121c16`.

Across primary and independent work: TEST and locked confirmation access are
false; predictions are false; optimizer steps are zero; requested/observed GPU
are null. The corrected H100 training gate is therefore open. This remains a
development comparison and does not create independent superiority evidence.
