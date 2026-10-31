# Shared input iteration performance

Profile the per-row open/stat/seek overhead observed during the temporal model
startup. Existing H100 fits remain on their pinned runtime. This family runs no
model, loss, optimizer, validation or test inference.

Use the first 5,000 TRAIN records from each of the corrected physical-assay,
observed-MS and generated-background files. Retain exact source hashes, sampled
bytes/hashes and capture command. No held-out rows. This deterministic convenience
sample benchmarks reader behavior, not biological accuracy or full-cohort runtime.

Compare the original indexed iteration with a shared sequential iterator retaining
one file handle for each contiguous source segment. Preserve every row's source
stat check, byte-offset semantics, evidence normalization, content, order and
random-access API. There is no cached preflight result or bypassed admission.

Before promoting: compare every normalized observation exactly; test source
mutation/replacement during iteration, multi-file order, independent iterators and
explicit iterator closure; demonstrate equivalent common-training checkpoints and
resume behavior. Benchmark three paired repeats, alternating condition order, on
the same local CPU/files and record Python/platform/CPU/memory. Measure iteration
without model work; count identical observation IDs and compare full content
outside timing. Promote only with all integrity tests and >=1.10x median throughput
improvement on this declared sample. Otherwise retain the result without claiming
a win. Freeze original and candidate source hashes and required reproduction files.

Close with runtime tables, source/mutation/parity checks, README and canonical log.
Predictive validation/test metrics and dumps are inapplicable: no predictor runs.

Capture transport correction: the first large stdout transfer was incomplete
(gzip EOF). Preserve it without admitting any rows. Save the same TRAIN prefixes
to an ephemeral worker archive, return its size/hash, then read bounded 128 KiB
chunks and require the final archive hash before extraction. This touches no
training input/output or durable volume state.

Second transport correction: even a bounded chunk was truncated. Inspection of
installed Modal 1.6.1 shows CLI exec waits for process exit while STDOUT printing
runs in background tasks. Use the same SDK command-router operation with explicit
PIPE streams, await complete stdout/stderr and exit status, then enforce archive
size/hash. No shared environment changes; freeze the exact wrapper and SDK version.

The first measured comparison passes (1.2812x median throughput). Before closure,
fix the replay launcher to use its frozen baseline source rather than whatever
HEAD contains after promotion; require the candidate source hash. Preserve the
initial measurements and repeat once with the corrected reproducibility wrapper.
This is a technical replay check, not selection among biological outcomes.
