# Explicit continuation after preflight RAM interruption

2026-10-27, Codex / GPT-6. Streaming controller85403 terminated exit1. Native101
evaluation completed with zero added updates and all original artifact hashes
preserved. Native102 failed after19.736seconds, peakRSS290,471,936bytes, on the
10GiB available-RAM floor while enumerating training rows, before initial
checkpoint/training. Only admission.json, resolved-config.json and the empty
`.training.lock` were written. The first preservation helper correctly refused
an incomplete two-file inventory before making changes; preserve and hash the
empty lock too. The terminal tool session establishes that no writer remains.
Startup available RAM was16,984,719,360bytes. No102prediction was produced.

Preserve the terminal controller, receipt, log and these two files. Hash and move
the incomplete preflight directory to `native102-preflight-attempt01`; never erase
or reuse it as a completed fit. The frozen native102 seed/source/init/optimizer
recipe can then execute afresh at its original intended destination. No numerical
source, guard floor, test selection or model budget changes. Original native101
evaluation is not repeated. All subsequent actions stop on their first failure.

The OS snapshot after failure has24,763,006,976available bytes and no unusually
large additional experiment process; this does not establish a sustained reserve
or explain the earlier transient. Do not stop user applications or older jobs.
Before each remaining action require at least20GiB available RAM sustained for
60seconds, with5second samples, within a30minute admission deadline. This adds a
stricter admission check; action-time limits remain unchanged. Save every admission
sample and launch in a new controller state. This admission wait is not an action
retry and does not restart a failed fit automatically.

Continue once in original order: fit/score native102, fit/score permuted101,
logistic features/fit. Pin the new wrapper and all original recovery source hashes.
Independent closure requires successful native101 recovery plus successful new
remaining-arms terminal state; it must not require rewriting the failed earlier
controller to success. Preserve full state-chain lineage in comparison receipts.

The independent comparison implementation remains unread against real outcomes;
its artificial tests and frozen prospective gates are unaffected. The wider
binding/MS/pathway goal remains active.
