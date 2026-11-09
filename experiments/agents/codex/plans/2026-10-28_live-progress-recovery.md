# Keep live progress independent of artifact collection

Observed during the fixed assay-transfer family: local `collect()` blocks Modal
Dict polling while downloading ~226 MB archives. The dashboard retained two
completed controls and stale packaging states while both GPUs had already moved
to the next prescribed conditions. Remote container status/mtime inspection
confirms training is continuing. No fit, dataset, checkpoint or validation/test
selection contract changes are needed.

1. Keep the original Modal launch/controller process alive; stopping it risks
   cancellation of already-spawned calls. Do not relaunch or retry any fit.
2. Add a read-only poller of the twelve existing family Dict keys, capped at four
   concurrent API reads per cycle and a ten-second cadence. Preserve last known
   state on transient errors, record polling errors/timestamps, and stop only when
   all twelve calls are terminal. Observe controller failure receipts too.
3. Write a separate atomic `results/live-workers.json` file; leave the original
   controller's downloaded/completed bookkeeping under its sole ownership.
4. Merge fresh remote worker states into the dashboard while retaining local
   downloaded counts. Display worker poll age and distinguish remotely finished
   fits from locally collected archives. Ignore a transient partially copied JSON
   report and retry on the next page refresh.
5. Restart only the owned loopback dashboard server, verify HTTP/JSON and test
   concealment, and preserve the poller invocation/source hash. No GPU allocation,
   model input, optimizer, tolerance, selected checkpoint or prediction changes.
6. Record the reporting failure/recovery in the experiment and lessons. Future
   collectors should perform network downloads independently of live polling.

Checks: remote and dashboard steps/status agree within a polling interval;
downloaded count remains sourced from the original controller; no test metrics
are disclosed before the existing sealed decision/verification gate.

## Bounded read-only transfer diagnostic

The original large download remained incomplete for over twenty minutes despite
ongoing writes. Its expected225,755,015-byte length is present but checksum fails;
roughly5.5% of bytes are zero, including two almost-empty4MiB regions. The original
collector remains alive and idle on CPU. Test the SDK's ordered `read_file` iterator
for this same already-finished artifact into a separate `alternate-downloads/`
path, with a120-second alarm and mandatory expected length/SHA256 checks. Preserve
any partial and error receipt. Do not replace canonical artifacts, stop the original
controller, restart fits, change tolerances or inspect test predictions. This is
operational transport diagnosis within the existing family, not a new model run.

The ordered API probe timed out after exactly64MiB; the original API still has
missing regions. Next, use a single CPU-only Modal function to read the committed
archive through the mounted filesystem and stream256KiB chunks through function
results. Limit to this family's allowed condition names, require the existing
receipt's full length and SHA256 before accepting any file, and preserve failures.
This allocates one CPU worker and zero additional GPUs. Cache recovered archives
separately. Keep the original training controller alive while any fit is pending.
Only after all twelve remote calls are terminal may collection ownership be handed
over, with the original receipt preserved and no fit retries. Canonical extraction
and report generation must reuse the original collector with a verified local-file
transport adapter, rather than introducing a second prediction/metric path.

## Bounded disk use during canonical collection

Header-only archive inventory finds ~3.20 GB expanded per six-output fit, despite
~226 MB compressed archives. With ~14.4 GB local free space, retaining all twelve
expanded directories would exhaust the shared disk. After each original collector
call, verify byte-for-byte SHA256 equality between the archive member and extracted
file for the six allowlisted redundant raw exports: `raw/evaluations/{train,
validation,test}.json` and `raw/{train,validation,test}-predictions.jsonl`. Preserve
a per-file receipt before unlinking only those verified local expansions. Keep the
original checksum-verified local archive, remote archive, all checkpoints, gzip
prediction dumps, summaries and diagnostics. Do not parse heldout contents or
expose scores while verifying bytes. Every removed expansion is reconstructable
exactly from its archive; no source observation or unique artifact is removed.

Before each extraction, calculate expanded bytes from archive headers and require
enough free space for the original collector's extraction/copies plus a 2 GiB
reserve. Keep initial/selected/resume checkpoints and both canonical/archived gzip
prediction dumps available. All downstream verifier, comparison and replay inputs
have been inspected: local analyses consume canonical gzip files and selected.pt;
fresh-GPU replay consumes original unchanged remote artifacts. Record compacted
bytes and authoritative archive paths in the handoff/README. This changes storage
only, never scientific membership, predictions, tolerances or model selection.

The final ownership handoff may be automated by a task-local watcher. Pin the
original collector PID and its exact `ps` start-time/command identity at watcher
launch, verify that identity again immediately before a normal SIGTERM, and call
`--prepare` first. Stop no process unless all twelve original calls have returned
and their archives pass the original hashes. If the original collector completes
independently, exit without intervention. Otherwise wait for its verified exit,
then call `--finish`; never start another closure watcher or any training fit.

## Completion/receipt race recovery

The CPU watcher stopped on `KeyError('artifact')` after five cached archives. The
worker writes `status=completed` before its parent packages/commits the archive;
one snapshot can expose that status without the final artifact receipt. Preserve
the failed watcher log/invocation/source and its five verified archives. Require
both completed status and an artifact receipt before readback or successful poller
termination. Restart only the read-only poller and CPU archive watcher, recording
separate recovery invocations and rechecking every existing cache hash. Do not stop
the original training controller, any GPU worker, or the guarded handoff/closure
watchers. Their original terminal-call/hash gates remain unchanged.

## Local-client disconnect recovery

At 06:56 UTC the reader log explicitly reports a local-client disconnect and
stopped non-detached CPU app. All eight cached archives remain verified; the next
read attempt produced zero bytes and failed before accepting an archive. Preserve
that failed receipt and empty partial under distinct recovery paths. Restart only
the CPU reader with `modal run --detach`, a new attempt receipt, and revalidation
of existing caches. The original GPU app was already detached and continues its
same twelve original calls. No refit, altered tolerance or scientific change.

Before closure starts, restart only the still-waiting local closure watcher to
pass `--detach` to the one read-only H100 replay. Preserve its original source,
confirm no decision/verification log exists and collection is incomplete, verify
the exact owned PID, and stop that watcher before starting its replacement.
Selection, independent metric checks, replay inputs and tolerances are unchanged.
