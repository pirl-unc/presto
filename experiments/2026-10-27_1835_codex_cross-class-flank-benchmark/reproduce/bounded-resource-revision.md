# Explicit resource-protocol revision before any optimizer update

The original6GiB startup gate was an unmeasured conservative estimate. Initial
admission/profile attempts stopped before optimization. A subsequent waiter for
8GiB sustained headroom was manually interrupted after about10minutes of variable
available memory, with zero profile/fit launches. Preserve all prior receipts,
controller snapshots, invocation and source files. This is an explicit revision,
not a claim that the earlier no-lower-floor continuation succeeded.

A bounded profiling-first contract now uses5.5GiB worker startup available RAM,
unchanged3GiB running available-RAM reserve, and a TIGHTER2GiB process RSS cap
(previously4GiB). Disk floors15/10GiB, one-hour limit, CPU float32, four threads and
one interop stay unchanged. After imports, require15seconds of available RAM above
5.5GiB, with at most5minutes waiting per worker. This is admission waiting, not a
retry of an already-started profile/fit. One worker at a time.

Run the same prespecified three warmups/five timed training-only steps on all three
pinned model versions. Require each profile to complete within its2GiB RSS cap
before starting the original ten fits. Each fit uses the same tighter guard.
Any started-worker failure stops the controller, with no automatic retry.
No held-out scores or biological fit outcomes informed this resource revision.
No model/data/seed/batch/loss/optimizer/step-budget change. All original code and
runtime pins remain intact; the separate continuation launcher freezes and records
the resource override explicitly in every resource receipt.

This avoids treating an arbitrary estimated startup floor as an indefinite blocker
while retaining runtime enforcement and reducing the process's maximum footprint.
