# Resource continuation, before any predictive fit

First controller stopped before the first training-only profile: imported worker
available RAM was below the frozen6GiB floor. Zero optimizer steps, zero holdout
scores. Preserve controller-attempt1.json, original receipts/logs/source pins and
launch.json. No change to model, data, numerical settings, step budget or limits.

A manually reviewed continuation may start only after available RAM stays at least
8GiB for60seconds (5second samples), at most20minutes waiting. This extra buffer
allows import-time fluctuations; it does not relax the6GiB worker startup or3GiB
running floor. Stop without launching if stable headroom never arrives. Run one
fresh first-profile attempt with a distinct receipt/log, then untouched remaining
profiles and the ten original condition workers. No automatic retries of a failed
profile or fit. All original source/code hashes are checked by each worker.

Freeze the continuation source and invocation separately. The original failed
controller JSON is copied byte-for-byte to an immutable attempt1 receipt; the live
controller path may then reflect the current attempt. Closure includes both.
