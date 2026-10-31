# Fixed-budget MHC-II censor refinement

Status: exact-only continuation completed; censor arm interrupted by RAM guard.
Codex / GPT-6, 2026-10-26.

[Plan](../agents/codex/plans/2026-10-26_mhc2-censor-refinement.md).
Compare exact-only continuation with inclusion of all existing censor bounds,
starting from the identical verified seed42 checkpoint. Each condition uses
1,476canonical optimizer updates; the prespecified primary comparison uses the
final checkpoint, while canonical validation-selected outputs remain secondary.
Source values, original partitions and inherited target scaling remain unchanged.

Local CPU/two numerical threads/one interop. The two new conditions are serial;
the older four-thread family may run concurrently. A separate resource guard
protects its memory reserve. No cloud/GPU or package installation.

Exact-only fixed-final predictions are saved; the paired comparison is unfinished.
Historical validation/test sets are development data, and this bounded comparison
cannot independently qualify MHC-II prediction.
Full binding/MS/pathway improvement remains the objective.

Preparation completed in13.695seconds, peakRSS265,420,800bytes, from clean
commit `37edb56560a65651da63b2d9ef95fa3fc014b1ac`. Both manifests pass canonical
preflight and have the same original holdout files. All33parent state tensors
are hash-pinned; inherited IC50 scale has mean2.768232837778882,
SD1.3434766242663534 and21,636exact training measurements. The preparation
receipt began clean; `launch.json` additionally records the generated untracked
manifests/source snapshots. The controller proceeds serially through both fits,
fixed-final scoring and independent comparison, stopping on any failure.

## Frozen comparison

- Sources: the original human direct IC50 cohort at
  `/Users/iskander/code/presto/.worktrees/presto-modular-model/data/modular_cohorts/2026-10-21-full-ic50/`.
  Exact training21,636 versus the same observations plus1,919reported censor bounds.
  Both retain validation2,925/test2,803, their original source values, qualifiers,
  complete MHC pairs and peptide-disjoint partitions. Shared studies and historical
  outcome inspection prevent an independent-test claim.
- Common initializer: old convergence seed42 selected epoch26, SHA256
  `db2585e63489cb5aac8c7a32fd7e522bcbc51035846c05f8cd2794c4449e3dc5`.
  The parent trained41epochs on the exact cohort. Its target scaler is inherited;
  adding bounds must not refit it. All parameters remain trainable.
- Native canonical trainer; seed101, AdamW1e-4/weight decay1e-4, batch64,
  dropout.1,1,476updates. The exact condition has four full epochs plus120batches;
  the censor condition has four full epochs. No new synthetic labels or auxiliary
  losses. IC50 maps to its native log10(nM) output with standardized squared
  error for exact measurements and one-sided squared error for source bounds.
- Primary endpoint: fixed final checkpoint. Canonical validation-selected
  checkpoint outputs remain secondary because their selection opportunities differ.
  Prospective class-II improvement/class-I retention gates and2,000paired peptide
  bootstrap draws are specified in the plan; paired outcomes remain unavailable.
- Raw checkpoints, histories, secondary predictions and primary predictions:
  `/private/tmp/presto-mhc2-20261026/censor-refinement/`.
  Portable primary prediction dumps and independently computed metrics will be
  written under `results/`. `reproduce/launch.json` will pin all source/runtime
  hashes, inherited tensors/scalers, git state and environment before fitting.

## Software verification

Four new artificial-fixture tests cover bound/source replay, metric gate failures,
controller failure/reentry and actual canonical fixed-update continuation with
different cohort sizes. Together with continuation, training and resume tests,
**46tests pass** in15.23seconds. Pinned Ruff lint/format and shell syntax pass.

An existing continuation test failed both in the combined run and in isolation
under two Torch threads: its trainable parent and frozen model used different
attention kernels. The test now checks every retained tensor exactly and compares
outputs under the same no-grad inference mode, still requiring bitwise equality.
Both failures and the passing run are preserved in `results/software/`. This
test-only correction changes no model, training recipe or scientific threshold.

## Preserved interruption and recovery

The exact arm completed1476 updates and fixed-final scoring at20:03:54UTC.
The original controller then stopped on the censor arm's10GiB available-RAM
floor after126.502seconds, peakRSS2,776,088,576bytes. Its durable snapshot is
epoch1/update100, position6400, with optimizer/RNG state preserved and copied
under the hash in `reproduce/interruption01.json`.

Recovery attempt02 stopped before updates because its environment omitted the
original `PYTORCH_ENABLE_MPS_FALLBACK=0`. A recursive full-contract comparison
found that sole difference; no guard was bypassed. Attempt03 restored the exact
environment and verified complete execution-contract equality before launch.
It stopped again at the original available-RAM floor after53.764seconds,
peakRSS2,237,251,584bytes. Attempt02 took7.269seconds/257,327,104bytes.
All controllers are terminal and their failure states, code, logs and receipts
remain separate. The newest durable snapshot advanced to150updates/9,600exposures and is copied
in `reproduce/interruption02.json`; the original100-step copy remains intact.
No automatic retry is scheduled. Wait for demonstrably improved
headroom before a separately registered recovery, without lowering any reserve.

The final scientific budget remains1476 logical updates per arm. Rolled-back
work can repeat between50-step snapshots; report that extra executed compute
and every attempt runtime at closure. No new metric result or preferred arm
is claimed from this incomplete comparison.
