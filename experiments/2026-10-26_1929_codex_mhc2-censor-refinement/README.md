# Fixed-budget MHC-II censor refinement

Status: prospectively registered; software checks passed, preparation and fits pending.
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

No new outcome is available yet. Historical validation/test sets are development
data, and this bounded comparison cannot independently qualify MHC-II prediction.
Full binding/MS/pathway improvement remains the objective.

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
  bootstrap draws are specified in the plan; no outcomes are available yet.
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
