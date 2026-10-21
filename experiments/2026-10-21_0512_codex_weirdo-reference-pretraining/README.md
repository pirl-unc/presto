# Peptide-source reference pretraining

Registered before preparation, 2026-10-21 UTC. Agent: Codex / GPT-6.
Status: amended reference admission passed; fixed fitting comparison pending.
No model fitted or test predictions inspected.

The [prospective plan](../agents/codex/plans/weirdo-reference-pretraining.md)
fixes source hashes, label-blind 5,000/1,000/1,000 membership, independent
reference audit, prior/composition/network conditions, CPU budget, validation
screen and interpretation limits. The network uses the shared modular encoder
and ten independent source-membership outputs. This auxiliary-only family does
not supply assay, paired-chain, species-of-origin or immune-function labels.

Preparation code is `code/prepare.py`. Actual invocation, git state and package
versions will be in `reproduce/preparation-launch.json`; source snapshot in
`reproduce/source/`. Selected rows and audit summaries will be under `results/`;
the full occurrence audit and later model artifacts remain under the linked
repository-level `artifacts/` directory. No test split is omitted: all three
source partitions are prospectively declared. Predictions will be generated
only after successful reference admission and checkpoint selection.

Completion requires full admission, executed conditions, held-out prediction
dumps and metrics, actual runtime/support, fresh-process training-batch replay
and canonical-log closure. A passing source audit alone is not completion.

## Preparation attempt 0

Frozen commit `42689fd` failed after 0.1085 seconds while reading the archive:
`train_groups` is absent. Actual NPZ headers confirm only k-mer and mask arrays
for train/validation/test/all. The older synthetic adapter test had supplied
group arrays, so it did not establish compatibility with these actual bytes.
The failure precedes selection output, reference scanning, optimization and
model evaluation. Its invocation, source snapshot and failure/log are retained.

The prospective plan now includes an explicit amendment: keep the exact hashes,
seeds, selection sizes, labels and fitting recipe; expose missing archive groups
as unknown, then reconstruct occurrences/families/folds and grouping from the
pinned reference before admitting any observation. No unknown bucket is filled
with a fabricated supplied value. The retry uses a separate attempt directory.

## Preparation attempt 1

Frozen source `de3cdfc` independently audited all 7,000 selected eight-mers
against 569,516 reference proteins. Membership masks and source folds agree
exactly; no selected peptide has cross-fold occurrences. All 7,000 absent
archive buckets were reconstructed and remain distinguished from supplied
values (zero supplied-bucket comparisons). Connected mnemonic-family components:
3,486 train / 705 validation / 723 test. Runtime 8.9635 seconds, peak process RSS
561,709,056 bytes. See `attempts/reference-groups/results/preparation.json`,
selected/audit gzip tables, actual invocation, source snapshots and scan log.
The full occurrence audit is linked with SHA256 in that report.

The manifest SHA256 is
`ca008ea769c6dc5332cb14f4ef1d7bd125c79064223692596608588b8e4cff8f`;
normalized observation SHA256 is
`8cbd609b018cb7b8c7816de7289be368c6ac775ee92fe92932590018ad67c38e`.
This is selected reference-label admission, not verified homology independence.

`code/launch.py` calls the canonical `fit_index` trainer in a bounded child,
fits the two fixed training-only baselines, preserves predictions and coefficients,
and computes the prospectively declared validation comparison. Its metrics are
verified on constant/perfect predictions, unequal component sizes and misaligned
prediction exports (eight focused experiment/reference checks pass). Actual fit
invocation and source snapshots are written before fitting. Fresh-process replay
uses only the first training batch.
