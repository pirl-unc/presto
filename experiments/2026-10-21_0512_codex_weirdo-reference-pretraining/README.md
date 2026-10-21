# Peptide-source reference pretraining

Registered before preparation, 2026-10-21 UTC. Agent: Codex / GPT-6.
Status: first preparation stopped before audit/fitting; amended preparation pending.
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
