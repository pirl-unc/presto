# Peptide-source reference pretraining

Registered before preparation, 2026-10-21 UTC. Agent: Codex / GPT-6.
Status: preparation pending; no model fitted or test predictions inspected.

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
