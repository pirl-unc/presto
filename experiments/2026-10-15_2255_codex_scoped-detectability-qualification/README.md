# Frozen within-search detectability qualification

Status: registered before execution; preflight and fitting results pending.
Agent: Codex (GPT-6). Plan: `experiments/agents/codex/plans/scoped-detectability-qualification.md`.

This family tests the canonical consumer of the verified PXD013455 HeLa search,
using 10,000 collapsed peptides and frozen 80/10/10 membership. The public input
manifest is `441f3c66c6e6bc6a90c7cb8c54d0131676c3cb4e7a04ddbd9346061ad8f9d84b`.
MS/MS, matching and scoped absence labels are proxies for observation in that
search. Source citations, cell conditions, assay metadata and candidate lineage
remain in the datacache source/partition artifacts. MHC restrictions do not apply.

One CPU/Torch/BLAS thread, no GPU; per-stage deadline 30 minutes. No pretraining,
live auxiliary inputs, fabricated assay IDs, generated negatives, excision labels
or depth labels. The prospective support thresholds and bounded fitting/baseline
conditions are fixed in the plan. Both validation and test splits are requested;
per-example predictions are required after fitting. No predictions or metrics
exist yet, and no training-readiness claim is made.

`reproduce/launch.json` freezes the preflight arguments. The launcher writes
complete reusable run outputs and an inventory into shared datacache. Small
results and exact producer commit/dirty status will be recorded here after each
materially informative run. The previous admission-only family did not freeze
partitions or fit a model; this family adds those stages under corrected enzyme
contracts. Experimental/donor/specimen independence remains unresolved.
