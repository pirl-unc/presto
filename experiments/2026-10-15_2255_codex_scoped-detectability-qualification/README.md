# Frozen within-search detectability qualification

Status: preflight succeeded; fitting is registered and has not run yet.
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
per-example predictions are required after fitting. No fitted predictions or
performance metrics exist yet, and no training-readiness claim is made.

`reproduce/launch.json` freezes the preflight arguments. The launcher writes
complete reusable run outputs and an inventory into shared datacache. Small
results and exact producer commit/dirty status will be recorded here after each
materially informative run. The previous admission-only family did not freeze
partitions or fit a model; this family adds those stages under corrected enzyme
contracts. Experimental/donor/specimen independence remains unresolved.

## Actual preflight

The registered preflight ran on clean commit
`a15c1a45f9d9eda9d111459b6f1d9f86a56e8e29`, before rebasing the pending PR onto
merged scoring correction #90. The package implementation is unchanged by that
rebase. [results/preflight.json](results/preflight.json) records the original
commit, environment, datacache directory, archive hashes and measured counts.
Runtime was 108.78 seconds; runner peak RSS was 998,916,096 bytes. No model was
constructed or fitted in that preflight.

| Partition | Peptides | MS/MS | Matching | Not observed in scope |
| --- | ---: | ---: | ---: | ---: |
| Train | 8,000 | 782 | 130 | 7,088 |
| Validation | 1,000 | 95 | 10 | 895 |
| Test | 1,000 | 91 | 18 | 891 |

All preset support minimums are met. No peptide, candidate ID or measured
observation ID crosses partitions. Known protein overlaps are 473 test/train,
123 test/validation and 495 train/validation. Every partition shares the same
search; donor/specimen independence remains inconclusive. There are zero
lineage validation issues or duplicate sample IDs. Ambiguous parent occurrences
remain masked rather than receiving a guessed protein or flank context.

The fitting declaration now freezes the exact preflight source/model contract
in [reproduce/supported_outputs.json](reproduce/supported_outputs.json), with the
unchanged prospective minimums. [reproduce/fit.json](reproduce/fit.json) requires
split-support fingerprint
`54e9c5527bd63925a508fe4ac89c0b1cd75e7a20fb159db58efe0ca4c7a18ec0`.
Every other output remains undeclared. The launcher uses the canonical trainer
and compares its selected checkpoint with train-only prevalence and standardized
length/amino-acid-composition logistic baselines. Baseline coefficients, all
validation/test scores, complete membership context and paired peptide-bootstrap
intervals will be retained in datacache. These intervals cannot establish
between-experiment uncertainty.

Status: fitting/evaluation and public qualification artifact pending.
Next step: execute the committed bounded fit and report outcomes, including failures.
Open questions: broader experiments, donor/specimen identities and other outputs.
