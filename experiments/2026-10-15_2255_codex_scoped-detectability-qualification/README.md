# Frozen within-search detectability qualification

Status: pipeline qualification completed; probability quality failed both baselines.
A hidden regularization default caused a protocol deviation (#95).
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
per-example validation/test predictions are preserved in the public fit archive.
No training-readiness claim is made.

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
intervals are retained in datacache and publicly downloadable. These intervals cannot establish
between-experiment uncertainty.

## Actual bounded fit and held-out comparison

The fit ran on clean commit `65173d2598746ebb848031783833f74c833679f2`,
preserved by tag `experiment-scoped-fit-20261016`: 1,027.01 seconds, runner peak
RSS 7,912,128,512 bytes, CPU only, one Torch/BLAS thread. Two epochs completed
200 optimizer steps; epoch 1 was selected by validation loss. No pretraining,
no generated labels, no calibration correction and no test-based tuning.
[results/fit.json](results/fit.json) contains exact versions/configuration and
source paths; [comparison.json](results/comparison.json) preserves every metric,
and [uncertainty.json](results/uncertainty.json) has 1,000 paired bootstrap draws.

| Split | Predictor | AUROC | AP | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | Presto | 0.80628 | 0.33324 | 0.55073 | 0.18739 |
| Validation | Train prevalence | 0.50000 | 0.10500 | 0.33634 | 0.09406 |
| Validation | Length/composition logistic | 0.82200 | 0.31577 | 0.27268 | 0.08142 |
| Test | Presto | 0.82984 | 0.38070 | 0.54864 | 0.18618 |
| Test | Train prevalence | 0.50000 | 0.10900 | 0.34454 | 0.09714 |
| Test | Length/composition logistic | 0.84156 | 0.34681 | 0.26899 | 0.08129 |

At the fixed 0.5 threshold, Presto test balanced accuracy is 0.76123,
precision 0.25356, recall 0.81651 and F1 0.38696. Binding regression and
500 nM metrics do not apply to this binary search-observation proxy.
The train-only logistic baseline is preferred for probability quality and
AUROC under this bounded contract; Presto has higher point-estimate AP.
All 1,000 predictions per held-out split are finite and reconcile IDs,
labels and lineage with frozen membership. The canonical held-out summaries
also retain total loss including the unintended prior below; the table reports
binary log loss directly from probabilities, consistently across predictors.

The supervised target is `row:ms_detectability` -> `ms_detectability_logit`,
from scoped MS/MS, matching and unobserved proxy categories. All 200 batches
have labels and finite output gradients. Actual exposure is 12,800 labels,
6,314 positive (49.33%), versus 912/8,000 (11.40%) in frozen training membership.
[detectability_updates.json](results/detectability_updates.json) records the
measured source/label exposure. Balanced and contrast sampling change exposure;
a prior-only correction has not been justified. See [#94](https://github.com/pirl-unc/presto/issues/94).
Every other output remains undeclared; optimizer activity is not evidence of
measured support for another output. Ungrounded default synthetic probe outputs
in the archive are diagnostics, not held-out measured evidence.

**Protocol deviation:** although all six exposed consistency weights were
zero, `_regularization_config_from_args` silently supplied
`binding_orthogonality_weight=0.01`. The observed epoch-1 regularizer loss is
0.000242965. This run therefore does not implement the proposed all-consistency-
losses-zero objective. The original launch files, saved config and archive are
unchanged; [#95](https://github.com/pirl-unc/presto/issues/95) tracks explicit
configuration and effective-loss reporting before a separately registered rerun.
Uncertainty task weighting and weight decay 0.01 were active canonical defaults.

## Public artifacts and decision

[Download the complete qualification](https://github.com/pirl-unc/presto/releases/tag/scoped-detectability-qualification-v1).
[publication.json](results/publication.json) pins each URL, size and SHA-256.
The fit archive SHA-256 is
`ab664220951f912433022684aa408ed0280c7786f9772661694acb87caa7351c`;
the preflight archive is
`fb0881f60ed993614e683a2f9e588e6f02b83d91228a78b01833db5ab47e9244`.
Archives include the complete partitions, readable membership/context, coverage,
leakage and lineage reports, checkpoint, per-example predictions, baseline state,
logs and per-file inventory. Original and public downloads are shared in datacache.

The pipeline acceptance gates pass. Probability quality fails both baselines,
and the regularization protocol deviated. Neither the bounded fit nor its label
support certifies full training readiness. Shared proteins/search and unresolved
donor/specimen identities limit generalization; all other outputs and independent
experiments still need qualification. Fix #95, investigate #94 prospectively on
train/validation, and use fresh untouched evaluation data for any confirmatory
claim after selecting a correction. Do not optimize against the inspected test.

