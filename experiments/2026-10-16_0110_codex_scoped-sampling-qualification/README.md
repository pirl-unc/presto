# Scoped sampling investigation

Status: completed; all14 public downloads, original outputs and predictions verified. Agent/model: Codex
(GPT-6). [Prospective plan](../agents/codex/plans/scoped-sampling-qualification.md).

Shuffled batches improved probability quality over balanced batches and the
prevalence baseline. Length/composition logistic remains preferred on validation.
No global sampling default changes and no training-readiness claim follow.
Repeated test outcomes are exploratory because this test set was inspected in
the earlier qualification; condition selection used validation only.

## Frozen source and execution

Public [PXD013455 HeLa Trypsin/P source](https://github.com/pirl-unc/hitlist/releases/tag/pxd013455-hela-tryp46-v1),
manifest SHA-256 `441f3c66c6e6bc6a90c7cb8c54d0131676c3cb4e7a04ddbd9346061ad8f9d84b`.
Same 10,000 globally collapsed peptides and frozen 8,000/1,000/1,000 partitions as
[the original qualification](../2026-10-15_2255_codex_scoped-detectability-qualification/README.md).
Positive counts912/105/109; train prevalence11.40%. MS/MS and matching positives
and scoped absences are proxy labels, not direct detectability measurements.
Source registry retains both papers, search settings, HeLa culture, reported
fractionation-buffer pH, reference FASTA and candidate/measurement lineage.
MHC restriction/allele sets and binding endpoints/units/censoring do not apply.
No excision/depth labels, synthetic controls, unrelated assays or live sources.
Only `ms_detectability_logit` is prospectively supported; other220 outputs remain
undeclared. Known protein overlap and shared search are retained; donor/specimen
independence is unresolved.

Clean successful source `82be704840d2aa7e87a2f646ab2f9e5866116d9a`, tag
`experiment-scoped-sampling-retry-20261016`; Presto0.1.11, Hitlist1.75.0,
datacache1.27.1, Torch2.14.1+CPU. Exact package pins, installation report, wheel,
source/code/config hashes and source snapshots are retained. Run
`reproduce/launch.sh` from that tag. Independent analysis/recomputation:
`python analysis/summarize.py /path/to/downloaded/run-files`.

CPU8 maximum, RAM32GiB, scratch512GiB, one Torch/BLAS thread, no GPU, 55-minute
work alarm. Actual total2,542.76s; runner peak RSS7,226,044,416bytes, children
excluded. Balanced condition1,206.60s; shuffled1,170.30s (including their baseline
analysis). All reusable sources/projections/partitions/output archives use shared
datacache. Temporary SQLite projection work now uses local POSIX scratch (#102).

The first shared-scratch attempt was stopped with SIGTERM during data-only
preflight, before any fit/checkpoint existed. Its logs, config and interruption
receipt are preserved separately; this was not a claimed timeout or unsupported
filesystem. The retry changed scratch placement, not scientific thresholds.
Before retry, the support manifest's policy-file hash was rebound solely for the
producer version0.1.8→0.1.11. All original assignment/membership/parent/source/
coverage/leakage/support fingerprints stayed fixed; both policy files and the
previous manifest are preserved. Every membership artifact is byte-identical
across the successful preflight and both conditions.

## Training contract and measured exposure

No pretraining. Expanded d32/l1/h4 model, batch64, lr0.001, AdamW weight decay0.01,
uncertainty task weighting enabled, seed/data_seed42, two epochs capped100 batches
each. Every explicit regularizer, including binding orthogonality, is zero in
base and per-epoch receipts. This corrects the earlier hidden-prior deviation
without rewriting that historical run. The only condition difference here is
`balanced_batches=true` versus `false` (ordinary shuffled batches).

| Condition | Optimizer steps | Positive proxy labels seen | Total labels seen | Positive exposure |
| --- | ---: | ---: | ---: | ---: |
| Balanced | 200 | 6,314 | 12,800 | 49.33% |
| Shuffled | 200 | 1,444 | 12,800 | 11.28% |

Both have200 finite-gradient detectability batches, zero direct-label batches.
Repeated exposure counts are not distinct peptides or biological replicates.
The canonical validation-loss checkpoint is epoch1 for balanced, epoch2 for
shuffled. Condition selection is minimum unweighted validation log loss, selecting
shuffled. No test value enters that selection.

## Held-out results

| Predictor | Split | AUROC | Average precision | Log loss | Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| Balanced Presto | Validation | 0.80960 | 0.32286 | 0.57755 | 0.19747 |
| Shuffled Presto | Validation | 0.77236 | 0.31172 | 0.28928 | 0.08376 |
| Train prevalence | Validation | 0.50000 | 0.10500 | 0.33634 | 0.09406 |
| Length/composition logistic | Validation | 0.82200 | 0.31577 | 0.27268 | 0.08142 |
| Balanced Presto | Test, exploratory | 0.83581 | 0.39296 | 0.57618 | 0.19732 |
| Shuffled Presto | Test, exploratory | 0.81882 | 0.37417 | 0.28081 | 0.08258 |
| Train prevalence | Test, exploratory | 0.50000 | 0.10900 | 0.34454 | 0.09714 |
| Length/composition logistic | Test, exploratory | 0.84156 | 0.34681 | 0.26899 | 0.08129 |

All probability metrics were independently recomputed from the saved per-example
predictions; identical held-out IDs and labels were checked across conditions.
Complete accuracy/balanced-accuracy/precision/recall/F1 and baseline coefficients
are retained. Exact-value binding regressions and500nM thresholds are inapplicable
to these proxy classification labels. Original loss ledgers and task-weighted
training/validation losses remain distinct from unweighted log loss above.

Paired bootstrap uses1,000 resamples, seed4242, peptide as sampling unit. Shuffled
minus logistic validation log-loss95% interval is[0.00410,0.02800]; exploratory
test interval[-0.00015,0.02361]. Test AP's apparent improvement is uncertain
(interval[-0.03526,0.09080]). Full intervals are saved for every comparison.
These intervals do not represent independent experiments/proteins/biological
material. One seed and short CPU fits limit generalization.

Decision: sampling exposure explains a substantial probability-quality difference
in this fixed contract, but the selected neural model still does not beat the
simple baseline on validation. Keep logistic preferred, preserve both conditions,
and require a prospectively frozen independent evaluation for stronger claims.
Presto #48/#53 and Hitlist #678 remain open.

## Download and inspect

[Public artifacts](https://github.com/pirl-unc/presto/releases/tag/scoped-sampling-qualification-v1)
include original complete outputs, checkpoints, every prediction and comparison
TSV, source/cell/search context, frozen membership and parent graph, support/
leakage reports, policies, effective loss configuration, baseline coefficients,
uncertainty intervals, logs and receipts. All122 archive members were verified.
Successful archive10,892,460bytes, SHA-256
`16935d480378d950367aba8e15bf2e85a654797bd1e7fde216413d123bbc9c0b`.
The stopped preflight archive and interruption evidence are published alongside
it; its partial SQLite file is not a reusable dataset.

Start with `qualification-summary.json` and `comparison.csv`. Full results stay
in their original archive; public inventories and SHA256SUMS bind each payload.
`results/sampling.json` retains original/shared datacache paths; portable URLs and
independent public-download verification are in the publication receipts.
