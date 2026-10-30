# Shared-model retraining after source curation

Status: both fits completed; held-out metrics and paired comparisons independently verified.
Codex / GPT-6, 2026-10-29. Source commit `858f5383` plus pinned dirty source.

[Prospective plan and pre-fit amendment](../agents/codex/plans/2026-10-29_curated-data-retrain.md)
· [live comparison](http://127.0.0.1:54503/dashboard.html)
· [Modal fits](https://modal.com/apps/iskandr/main/ap-L2Gu45s9NY8QIyGFBycFZV).

## Frozen data revision

Compared with the completed temporal parent, apply 506 source-scoped inferred
bounds (482 ceilings / 24 floors), withhold 618 unresolved physical TRAIN rows,
add 13,549 source-verified unmodified human MHC-II ligand observations and 13,549
length-matched generated comparisons, and remove 35 old generated TRAIN rows
matching newly verified positive sequences. The expanded positive exclusion pool
adds 292 sequences, including reviewed positives outside architectural admission.

All validation/test files remain byte-identical. One additional recovered positive
was excluded because its sequence occurs in a held-out generated comparison.
Three generated TEST rows matching that sequence are flagged as suspect. Main
comparisons retain them; an additional analysis excludes their IDs for both model
generations. Source ligand evidence is not proof of binding each comparator's MHC.
These previously inspected holdouts support repeated development comparisons,
not a fresh blind evaluation. Test predictions never select the checkpoint.

The initial preparation stopped on the blacklist assertion before any optimizer
update. Its exact source, call and log are preserved in
`attempts/blacklist-assumption/`. An independent audit established affected IDs
and hashes before the revised `source-review-v2` preparation was launched.

## Architecture, loss and training

One canonical observation-manifest trainer and shared-v5 wide model: width 192,
three layers, six attention heads, FF 768, dropout .1. No pretraining, identity
heads, new ranking loss or architecture change. Distinct measured outputs are
IC50, KD, EC50, KD(~IC50), KD(~EC50), half-life, koff, td50, qualitative binding
and elution. Unsupported kon remains absent. A quantity label is not assay-method
verification. TRAIN-only exact-value scalers, unclipped regression, one-sided
inequality residuals and binary cross-entropy are unchanged.

Two fresh seeds 42/43, batch 128, AdamW learning rate 3e-4, weight decay 1e-4,
constant schedule. Sampling: 60% elution, 20% literal IC50, 15% other physical,
5% qualitative; square-root endpoint/class allocation within each family.
Recompute canonical quotas to expose every TRAIN row at least once. This gives
**3,519,577 TRAIN observations**, **6,588,275 weighted examples** and
**51,471 updates per seed** versus the parent's 51,099. It is the same complete-
epoch policy, not exactly matched compute. Checkpoint every 500 updates, validation
every 1,000, select minimum equal-endpoint validation loss, seed breaks ties.

Synthetic comparisons use the original sampler seed 20261028 and pinned Ensembl
112 human proteome. New comparisons: 13,546 same-protein / three reference-proteome.
They are unobserved peptides, not measured nonbinders; original evaluation ratios
stay fixed. Source-reported MS nondetections are evaluated separately.

## Verification, resources and reproduction

172 focused source/admission and checkpoint/reporting/prediction checks pass.
Independent source replay verifies original CSV and PSM chemistry/row evidence;
preparation verifies exact changed/removed row IDs, raw fields, source hashes,
held-out exclusion, numerical values, scalers and complete quotas. A second full
cohort census independently passed. Both GPU workers passed canonical preflight
and independently reproduced the frozen scalers and manifest hashes: 1,679.08 s
(seed 42) / 1,723.61 s (seed 43). No optimizer update had been observed at the
2026-10-29 21:04 UTC check; `results/worker-preflight-verification.json` records it.
Hosted CI at `865eac08` passed 2,879 tests / three skips, full lint and docs.
CPU preparation: eight CPUs / 64 GiB requested,
246.44 seconds. Requested fit hardware: at most two H100!, eight CPUs / 64 GiB
host RAM and 512 GiB scratch each. Both workers report NVIDIA H100 80GB HBM3, 85,009,367,040 device bytes;
see `results/initial-hardware.json`. Final runtime and peak memory are recorded below.

Model and trainer source are byte-identical to the parent runtime. Only the new
measurement-curation module, common Hitlist admission and already validated faster
common input reader differ. All 214 source files, dependency versions, proteome,
prepared record and manifests are hash-pinned. No active input mutation is allowed.

`reproduce/launch.json`, `training-contract.json`, `runtime-pins.json`, manifests,
call IDs and `reproduce/source/` preserve exact invocations and executed source.
Preparation receipt: `results/prepared.json`; full source delta and blacklist
receipts are in `results/`. Durable prepared data and run artifacts are on Modal
volume `presto-checkpoints` at this family / `revisions/source-review-v2/`.
Local upload runtime is under `artifacts/2026-10-29_1956_codex_curated-data-retrain/`.
Large collected per-example predictions/checkpoints will remain there, referenced
by hashed collection receipts in the canonical experiment directory.

The completed continuation (`code/advance.py`) collects complete receipts, independently
recomputes held-out target/loss and per-output/class/era metrics, writes the full
report, and verifies matched observations before paired parent comparisons.
`analysis/compare_parent.py` also reports sensitivity to the three suspect test
labels. The live report shows parent selected checkpoints against latest new
validation checkpoints, explicitly different training stages. The results below record measured differences; model promotion and scientific qualification require separate review.

The prospective post-run closeout now also writes the measured summary and paired
deltas to this README and the canonical registry, only after independent prediction,
cohort and comparison checks pass. Its frozen pre-prediction source and fail-closed
verification are in `reproduce/analysis-before-predictions/` and
`results/closeout-readiness.json`. The separately frozen v2 reporting revision
checks non-null binary counts, removes stale pending wording, preserves unrelated
registry entries and passes repeated-closeout checks using historical parent
artifacts only in a temporary directory; see
`reproduce/analysis-before-predictions-v2/` and `results/closeout-readiness-v2.json`.
Completion does not automatically promote the
model or qualify TCR, DO:DM or processing components.

Later source curation, completed after launch but before new validation results,
identifies 33 additional generated TRAIN comparisons whose sequences now have
source-positive evidence, and one more suspect TEST comparison. These appear in
the separate curation family's `results/late-candidate-audit.json`. Active inputs,
main test denominators and the original three-row sensitivity are preserved.
This is an explicit limitation of the current frozen cohort; the next preparation
must incorporate the later positive exclusion pool before training.

## Verified retrain results

Both complete weighted-epoch fits, independent target/loss verification, cohort census, serving replay and paired held-out observation checks passed.
Validation-selected condition: `curated-multi-s43` (minimum equal-endpoint validation loss).
Selection scores: curated-multi-s42 = 0.680632; curated-multi-s43 = 0.654264.
All results remain repeated fixed-holdout development evidence; no model is automatically promoted and no mechanistic qualification follows.

| Condition | Split / era | Output / class / population | n (exact for IC50) | Spearman | RMSE log10(nM) | AUROC | AUPRC |
|---|---|---|---:|---:|---:|---:|---:|
| curated-multi-s42 | test / 2026 | elution_observation / I / source_reported_outcomes / pmid:42069060 | 5169 | — | — | — | 1.0000 |
| curated-multi-s42 | test / 2026 | elution_observation / I / synthetic_same_protein | 31002 | — | — | 0.9726 | 0.9319 |
| curated-multi-s42 | test / 2026 | elution_observation / II / source_reported_outcomes / pmid:41534530 | 6611 | — | — | — | 1.0000 |
| curated-multi-s42 | test / 2026 | elution_observation / II / synthetic_same_protein | 39108 | — | — | 0.8282 | 0.6677 |
| curated-multi-s42 | test / older_or_mixed | elution_observation / I / source_reported_outcomes / pmid:29308320 | 136 | — | — | 0.2428 | 0.4741 |
| curated-multi-s42 | test / older_or_mixed | elution_observation / I / synthetic_same_protein | 432 | — | — | 0.9959 | 0.9740 |
| curated-multi-s42 | test / older_or_mixed | ic50 / I / reported_measurements | 100 | 0.3758 | 1.0880 | — | — |
| curated-multi-s42 | test / older_or_mixed | ic50 / II / reported_measurements | 267 | 0.5837 | 0.9823 | — | — |
| curated-multi-s42 | validation / older_or_mixed | elution_observation / I / source_reported_outcomes / pmid:35474673 | 53 | — | — | 0.7639 | 0.8525 |
| curated-multi-s42 | validation / older_or_mixed | elution_observation / I / source_reported_outcomes / pmid:35561310 | 1643 | — | — | — | 1.0000 |
| curated-multi-s42 | validation / older_or_mixed | elution_observation / I / source_reported_outcomes / pmid:40138458 | 52 | — | — | 0.4381 | 0.7894 |
| curated-multi-s42 | validation / older_or_mixed | elution_observation / I / synthetic_same_protein | 10116 | — | — | 0.9945 | 0.9716 |
| curated-multi-s42 | validation / older_or_mixed | elution_observation / II / source_reported_outcomes / pmid:35561310 | 694 | — | — | — | 1.0000 |
| curated-multi-s42 | validation / older_or_mixed | elution_observation / II / synthetic_same_protein | 4152 | — | — | 0.8675 | 0.6543 |
| curated-multi-s42 | validation / older_or_mixed | ic50 / I / reported_measurements | 141 | 0.5707 | 1.1341 | — | — |
| curated-multi-s42 | validation / older_or_mixed | ic50 / II / reported_measurements | 1129 | 0.5306 | 0.8551 | — | — |
| curated-multi-s43 | test / 2026 | elution_observation / I / source_reported_outcomes / pmid:42069060 | 5169 | — | — | — | 1.0000 |
| curated-multi-s43 | test / 2026 | elution_observation / I / synthetic_same_protein | 31002 | — | — | 0.9712 | 0.9293 |
| curated-multi-s43 | test / 2026 | elution_observation / II / source_reported_outcomes / pmid:41534530 | 6611 | — | — | — | 1.0000 |
| curated-multi-s43 | test / 2026 | elution_observation / II / synthetic_same_protein | 39108 | — | — | 0.8337 | 0.6738 |
| curated-multi-s43 | test / older_or_mixed | elution_observation / I / source_reported_outcomes / pmid:29308320 | 136 | — | — | 0.1012 | 0.4159 |
| curated-multi-s43 | test / older_or_mixed | elution_observation / I / synthetic_same_protein | 432 | — | — | 0.9966 | 0.9805 |
| curated-multi-s43 | test / older_or_mixed | ic50 / I / reported_measurements | 100 | 0.3023 | 1.0443 | — | — |
| curated-multi-s43 | test / older_or_mixed | ic50 / II / reported_measurements | 267 | 0.5662 | 1.0001 | — | — |
| curated-multi-s43 | validation / older_or_mixed | elution_observation / I / source_reported_outcomes / pmid:35474673 | 53 | — | — | 0.7610 | 0.8564 |
| curated-multi-s43 | validation / older_or_mixed | elution_observation / I / source_reported_outcomes / pmid:35561310 | 1643 | — | — | — | 1.0000 |
| curated-multi-s43 | validation / older_or_mixed | elution_observation / I / source_reported_outcomes / pmid:40138458 | 52 | — | — | 0.4381 | 0.7894 |
| curated-multi-s43 | validation / older_or_mixed | elution_observation / I / synthetic_same_protein | 10116 | — | — | 0.9929 | 0.9658 |
| curated-multi-s43 | validation / older_or_mixed | elution_observation / II / source_reported_outcomes / pmid:35561310 | 694 | — | — | — | 1.0000 |
| curated-multi-s43 | validation / older_or_mixed | elution_observation / II / synthetic_same_protein | 4152 | — | — | 0.8638 | 0.6575 |
| curated-multi-s43 | validation / older_or_mixed | ic50 / I / reported_measurements | 141 | 0.5491 | 1.2805 | — | — |
| curated-multi-s43 | validation / older_or_mixed | ic50 / II / reported_measurements | 1129 | 0.4928 | 0.8342 | — | — |

Missing metrics indicate unsupported or single-label evaluation cells.

- `curated-multi-s42`: selected step 51,471; completed 51,471 updates and exposed 3,519,577 unique TRAIN rows. Requested H100!, observed NVIDIA H100 80GB HBM3 (85,009,367,040 device bytes); wall 41898.4 s, optimizer 29539.3 s, peak allocated/reserved 3.77 / 12.01 GB.
- `curated-multi-s43`: selected step 51,471; completed 51,471 updates and exposed 3,519,577 unique TRAIN rows. Requested H100!, observed NVIDIA H100 80GB HBM3 (85,009,367,040 device bytes); wall 55496.4 s, optimizer 32625.5 s, peak allocated/reserved 3.81 / 13.30 GB.

The full report preserves every output/class/era/study, exact and bounded loss, Pearson, 500 nM classification, uncertainty, source-reported outcomes, parameter/feature coverage and register diagnostics. The paired comparison checks identical held-out observations and separately excludes the three source-identified suspect generated test labels for both model generations.
[Full report](results/summary.md) · [Paired comparison](results/parent-comparison.md)
