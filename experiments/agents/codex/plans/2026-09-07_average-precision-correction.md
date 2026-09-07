# Issue #52 archived AP correction plan

Agent/model: Codex / GPT-6. Date: 2026-09-07.

Question: which available canonical held-out CSV artifacts have AP values
affected by grouping tied scores? This is a retrospective metric correction,
with no new model forward, optimization, model selection or pooled ranking.

Freeze all locally available prediction CSV paths discovered under experiments,
modal_runs and artifacts. Select the canonical task/y_true/y_pred schema;
preserve exclusions for focused/distributional formats and unlabeled probes.
Record SHA-256 of every selected CSV and paired summary. Preserve each original
validation/test split, censor qualifier and real/decoy/mapping stratum.
The selected-schema sample dump is partial and must not be described as a full
split. Recompute only task families with an unambiguous canonical loss type.

Compare the old row-wise AP and corrected threshold-group AP on exactly the
same score vectors. For historical summaries, replace AP only when the old
recomputed value agrees with the archived value; report mismatches without
silently changing unrelated metrics. Preserve all non-AP metrics and loss
metadata verbatim. Save corrected copies and a per-metric delta table.
Keep source prediction dumps immutable and link their checksummed paths.

Cross-check corrected AP against sklearn 1.5.2 on 500 seeded tied/untied arrays.
Production and CI tests retain no sklearn dependency. Preserve the legacy
one-class undefined policy separately from sklearn's own convention.

Record code revision, dirty status, source snapshots, exact command/env, CPU
runtime, considered/excluded files, sample and tie counts, corrected metrics,
all mismatches, summary paths and limitations. Add the completed family to
experiment_log.md; no new training result or new-model quality claim.

