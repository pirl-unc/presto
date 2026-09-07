# Retrospective average precision correction (#52)

Date: 2026-09-07. Agent/model: Codex / GPT-6. Status: complete.

Plan: [metric correction specification](../agents/codex/plans/2026-09-07_average-precision-correction.md).
Reproduce from the repository root with
`bash experiments/2026-09-07_1851_codex_average-precision-correction/reproduce/launch.sh`.

This reuses stored predictions without model initialization, pretraining,
optimization or new inference. Source training/splits/assay/curation and
synthetic-label contracts remain those of each original experiment, identified
by paths and checksums in `reproduce/inputs.json`. Loss terms/weights and
non-AP metrics are retained from the original summary. The only changed factor
is AP tie handling. No single pooled dataset or cross-model winner is defined.

Both validation and test artifacts are included when present; the one sample
dump is explicitly partial. Full prediction files stay at their original paths
with SHA-256 verification; no test split is created or omitted for this audit.
Coverage excludes unrelated focused/distributional trainers and unlabeled
probes, whose estimators are outside this issue's canonical held-out scope.

Code revision and dirty status, exact invocation/env, dependency versions and
frozen analysis/estimator sources are in `reproduce/`. Requested GPU: none;
observed hardware: local CPU. This is not a new-encoder predictive baseline.

## Results

The local discovery considered 301 prediction CSVs. The frozen manifest selects
39 canonical-schema files (428,537 rows, including one partial sample) and
excludes 262 other-schema/empty/probe files. All 1,123 defined comparable AP
values reproduce the archived estimator before correction: **738 change,
385 are unchanged, and none mismatch**. There are 36 affected full-split files.
The partial September 2 panel sample supplies no comparable defined AP metric.

| Original experiment | Split | Comparable AP metrics | Corrected | New minus old AP range |
|---|---|---:|---:|---:|
| September 2 source-junction masking | validation | 184 | 121 | -0.027778 to +0.083333 |
| September 2 source-junction masking | test | 186 | 114 | -0.008080 to +0.037798 |
| September 3 flank-context fixes | validation | 364 | 253 | -0.031746 to +0.055258 |
| September 3 flank-context fixes | test | 364 | 250 | -0.005171 to +0.083333 |
| September 5 PR #45 smoke | validation | 12 | 0 | 0 |
| September 5 PR #45 smoke | test | 13 | 0 | 0 |

The largest absolute difference is 0.083333 in small mapping strata, e.g. a
five-observation foreignness stratum changes from 0.583333 to 0.666667. Ties can
bias the original result in either direction. The latest PR #45 smoke's AP
values are unchanged; this does not strengthen its narrow quality evidence.

Artifacts:

- [summary.json](results/summary.json): file-level provenance, correction paths,
  counts and runtime.
- [ap_deltas.csv](results/ap_deltas.csv): every defined task/stratum AP, sample
  and tie counts, original/recomputed/corrected value, delta and verification.
- `results/reissued/`: corrected copies of the 36 affected summaries. Each
  contains exact source paths/hashes and the list of reissued metrics.
- [inputs.json](reproduce/inputs.json): source validation/test prediction paths,
  SHA-256 hashes, original summary hashes, partial status and exclusions.

All non-AP values, including exact Spearman/Pearson/RMSE, binary metrics and
available loss metadata, remain byte-value equivalent after JSON parsing.
Verification restored only corrected AP fields and removed correction metadata,
then asserted equality with each original summary. This does not supply loss
terms or missing outputs absent from historical artifacts; #51/#53 cover those
measurement gaps. Historical null-to-NAN/masking validity errata remain in force.

The corrected estimator agrees with scikit-learn 1.5.2 on 500 seeded tied and
untied arrays (maximum absolute error 3.33e-16). One-class results intentionally
remain omitted under Presto's reporting policy. Runtime: 7.065 seconds on local
CPU. Production commit: `93a9c7b24c6108577dcf40c94a9ff34e6c30cc5c`; only the
experiment bundle/plan were uncommitted. Frozen estimator and analyzer sources
are retained in `reproduce/source/`; the analyzer imports the frozen estimator.
The original frozen analyzer used CSV CRLF terminators. The canonical analyzer
and checked-in delta CSV use LF for repository whitespace checks; parsed CSV
records were asserted identical during normalization. No metric changed.

## Decision and handoff

Use `average_precision_distinct_thresholds_v2` for subsequent held-out metrics.
For affected old runs, use these explicitly reissued AP values and preserve the
original files as historical evidence. There is no new model winner, training
run, source/synthetic mixture or loss weighting in this experiment. BCE labels
map to their existing task logits; binding values/qualifiers map to the existing
500 nM threshold classifier, retaining exact and only definite censored classes.

Status: complete. Next step: merge the metric repair after CI/review, then use
the corrected estimator for #53. Open questions: unrelated trainer estimators
and unsupported/missing historical output artifacts remain outside this repair.
