# Native peptide response recipe admission

The canonical native-cell response recipe passes independent source, reference,
split, target and weight checks. It supplies a defined biological benchmark for
the next fit, not a concentration predictor.

Agent/model: Codex / GPT-6, October 24, 2026. Presto source
`593e3b5c2bc08ff9242b363145096f2ba6234e95`, with recorded task-file dirt. Hitlist
source is the frozen working tree merging `ddc6581` and `7649ffe`; all changed
source files, the complete diff and prospective specs are in `reproduce/`.
This is the v3/schema-17/catalog-22 successor, development version 2.0.17.
It is not the concurrently merged PR794's v2 package and is not yet a verified
published release. Shared implementation: [Hitlist PR793](https://github.com/pirl-unc/hitlist/pull/793).

## Source and target contract

Public Dasgupta2014/Gelman2013 native human HEK293T/SH-SY5Y cell-extract peptidomics,
the same four pinned primary/heatmap workbooks as the 0117 source audit. All
10,564 rows and 61,908 measurement slots remain unchanged in source identity,
quantities, qualifiers and missingness. The source replay verifies 1,736,376
original-field invariants. Cross-publication fingerprints retain all 830 prior
identified-sequence candidate groups and add 553 unidentified-peak candidates.
These are possible reuse, not adjudicated biological replicate identities.

The registered `tmab_native_response_v1` recipe admits reported supported peptide
identities and positive exact or explicitly bounded treated/control ratios.
Controls, derived averages, unknown/tentative/unsupported identities, nonquantifiable
values and conflicting possible reuse remain in exported dispositions. No
negatives, imputation, affinity/length filter or synthetic biological targets.
Charge states and ratio slots are grouped as contributors, without claiming
independent specimens. Exact source values are not averaged together.

All source sequences, source parent/gene assertions, possible reuse and shared
full-reference alternatives define connected components before admission. The
deterministic hash split is unchanged. Reference mapping preserves every exact
occurrence in all 42,562 reviewed human UniProt 2026_03 isoform-inclusive records;
1,155 occurrences involve 345 records. This modern model reference is not the
original IPI search library or proof of a unique origin.

| Split | Response groups | Components with targets | Effective observations | Exact | Bounded | Full-parent-supported groups |
|---|---:|---:|---:|---:|---:|---:|
| Train |1,861|86|6,735|4,301|2,434|1,855|
| Validation |245|19|932|504|428|245|
| Test |631|16|2,447|1,278|1,169|631|

Total: 2,737 response groups, 10,114 effective observations (6,083 exact / 4,031
bounded), representing 11,495 original slots. Each split's weights sum to one,
averaging condition, component, peptide form, then effective observation. The
recipe retains 50,413 deferred slots; 1,100 were otherwise eligible but have
conflicting possible reuse contexts. There are 412 conflicting candidate groups
across all source identities, and 4,843 all-source components including deferred
rows. These are not counts of independent biological samples.

## Verification and resources

The public Hitlist writer/materializer and its portable replay pass in 54.87 s,
peak RSS 1.415 GiB. An independent verifier uses whole-string identity grammar,
Aho–Corasick matching checked against naive overlapping searches, and bipartite
graph traversal instead of the producer's token scanner, prefix index and union-find.
It checks every original table, all 42,562 reference proteins and 1,155 occurrences,
all component assignments, every target/contributor/disposition and weight. It
passes in 10.59 s, peak RSS 1.069 GiB. Exact receipts are in `results/`.

The first independent verifier stopped because its taxonomy check required a space
after OX=9606 and rejected a valid header ending at that token. The failed log and
source remain preserved. The documented correction accepts the token at the end
of a header; no data, split, target, gate or producer changes were made.

Local CPU, one OMP/OpenBLAS/MKL thread, shared Python environment unchanged; no
Modal/GPU. Prospective producer cap 600 s / 4 GiB; verifier cap 180 s / 4 GiB.
Shared Hitlist format/lint pass and 3,414 normal full-suite tests pass in 630.18 s
(runner wall time 640.03 s, peak RSS 930,889,728 bytes). These validate the frozen
local implementation, not a later commit's hosted gates or PyPI publication.

## Reproduction and artifacts

- `reproduce/launch.sh`, `launch.json`, `execution.json`, source snapshots and
  `hitlist.patch`: exact producer invocation and implementation.
- `reproduce/verify.json`, `verify-v2.json`, both verifier snapshots and correction
  note: independent execution contract; run `analysis/verify.py` on restored inputs.
- `results/source-replay.json`, both manifests, recipe summary, split coverage,
  materialization and independent verification receipts.
- Raw source and recipe directory:
  `../../data/native_peptidome_recipes/2026-10-24_0259_codex_native-response-admission/`.
  Source manifest SHA256 `7f80844bc39748bb4fbd06e5747bac0769e06d1f7a1038f82daeae29b41b406f`;
  recipe manifest SHA256 `c72980c070cb1296609bbe7d296e27b6a2a22d3ff354ef846c1c99ad7959ccfc`.
- Full reference SHA256 `8c4459cc1e190caad54150511e5a0b5779863557f88387e5a86647a48d19a9d4`;
  source URLs/hashes and portable file inventory are in the frozen manifests.

No model, training/pretraining, loss optimization or predictive validation/test
evaluation occurs in this admission family. Prediction dumps and regression/binding
metrics therefore do not apply; the prospective split is preserved for fitting.

## Limits and handoff

Thirty-three shared-control label proxies cross parent splits. Actual denominator,
specimen, original acquisition and some reuse identities remain unresolved. The
split is not specimen-, homology- or external-study-independent. Chemical exposure
is not a genetic knockout or measured enzyme activity. Same-form MS response
cancellation is a modeling assumption. Ratios do not identify absolute amounts,
precursor changes, compartment-specific free peptide pools or APC concentrations.

Status: source/recipe admission closed; shared release delivery still pending.
Next step: the separately registered 0314 biological response baselines, then the
matched full-parent differentiable/sampled model. Open questions: biological
generalization, precursor and observation effects, APC machinery and concentration
validation. The full goal stays active; the MAPP transfer stop rule remains enforced.
