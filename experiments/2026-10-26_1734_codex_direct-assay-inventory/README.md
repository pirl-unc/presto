# Literal direct-assay source inventory

**Closed: both independent scans agree on all840,805 source rows and384 joint cells.**
Registered and executed2026-10-26 by Codex / GPT-6. Zero rows admitted for fitting.

Frozen source: Hitlist ci-corpus-v4, the existing2026-10-16 export; this is not
a latest-release claim. The original binding.parquet (814,268 rows) and
other_assays.parquet (26,537 rows) are hash-verified against the pinned
[source contract](reproduce/plan.md), under the existing local release path
`/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/2026-10-16T07-20-43Z/`.

The census preserves literal response, method, unit, qualifier, source, class,
PTM and binding/MS flags, including null separately from empty strings.
Numeric state is only missing/nonfinite/nonpositive/positive finite. No
qualifier interpretation, unit conversion, target mapping, sequence/groove
resolution, deduplication, split assignment or biological admission occurred.

| Literal source response | Unit | Raw rows | Positive finite | Purified metadata candidates |
| --- | --- | ---: | ---: | ---: |
| MHC binding | (empty) | 468,548 | 0 | 0 |
| qualitative binding | (empty) | 89,787 | 6,231 | 5,388 |
| dissociation constant KD (~IC50) | nM | 88,645 | 88,633 | 88,418 |
| half maximal inhibitory concentration (IC50) | nM | 74,267 | 74,027 | 65,420 |
| dissociation constant KD (~EC50) | nM | 57,523 | 57,523 | 57,518 |
| ligand presentation | (empty) | 24,977 | 0 | 0 |
| dissociation constant KD | nM | 21,023 | 21,018 | 20,344 |
| half life | min | 11,358 | 10,611 | 10,052 |
| half maximal effective concentration (EC50) | nM | 1,854 | 1,795 | 683 |
| 3D structure | angstroms | 1,554 | 1,537 | 0 |
| 50% dissociation temperature | °C | 1,152 | 1,140 | 1,043 |
| off rate | 1/s | 71 | 69 | 49 |
| on rate | nM^-1s^-1 | 36 | 36 | 30 |
| 3D structure | (empty) | 6 | 0 | 0 |
| association constant KA | 1/M | 4 | 4 | 0 |

The last column is a metadata-only screen: positive finite value,
has_ptm=False, IEDB/CEDAR, classI/II and a literal method beginning
`purified MHC/`. It does **not** check sequence chemistry, human restriction,
explicit class-II partners, catalog agreement, raw numeric/bound text, original
assay lineage, duplicates or leakage. Counts cannot be called training support.

Decision: qualify the literal **dissociation constant KD** source next, then
assess incremental retained-IC50 performance only after the fixed affinity
comparison is reviewed. Preserve `KD (~IC50)` and `KD (~EC50)` as separate
proxy families. Likewise, `50% dissociation temperature` requires assay-level
qualification before any melting-temperature mapping; neither qualitative
binding values nor off/on rates are silently reassigned to another endpoint.

Verification uses a second scan with Arrow vector predicates/group-by and2048
row batches, independent of the first Python row-wise4096-batch aggregation.
All joint counts, null/empty counts, distinct nonempty assay/reference ID
cardinalities, parquet totals and source hashes agree. The full source schema
and per-source summaries are in [census.json](results/census.json); every cell
is in [joint-counts.csv](results/joint-counts.csv), with JSON-encoded field
values distinguish null/empty/false. [Replay](results/independent-verification.json).

Execution commit: `4008ae1573ad4afe480db42dfac3ffeb6b25d66c`; exact dirty state, invocation,
environment and package versions are recorded in [launch.json](reproduce/launch.json).
Only independent experiment artifacts were untracked; existing model/runtime
code stayed unchanged. Reproduction uses [launch.sh](reproduce/launch.sh)
at the executed code commit in a fresh checkout with the same pinned source
files. Existing results are protected from overwrite. Code snapshots and the
saved-output report source are under `reproduce/source/`.

Local CPU, one Arrow/BLAS thread; existing Python3.12.6, no installs, network,
cloud or GPU. No model, pretraining, optimizer, loss weights, synthetic data,
assay-output mapping or selected condition exists. Predictive metrics and
validation/test dumps are inapplicable because no cohort/splits or model were
created. This source inventory does not qualify any scientific endpoint.

| Operation | Elapsed seconds | Peak RSS bytes | Status |
| --- | ---: | ---: | --- |
| inventory | 3.428574 | 217,513,984 | completed |
| independent-replay | 1.128621 | 262,078,464 | completed |

Both scans stayed within the registered300s/3GiB process ceilings and
available-RAM/disk floors. The raw sources remain at the pinned original
paths; no new large cohort was materialized alongside IC50 training.

Status: closed source inventory. Next step: strict direct-KD admission.
Open questions: original-assay interpretation, deduplication, explicit partner
coverage, leakage control and useful retained-task fitting. PR131 remains draft.
