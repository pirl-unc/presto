# Original MAPP source and baseline candidates verify; release pending

Codex / GPT-6, October24,2026. Registered before real compilation at Presto
`a8d5ff0074495826bfc7ba3d7f028787dddfff50`, with unrelated task dirt and new
experiment files. Hitlist base `9583aeb346e3181e800afa3e0293e51492b4c181`, isolated
implementation candidate2.0.30. [Prospective specification](reproduce/plan.md).
No cloud, installation, or shared-environment modification.

**Source and candidate verification pass. No model was trained and no predictive
improvement is claimed.** Full repository verification and publication/release are
recorded below as they complete. This advances the data prerequisites for the full
APC availability/concentration goal, which remains active and unachieved.

## Source and curation contract

Six pinned original Goldberg2025/PXD052314 assets from the preceding
[source audit](../2026-10-24_2130_codex_mapp2025-source/README.md), reused without
new downloads: original MAPP processed ZIP, full publisher supplement ZIP, original
deposition workbook, community SDRF and PRIDE project/file inventories. Complete
source assets remain immutable in the new package. Other publisher activity tables
are retained only as source ZIP bytes; no activity ranking or pathogen-response
modeling occurs.

Original archive rows, field order, raw values, qualifiers, parent alternatives,
contributor IDs and exact source locators are retained. Peptide totals, molecular
forms and individual identifications are different grains. Zeros, missing and
invalid values do not become biological negatives. Source header taxonomy is an
annotation assertion, not independently verified peptide origin. The full original
human-plus-other search FASTA remains absent.

The common material graph distinguishes project, labeled study arm and acquisition.
It contains no invented specimen or aliquot pairs. Unknown machinery is not assumed
wild type. Original deposition, search-run and community assertions keep separate
scopes; community raw-name/preparation conflicts remain inspectable. Publisher
ReadMe values and date-styled cells remain literal source cells. Whole-cell digest,
captured protein, captured peptide and secretome quantities retain distinct pools.
No exact physical precursor pairing or professionally antigen-presenting cell
population is established by this A549 source.

## Materialized source evidence

| Relation or check | Result |
|---|---:|
| Original peptide rows |30,364|
| Original molecular-form rows |32,636|
| Individual identification rows |294,756|
| Original peptide/form/identification intensity cells |1,113,756|
| Original publisher XML cells |3,851,119|
| Publisher quantity cells, kept separate by assay/representation |1,062,087|
| Ordered pooled Table3 slots |3,095,695|
| Original acquisitions |36|
| Material contexts / links |73 /72|
| Exact specimen/aliquot links |0|

All21,082publisher MAPP row IDs/sequences reconcile to originals. Ten exact run
columns plus the aggregate intensity column agree on231,902cells. Two unmatched
publisher run names remain unresolved for42,164cells; no aliases are manufactured.
The earlier210,820cell result counted only the ten run columns, not the aggregate.
The typed source contains both published and omitted original rows.

Source schema25/catalog31 exposes16queryable relations through EvidenceStore.
Reopening checks pinned original bytes and replays every typed relation, detecting
mutations even after inventory rehashing. Independent readers additionally compare
23,721,081original TSV fields, all1,113,756search intensity cells, all3,851,119publisher
cells using openpyxl against the OOXML producer, all3,095,695pooled slots, original
acquisition metadata and source taxonomy/disposition rules. All202date-styled gene
cells retain their original serial/style rather than guessed symbols.

## Registered baseline candidate recipe

`goldberg2025_baseline_candidates_v1` retains every original peptide-intensity
cell decision, including aggregate/other-arm cells outside the baseline scope.
Candidate conditions are positive finite signal, canonical sequence, blank source
reverse/contaminant flags, human-only source header annotations, and an original
control acquisition in the captured-peptide assay. Publisher inclusion is never a
filter. Confidence and matching method are exposed for a later declared policy.

| Candidate disposition | Cells |
|---|---:|
| All original peptide intensity cells |394,732|
| Baseline control cells |182,184|
| Positive baseline review candidates |94,034|
| Excluded baseline cells |88,150|
| Aggregate/other-arm cells outside scope |212,548|
| Candidate cells omitted from publisher view |19,058|

Among candidates,58,676cells are labeledBy MS/MS and35,358By matching. Applying a
possible peptide-levelPEP<.01 policy would leave50,921direct-MS/MS and20,919matching
cells; this is a descriptive confidence axis, not a retrospectively selected fit
or a new study-levelFDR assertion.22,194candidates havePEP>=.01. Candidate status
is deliberately not final processing-model admission: full-reference support,
physical precursor pairing, confidence policy and training eligibility remain
separately unresolved. No split was created.

Every394,732decision and94,034candidate value/status was independently reconstructed
from the original archive and deposition workbook. Candidate-only output equals
exactly the candidate subsequence of all decisions. Source contributor IDs, parents,
raw and log10positive quantities, confidence and original context are preserved.
Recipe verification snapshots source tables and recomputes decisions, including
checks against tampering after output inventory rehashing.

## Runtime and verification

| Phase | Wall time | Peak process RSS |
|---|---:|---:|
| Source build plus internal original-source replay |190.29s|1.80GiB polled|
| Independent source/workbook replay |41.91s|1.93GiB|
| Candidate build, source/recipe replay and independent decision audit |124.60s|2.32GiB|

Candidate phase includes120.78sbuild/replay and3.83sindependent decision checking.
Combined source package, candidate package and audit outputs occupy1,385,928,868
logical bytes at phase closure, below10GiB. All6GiB/20minute gates pass. Local CPU,
one numerical thread; these are single-run resource observations, not throughput
benchmarks. Original assets add no downloads and are outside the output-size tally.

Nine focused importer/recipe tests pass; the earlier source/context/query subset
passed26tests before real source compilation. Initial fixture/catalog failures
were fixed before source launch: an undeclared common source_field, and a fixture
expecting absent XML<value> where the original had an empty element. The reader
retained that distinction. The initial full suite passed3481tests and failed seven
assertions expecting global catalog30 after the intentional31bump; those assertions
were updated without weakening their exact-version checks. The guarded script's
sandbox memory probe failed; its host run kept the memory guard enabled. The final required ./test.sh run passes **3,489non-integration tests** in321.82s;
format/lint pass. Full-corpus integration remains a release/CI check, not a local
claim. No test guard was bypassed.

No training, pretraining, optimizer, loss terms, synthetic biological labels, assay
heads, held-out metrics or prediction dumps apply: this experiment qualifies source
and candidate evidence. Fixtures are software tests only. Existing biological test
sets remain closed. Full-parent, selection/detection, calibration and independent
APC population constraints remain unresolved.

## Artifacts, reproduction and next step

- Original input directory: `data/mapp2025_source/2026-10-24_2130`.
- Verified source: `data/mapp2025_source/2026-10-24_2210_canonical`.
- Verified candidates: `data/mapp2025_source/2026-10-24_2210_baseline_candidates`.
- [Source manifest](results/package-manifest.json), [recipe manifest](results/recipe-manifest.json),
  [independent source audit](results/independent-source-replay.json),
  [candidate decisions/coverage audit](results/candidate-admission.json),
  [resource receipt](results/build-resource.json).
- Exact commands: `reproduce/launch.sh`, `reproduce/audit.sh`, `reproduce/candidates.sh`.
  Frozen source, base commit, dirty diff, package versions and non-default environment
  are in `reproduce/launch.json`, `reproduce/source/` and `reproduce/candidate-source/`.
  Output destinations are immutable; a rerun requires a new generation and receipt.
- [Hitlist825](https://github.com/pirl-unc/hitlist/issues/825) remains open for unresolved
  reference/selection/material and pooled-run metadata recovery. Source ingestion
  does not close those requirements. [Hitlist PR827](https://github.com/pirl-unc/hitlist/pull/827), commit
  `176a62e3e86a98d986d80fcb7dd3120da5e7f92d`, is published as a draft. CI, merge and
  verified release remain pending.
- [Next prospective biological comparison](../agents/codex/plans/mapp2025-baseline-prediction.md):
  a simple sequence score versus length/uniform controls on conditional baseline
  compositions, followed by matched exact versus discrete REINFORCE objectives.
  This requires a new registered experiment and qualified component support; it
  does not require inventing missing parent sequences or treating captured signal
  as intracellular concentration.

Publication note: the executed build launcher is preserved under reproduce/source/build.py.
The canonical code/build.py received publication-only formatting and an equivalent
dict-constructor-to-literal lint correction after execution; original logs and
executed bytes remain unchanged.

Publication recovery: Git whitespace checks treat valid blank context lines in
literal .diff artifacts as trailing whitespace. Both patches are now published
as lossless .diff.gz archives with exact round-trip hashes; original patch bytes
remain locally unchanged. The failed publication stopped before commit/push.
