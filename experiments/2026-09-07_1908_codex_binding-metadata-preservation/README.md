# Quantitative assay metadata preservation (#49)

Date: 2026-09-07. Agent/model: Codex / GPT-6. Status: complete.
Plan: [source audit](../agents/codex/plans/2026-09-07_binding-metadata-preservation.md).

Run from the repository root:
`bash experiments/2026-09-07_1908_codex_binding-metadata-preservation/reproduce/launch.sh`.
Exact source hash, production-file hashes, before/after loader sources, command,
environment, dependency provenance and analyzer snapshot are in `reproduce/`.

Both conditions scan the same full `data/merged_deduped.tsv` with the actual
canonical adapter. Instrument quantitative constructors before caps; retain
only one head-selected record per modality to bound memory. Compare ordered
fingerprints of every non-descriptor record field, routed counts, loader stats,
descriptor missingness and selected output-column counts. These are source
routing counts before MHC filtering/splitting, not a per-split support census.

No pretraining, training, optimization, augmentation or predictive evaluation
is performed. Validation/test metrics and prediction dumps are intentionally
absent because this is a source/selector audit with no dataset split. Source
qualifiers/numeric values remain unchanged; affinity/stability/kinetic labels
retain existing objectives and units. Kinetic/stability metadata does not add
affinity-panel supervision. Requested GPU: none; observed hardware: local CPU.

## Results

Both actual adapters scanned the identical 1,028,889,889-byte source file,
SHA-256 `46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`.
The routed quantitative row counts, ordered fingerprints of every
non-descriptor record field, and complete loader statistics are **identical**.
Binding assay-family selector counts are also unchanged on this corpus.

| Quantitative family | Routed numeric rows | Missing method before | Missing method after |
|---|---:|---:|---:|
| Binding | 249,292 | 249,292 | 0 |
| Stability | 12,259 | 12,259 | 0 |
| Kinetics | 106 | 106 | 0 |

Previously every preparation/geometry/readout selector was unknown. After
preservation, binding preparation counts are PURIFIED 238,258; CELLULAR 9,508;
LYSATE 17; BINDING_ASSAY 270; OTHER 1,239. Binding readouts are FLUORESCENCE
106,010; RADIOACTIVITY 141,480; OTHER 1,802. OTHER is observed metadata not
recognized as a named vocabulary category; it is distinct from missingness.

All 249,292 binding rows regain their source assay type; its fallback had
already selected the same six populated type columns for this corpus. All
binding culture fields remain absent in this source file; their new preservation
is verified with a fixture and must not be counted as added real support.

Stability and kinetics likewise regain their observed method descriptors.
Their existing batch metadata supports these selectors, but their rows remain
masked from binding-affinity panel losses. This repair does not claim that
kinetic/stability panel columns receive quantitative supervision.

The real **EVMPVSMAK / HLA-A*03:01 / 473 nM / exact** example retains
`dissociation constant KD (~EC50)` and `purified MHC/direct/fluorescence` through
the real MHC resolver, PrestoDataset and collation. Its type/method/prep/geometry/
readout indices are **3 / 2 / 1 / 2 / 2**. The resolved class-I groove sequences,
record fields and complete sample are frozen in
[real_example_sample.json](results/real_example_sample.json).

Artifacts:

- [before.json](results/before.json) and [after.json](results/after.json): all
  source/funnel counts, missingness, per-axis selected-column counts, ordered
  invariant fingerprints and the source example.
- [summary.json](results/summary.json): every selector's before/after/delta,
  numerical conservation assertions and real-example selectors.
- `reproduce/source/`: actual pre/post loader functions and analyzer snapshot;
  execution uses those functions with hash-verified production helpers.

Runtime: 88.080 seconds before, 70.038 seconds after, 159.657 seconds overall.
This is not a runtime comparison: order/cache effects were uncontrolled.
Production code was committed at `ddc06448fc906ea4fd51dc12c670accc137a23b9`;
only experiment/plan/design files were dirty. Real-row resolution used installed
mhcseqs 2.5.12; resolved sequences are retained in the example artifact.

## Decision and handoff

Preserve observed quantitative descriptors at ingestion. The 184 focused
ingestion/collation/loss/routing tests pass, including selected preparation and
readout embedding gradients and fixed-output invariance to observed metadata.
Seven new adapter assertions failed before the repair; all 12 new tests pass
after it. Pinned full lint/format and strict docs build pass.

This closes the specific adapter defect; #48 still needs post-curation,
per-source/column/split coverage and adequate-supervision gates. #53 still needs
fresh training and measured fitting/generalization evidence. No model winner
or prediction-quality conclusion follows from these source counts.

Status: complete. Next step: review/merge #49's repair, then implement the shared
supervision contract for #48/#50/#51. Open questions: complete split support,
unknown biological contexts and quantitative family-specific panels remain
explicit follow-up work.
