# Actual human-corpus frozen-input preflight

Status: execution, analysis and ten public artifact downloads verified; software
release gates remain tracked in the PR. Agent/model: Codex (GPT-6). [Prospective plan](../agents/codex/plans/human-corpus-preflight.md).

The canonical consumer passed a data-only preflight and froze 9,000 actual model
samples. This does not establish training readiness: no model was constructed,
no predictions exist and all 221 output requirements remain undeclared. The
measured support and limitations below determine what can be tested next.

## Source and reproducibility

Input: [human-cohort-baseline-v1](https://github.com/pirl-unc/hitlist/releases/tag/human-cohort-baseline-v1),
native manifest SHA-256 `039efc7854098951b5c10b3fb5b38294452eeb425e367ed1fafdeebddf34b38d`.
The 4,071,463 observations retain contributors, citations, cell types/conditions,
MHC candidate sets versus restrictions, endpoints, units and censoring. Human
MHC, human-reported peptide source, lengths 8–25. Producer Hitlist 1.76.0 predates
curation fixes #694/#696; original export release dates are unverified. Donor,
specimen and acquisition identity are unresolved for every source observation.

Successful clean source `2c73f2fb57c5d47c09c492e1f55d1b3ccecfe312`, tag
`experiment-human-preflight-20261016`; Presto 0.1.10 wheel SHA-256
`78dc67717df08fc9e234fa57586312da8ae9d7782336194dcc284a2a262b1c7f`.
Run `reproduce/launch.sh` from that tag. `reproduce/launch.json`, source snapshots,
hash-locked Hitlist environment, consumer constraints and official MHC FASTA
commit/URL/hash pins preserve the exact contract. No live assay files or flanks.
Both MHC classes eligible; strict MHC sequence filtering. No synthetic/generated
controls, UniProt, MHC-only, or bulk-MS augmentation. All regularizers explicitly
zero; no pretraining, optimizer or fit. Output mapping follows original affinity
endpoints, half-life minutes converted through the documented hours record,
temperature and off-rate units, and unchanged exact/lower/upper censoring policy.
Elution is measured presentation evidence, not MS detectability or excision.

CPU-only Modal: eight CPUs, 64 GiB memory ceiling, 512 GiB scratch, one Torch/BLAS
thread, 45-minute work alarm. Actual runtime 1,950.43 seconds; runner peak RSS
21,010,509,824 bytes (children excluded). Full dependencies, installation report,
pip check, runtime and source receipts are archived. MHC catalogs were rebuilt
with mhcseqs 2.6.13 from official pinned IMGTHLA/IPDMHC FASTAs; complete raw/full
catalogs plus merge/validation reports are included. Datacache shares inputs,
catalogs, partitions and outputs. Earlier failures before data execution are
preserved: disk minimum (#98) and worker import path (#99).

## Actual selection and support

Protocol deviation (#103): the requested `cap_sampling=reservoir` flag was ignored
by this historical Hitlist route. Actual selection was independent per-family
`random.Random(59).sample` over full eligible lists (data seed42 plus canonical
offset17). The original configuration and selected membership remain unchanged.
The future fix is separate; these results must not be described as Algorithm R.

| Family | Eligible before cap | After cap | After MHC filter |
| --- | ---: | ---: | ---: |
| Binding | 13,190 | 5,000 | 4,490 |
| Elution | 3,909,515 | 5,000 | 3,805 |
| Kinetics | 19 | 19 | 19 |
| Stability | 741 | 741 | 686 |

Strict sequence filtering also removed 3,489 unresolved allele occurrences within
elution bags. Original and resolved sets remain separate in membership metadata.
Coverage denominators count allele occurrences, not unique samples; species
buckets prioritize assay-host context. Nine murine-host occurrences coexist with
human HLA; every retained resolved allele begins `HLA-`.

| Target / samples | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Samples | 7,200 | 900 | 900 |
| Binding, all endpoints | 4,121 | 177 | 192 |
| Direct KD | 506 | 0 | 4 |
| Direct IC50 | 2,334 | 104 | 137 |
| Direct EC50 | 88 | 5 | 6 |
| KD (~IC50) proxy | 1,056 | 62 | 37 |
| KD (~EC50) proxy | 137 | 6 | 8 |
| Elution positives | 2,447 | 679 | 679 |
| Elution negatives | 0 | 0 | 0 |
| Half-life | 482 | 41 | 24 |
| Temperature | 131 | 3 | 5 |
| Off-rate | 19 | 0 | 0 |

Exact/censored counts by endpoint are in `results/split_support.csv`; full
per-output/family/source coverage is archived. Human-source-only foreignness has
only one class. Positive-only elution cannot qualify discrimination. Direct KD
and kinetics cannot be qualified across these held-out partitions. No thresholds
were changed to create support, and no model/baseline winner exists for this
preflight. Validation/test loss, regression, classification metrics and prediction
dumps are inapplicable because no fitting or inference ran.

## Leakage and biological independence

Frozen component assignment uses seed42, peptide-disjoint 80/10/10 membership:
5,843 components, largest31 observations. Every selected root converts to one
model sample; zero generated samples and zero records without a model sample.
No shared peptides, observation IDs, pMHC identities or known experimental origins
across partitions. Known experimental origins cover only 79 selected observations
(train78, test1); all 9,000 donor/specimen/acquisition identities remain unknown.
Studies overlap: train/test161, train/validation162, validation/test117. These are
peptide-disjoint partitions, not independent-study or biological replication.
The peptide-only model audit is intentionally narrower; use the root lineage
audit for source/identity conclusions.

## Download and inspect

[Public artifacts](https://github.com/pirl-unc/presto/releases/tag/human-corpus-preflight-v1)
include the original run archive, per-file inventory, runtime receipt, readable
summary and checksums. Archive SHA-256
`444ac92e859da9afd3fcdf3c27ee3ae282f70a6134ac9c42f3c528cdcc1dae80`
(7,320,003 bytes); all 46 members verified against their original inventory.

Start with `qualification-summary.json`, `split_support.csv`, data funnel and
root leakage audit. The archive contains TSV/parquet measured assignments and
final membership, parent graph, full output coverage, source-native manifest,
MHC catalog/source pins, label/partition policies, original configs and logs.
Join `source_bundle_id`/`evidence_row_id` to the public source's complete cell,
assay and contributor tables. No raw source rows were silently overwritten.
`analysis/summarize.py /path/to/downloaded/files` verifies original bytes and
recreates the small reports. Local/shared paths remain in results/preflight.json;
portable URLs and independent download verification are in publication receipts.

Decision: preserve this baseline; fix cap-policy reporting, build a separately
versioned corrected corpus, and register target-specific evaluation only where
held-out evidence supports it. Hitlist #678/#680 and Presto #48/#53 remain open.
