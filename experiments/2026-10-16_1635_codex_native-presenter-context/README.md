# Native presenting-cell processing context audit

Codex / GPT-6, 2026-10-16. Successful clean recipe
`060fbfed9d2f99f2937e637624a9e88d1df786c3`, Presto 0.1.21 candidate, based on
merged 0.1.20. The historical fit producer remains Presto 0.1.16 / Hitlist 1.78.0.

## Result and interpretation

The canonical native consumer now uses the explicit presenting-cell species for
processing context. Recorded host, antigen source, molecular MHC, in-vivo host and
lineage host remain independent source fields. No fallback fills an unknown
presenter from another role. The source-linked tables preserve every original
observation ID, row order, partition, citation, cell/condition/assay field and allele
set. Molecular MHC labels remain human for all 9,053 rows.

| Processing context | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| Explicit human presenter | 86 | 18 | 5 | 109 |
| Explicit rat presenter (murine model bucket) | 13 | 1 | 1 | 15 |
| Unknown presenter | 7,142 | 887 | 900 | 8,929 |
| All selected observations | 7,241 | 906 | 906 | 9,053 |
| Changed from historical host-based processing bucket | 7,148 | 887 | 901 | 8,936 |

The historical input used 9,045 human and eight rat recorded hosts. The correction
masks 8,929 unsupported presenter assignments and changes seven further rows from
human-host to rat-presenter context. This does not alter any native source claim.
All 2,614 IC50 observations have unknown native presenters; their raw nM values,
operators, partitions and clipped float32 log10(nM) targets are unchanged. The
separate IC50 table exposes these gaps and both historical/prospective targets.

Native context review status remains 50 resolved, 47 partial and 8,956 unknown.
Of the 124 known presenters, 27 retain per-field agreement across candidate arms
while the arm-specific review ID/status/reference is empty. Their original PMIDs
and provenance remain, but no unique arm or arm-specific citation is invented.
This inventory is evidence about source metadata, not biological independence or
successful predictive learning.

All 15 rat-presenter rows come from [PMID 29393594](https://pubmed.ncbi.nlm.nih.gov/29393594/),
[DOI 10.1002/pmic.201700249](https://doi.org/10.1002/pmic.201700249),
[PRIDE PXD007756](https://www.ebi.ac.uk/pride/archive/projects/PXD007756).
The complete table retains the exact native reference sections, reviewed cell and
condition context and unresolved notes for all studies, including human presenters.

## Contract and scope

Schema 4 prospectively binds `native_presenter_and_generated_parent_context_v1`
and the unchanged `explicit_allele_identity_consensus_v1` molecular policy.
Unknown, ambiguous or unsupported native roles use the unknown processing bucket.
Generated records use unanimous parent processing context independently of swapped
MHC. Legacy records without a native role dictionary retain their explicit species
behavior, with that compatibility basis disclosed. Row/MS-MIL/T-cell-MIL routing,
device transfer, generated parents, prediction exports and frozen fingerprints are
covered by focused regressions. The real-data audit checks every canonical row
processing input and molecular auxiliary label; it creates no MHC sequence inputs
or new MIL training examples.

No model, optimization, pretraining, synthetic training data, predictions or new
full-input qualification is produced. There are no new validation/test performance
metrics, loss weights or preferred predictor. All three frozen partitions are
inspected for provenance only. The earlier IC50 quality screen still fails and its
checkpoints/predictions/selection remain unchanged. Shared studies and unknown
donor/specimen/acquisition identities still prevent a biological-independence claim.
Antigen attribution, repertoire roles and legacy sampler/API semantics remain under
#115/#46/#123. No MHCflurry work is included.

## Download and provenance

[Download the complete audit](https://github.com/pirl-unc/presto/releases/tag/native-presenter-context-audit-v1).
The publication has 18 assets: complete/changed/IC50 TSV tables, audit and output
policy, successful executed recipe and species policy module, reproduction commands,
independent verifier, publication recipe, plan, manifest/SHA256SUMS and the preserved
initial failed recipe/log/receipt. Large tables live in shared datacache; small
receipts and summaries are retained in this experiment directory.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| presenter-context-observations.tsv | 10,513,967 | feb29f9a60b8e8ae5dc6a3af0477a25ade615e370f0d28044e729dfcde11c11e |
| changed-processing-contexts.tsv | 10,332,030 | 5b877033a5eded0ac4a57f1a7f739612665f2cbe62a0b4085459fe2dcfed7210 |
| ic50-processing-contexts.tsv | 3,046,490 | cb74c379457e07eafdc91adc3a0256c0d8bfad3ec3ba2c70933bafdf1f6d6f69 |

The pinned original native training parquet comes from Hitlist's
`human-cohort-corrected-v1`; original membership, IC50 inspection and source species
come from Presto's `effective-target-fit-qualification-v1`. The previous source-role
table comes from `molecular-species-lineage-audit-v1`. Exact URLs, sizes and hashes
are in `results/audit.json` and the executed recipe. Every exposed native column was
independently compared directly with the original parquet. Every previous membership
and role-table column was retained. All acquisition, generation and public download
verification uses datacache under `~/.hitlist/presto-presenter-context-audit`.

## Reproduction and resources

Run `reproduce/launch.sh` from the frozen clean recipe commit. The actual invocation
used Python 3.12.6 with
`PYTHONPATH=/private/tmp/presto-presenter-import:/private/tmp/hitlist180-wheel-import:/private/tmp/hitlist175-wheel-import`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Hitlist 1.80.0 supplied runtime helpers;
it did not produce the historical dataset. Observed dependencies: datacache 1.27.1,
numpy 2.5.3, pandas 2.3.3, pyarrow 25.0.1, torch 2.14.1 and mhcgnomes 3.64.4.
No cloud or GPU was used. Native parquet was streamed in 5,000-row batches.

The successful builder took 53.443 seconds, excluding source acquisition/imports;
peak process RSS was 707,395,584 bytes. The independent verifier imports no Presto
helpers. It rechecks all selected native columns, original membership/roles,
source-linked raw/effective targets, processing mapping, counts and policy. The
local verification passed; public-download receipts are added after publication.

## Preserved failed attempt

Initial recipe `fb39e8d302ef77a1b5a3160eb3fe442398418cf0` stopped with a
copyfile argument-count error while snapshotting code (#125). It produced no atomic
generation or public release. `code/audit.py` remains byte-for-byte intact;
`code/audit_v2.py` corrects packaging and records that failure. Executing both actual
snapshot calls in isolation reproduces the initial exception and verifies the
corrected files byte-for-byte. The initial recipe, log and receipt are included in
the public evidence. Only the successful v2 run is reported as completed.
