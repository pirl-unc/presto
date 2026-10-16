# Molecular MHC species and native role audit

Codex / GPT-6, 2026-10-16. Clean executed recipe
`ec0446fcb820d72d0041067cbee6074882837b3f`, Presto 0.1.19 candidate.
The historical fit producer remains Presto 0.1.16 / Hitlist 1.78.0.

## Result and interpretation

All 9,053 selected observations retain their exact original membership, order,
source identifiers, citations, cell/condition/assay context, MHC allele sets,
measurements and censor operators. Molecular MHC identity now follows the explicit
resolved allele set, independently of host or processing context.

| Check | Result |
| --- | ---: |
| Selected train / validation / test observations | 7,241 / 906 / 906 |
| Canonical collated molecular MHC labels verified | 9,053 human |
| Auxiliary labels changed from host-derived murine to molecular human | 8 |
| Changed train / validation / test labels | 7 / 1 / 0 |
| IC50 values, operators, partitions and effective float32 targets unchanged | 2,614 |
| Native context status: resolved / partial / unknown | 50 / 47 / 8,956 |

The eight changes are HLA-B*27:05 elution observations from
[PMID 29393594](https://pubmed.ncbi.nlm.nih.gov/29393594/),
[DOI 10.1002/pmic.201700249](https://doi.org/10.1002/pmic.201700249),
[deposit PXD007756](https://www.ebi.ac.uk/pride/archive/projects/PXD007756).
Their recorded host is rat; the molecule remains human HLA. Native cell labels
and reviewed context evidence are retained verbatim, including unresolved conflicts.
No IC50 observation belongs to this changed subset.

| Native species axis | Human | Mouse | Rat | Unknown |
| --- | ---: | ---: | ---: | ---: |
| Recorded host | 9,045 | 0 | 8 | 0 |
| Reviewed presenting species | 109 | 0 | 15 | 8,929 |
| Reviewed in-vivo host species | 56 | 22 | 15 | 8,960 |

Recorded source-organism and MHC-species fields say human for all 9,053 rows.
These fields do not resolve the separate presenter, host or antigen attribution.
In particular, the 15 reviewed rat presenters do not match the eight rat-host
records one for one. Processing model conditioning and antigen-source attribution
remain follow-ups under #115/#46. Unknown donor/specimen/acquisition identities,
shared studies and the failed IC50 quality screen remain limitations.

## Contract and evidence

This is a CPU data audit, with no model, MHC sequence reconstruction, pretraining,
optimization, synthetic data, new predictions or full-input qualification.
It includes every frozen train, validation and test row for provenance checks;
no new validation/test performance metrics or preferred predictor are produced.
The historical fit, checkpoints, predictions, loss weights and qualification
contract remain unchanged. IC50 uses the historical explicit float32 clipped
log10(nM) policy; raw nM values and censor operators remain separate.

The prospective output contract is schema 3, binding
`explicit_allele_identity_consensus_v1`. Unknown, mixed or unresolvable molecular
species are masked rather than inferred from a host. Native role provenance enters
membership/support fingerprints and prediction exports, not model input features.
The audit verifies row auxiliary labels; no new per-instance MIL auxiliary loss is
introduced. Generated examples preserve per-field parent consensus and parent links.

[Download the complete audit](https://github.com/pirl-unc/presto/releases/tag/molecular-species-lineage-audit-v1).
The publication contains 14 assets: the full 9,053-row source/context table, eight-row
change table, audit/policy JSON, frozen recipe and species policy module, reproduction
commands, independent verifier, publication recipe, plan, manifest and SHA256SUMS.
The full table includes all original membership fields and native source evidence.
`results/audit-receipt.json` locates the shared datacache generation and hashes;
`results/publication.json` and `results/public-verification.json` record publication
and independent public-download verification when available.

| Input | SHA-256 |
| --- | --- |
| Native training.parquet | f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd |
| Frozen membership.tsv | 8959e60c0d9b8049aec25095b9c354d456b48f97ba8fb31d7b786e2861113e99 |
| Historical ic50-inspection.tsv | ab0091db1284b72418a9eed16bb956194323f68154948fe04b5a20e993c8ab6c |
| Historical source-species.tsv | 935cc99a385f06ea083ec23d67ab9a8c9ecc04fdc17beeef1642fbd0746c8f2e |

The native parquet comes from Hitlist's `human-cohort-corrected-v1` release;
the other three inputs come from Presto's `effective-target-fit-qualification-v1`.
Exact URLs and sizes are in `results/audit.json`. Inputs, derived generations and
public verification downloads use datacache; no historical generation is replaced.

## Reproduction and resources

Run `reproduce/launch.sh` from the pinned checkout with its dependencies installed.
The actual local invocation used shared Python 3.12.6 and isolated imports via
`PYTHONPATH=/private/tmp/presto-species-import:/private/tmp/hitlist180-wheel-import:/private/tmp/hitlist175-wheel-import`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. Hitlist 1.80.0 supplied runtime helpers;
it is not the historical corpus producer. Datacache 1.27.1, mhcgnomes 3.64.4,
numpy 2.5.3, pandas 2.3.3, pyarrow 25.0.1 and torch 2.14.1 were observed.
No cloud or GPU was used. Native parquet was streamed in 5,000-row batches.

Builder time was 128.576 seconds, excluding source acquisition/imports; process peak
RSS was 631,259,136 bytes. The independent verifier imports no Presto helpers:
it checks downloaded hashes, all original membership/species columns, every exposed
native column, explicit HLA identities, changed-label partitions and IC50 raw values,
units and censor operators. The original audit additionally checks effective targets
and canonical collation against the unchanged historical inspection table.
