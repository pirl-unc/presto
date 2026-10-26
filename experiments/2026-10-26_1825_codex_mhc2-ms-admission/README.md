# MHC-II MS and context admission

Status: source acquisition/checksum verification passed; training admission pending.
Codex / GPT-6, 2026-10-26.

[Prospective plan](../agents/codex/plans/2026-10-26_mhc2-ms-admission.md).
Downloaded the canonical corrected human training table, native manifest,
qualification report and checksums. All three content files match the published
SHA256SUMS;188,213,823content bytes,4,071,463rows/161columns. Reproduction uses
`reproduce/launch.sh` followed by the existing Python environment executing
`code/verify_source.py`; receipts and the verification source snapshot are retained.
Raw bytes remain at `/private/tmp/presto-mhc2-20261026/ms-source/`.

The upstream qualification states that the Oct16 export is human source/MHC and
lengths8–25. It has no resolved donor/specimen/acquisition identities and claims
no real flanks or unique parent proteins. This source cannot alone satisfy the
full class-II length range, parent-context or independent-sample requirements.
The older fullv4 export retains longer peptides; extending compatible canonical
context admission beyond this published cohort remains necessary.

Two source-row inspections confirm usable distinctions: Stražar's exact allele
attribution retains a condition ID and HLA-DM transfection; MAPTAC's unresolved
DM arms retain empty conditions and `pmid_ambiguous` attribution. Neither should
be overwritten with guessed conditions. `matched_sample_count` is the number of
curated study samples, not the number of uniquely matched arms for the row:
Stražar shows42despite exact attribution. Current Hitlist `export.py` confirms
the PMID-level count; do not require that field to equal1for unique attribution.

No source observations have been admitted for model training yet. No generated
negatives, fitted model, validation/test split or predictive improvement is
claimed. The full binding/MS/pathway goal remains active; next work qualifies
actual candidate pairs, source-row identity and condition evidence and declares
the predictive comparison before fitting.
