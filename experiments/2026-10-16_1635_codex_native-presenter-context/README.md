# Native presenting-cell processing context audit

Codex / GPT-6, 2026-10-16. Registered before execution; Presto 0.1.21 candidate,
based on merged 0.1.20. The historical fit producer remains Presto 0.1.16 and
Hitlist 1.78.0. This audit has not yet run.

## Prospective contract

Use the explicit native presenting_species as the model processing context;
missing or ambiguous presenters remain unknown. Keep molecular MHC, antigen,
recorded host, in-vivo host and lineage host as independent roles. The output
contract advances to schema 4 and binds native_presenter_and_generated_parent_context_v1.
Legacy explicit API behavior is separately disclosed; generated examples use
unanimous parent processing context. Unit tests exercise both MIL routes,
generated parents, prediction exports, device transfer and frozen fingerprints.

## Data and comparison

Audit all 9,053 original observations in original order and frozen 7,241/906/906
train/validation/test membership, with all source IDs, citations, cell types,
conditions, assay endpoints, units, censoring and MHC allele sets intact. Stream
the pinned native human-cohort-corrected-v1 training parquet and cross-check every
selected source field against the public molecular-species-lineage-audit-v1 table.
The three original membership/IC50/species tables come from
effective-target-fit-qualification-v1. Exact URLs, sizes and hashes are executable
in code/audit.py and copied to the audit receipt. All acquisition and derived
artifacts use datacache under ~/.hitlist/presto-presenter-context-audit.

Expected role inventory, already independently established by the previous audit:
109 human and 15 rat presenters, 8,929 unknown, compared with recorded hosts of
9,045 human and eight rat. Check canonical row model inputs and all 9,053 molecular
auxiliary labels, plus every one of 2,614 raw and effective IC50 values/operators.
No peptide, row, partition, native role or historical artifact may be rewritten.

## Execution and limits

Run reproduce/launch.sh from the frozen clean candidate commit. CPU only; no cloud,
GPU, model, MHC-sequence reconstruction, fitting, pretraining, generated training
rows, predictions or new full-input qualification. No optimizer, loss weights or
new validation/test performance metrics apply. All three partitions are inspected
for provenance, without selecting model parameters or revisiting the failed quality
criterion. The earlier IC50 quality failure and unresolved biological independence
remain. Antigen attribution, repertoire roles and legacy sampler/API semantics
remain tracked in #115/#46/#123.

The audit writes complete/changed/IC50 TSV tables and policy/audit JSON. Large
outputs stay in shared datacache and are published with hashes, exact executed
recipe, independent verifier, reproduction commands and this plan. The verifier
imports no Presto helpers and independently rechecks all exposed native parquet
columns, original membership and source roles, target values and presenter mapping.
No original result or release asset is replaced. Publication and execution receipts
will be added after the run; current status is registered.
