# Assay-context applicability on frozen native observations

Codex / GPT-6, 2026-10-16. Presto #126; Hitlist #678/#680.
This is an additive CPU metadata audit, not model fitting or evaluation.
No optimizer, losses, synthetic labels, GPU or predictions are created. The
previous train/validation/test IDs and all source/target fields are preserved.

## Prospective contract

Use the 9,053 rows and 2,614 IC50 rows in the published native-presenter-context
source audit. Verify source method and evidence modality by an exact observation
ID join to the pinned 4,071,463-row corrected native export. Add raw source method,
raw evidence kind, explicit empty generated-parent metadata and the new canonical
applicability/basis/availability fields. No original field may change. The IC50
rows retain raw and historical/prospective effective targets byte-for-byte.
Expected IC50 preparation counts are 2,533 purified MHC and 81 cellular MHC.

A missing presenter is not a missing requirement for a purified-MHC binding assay.
Explicit cellular MHC and elution evidence establish applicability; absent or
contradictory preparation stays unknown. The derived fields are inspection
metadata only, bound in schema 5 and the prospective membership/support receipts.
They do not grant broader output support or biological independence.

The historical producer remains Presto 0.1.16 / Hitlist 1.78.0. Its failed IC50
quality result, old schemas, executed recipes, membership and predictions remain
unchanged. This audit does not perform new complete model-input qualification.

## Reproduce

Run `python code/audit.py` from this family in the recorded clean checkout with
Presto 0.1.22 candidate and datacache 1.27.1. `reproduce/launch.json` records the
invocation. `code/verify.py` independently checks the original source joins,
all unchanged table cells, each applicability result and aggregate counts,
without importing Presto. Exact commit/runtime/hash receipts are in `results/`.

All raw and derived data stays in shared datacache generations. Public artifacts
include both complete source-linked inspection tables, split/assay coverage,
policy, original source pins, immutable recipe, verifier and hash manifest.

## Results and decision

The clean executed recipe is `18d04533d3ef1f042c49910042b96022627aee3f`;
its source snapshot and all 11 implementation-file hashes are retained. The
builder took 2.980692 seconds, peak process RSS 401,276,928 bytes; imports and
acquisition are excluded from this runtime. Local CPU, Python 3.12.6, one
Torch/BLAS thread; exact library versions are in `results/audit.json`. Actual import overlays and
thread limits are recorded in `results/execution-environment.json`.

| Verification | Result |
| --- | ---: |
| Native observation-ID joins and unchanged prior rows | 9,053 |
| IC50 rows with all raw/effective targets unchanged | 2,614 |
| IC50: cellular processing context not applicable (purified MHC) | 2,533 |
| IC50: cellular processing context applicable (cellular MHC) | 81 |
| Core artifacts independently hash-verified and replayed | 11 |

Every IC50 presenter remains unknown. This is a real processing-context gap for
81 explicitly cellular observations; it is not a missing cellular requirement
for the 2,533 purified-MHC observations. Unknown preparation elsewhere remains
unknown, and an endpoint name alone never establishes preparation. Coverage is
available by unchanged partition, source modality, method and applicability.
The verifier uses original/native source joins and independent classification,
without importing Presto. All prior table cells and label policies are retained.

[Download the evidence release](https://github.com/pirl-unc/presto/releases/tag/assay-context-applicability-audit-v1).
`results/publication.json` records each public URL, SHA-256 and byte size;
`SHA256SUMS` is included in the release. Data acquisition, generation and public
verification use shared datacache generations. Full source context, citations,
allele sets, cell conditions and assays remain in `observations.tsv` and `ic50.tsv`.

This is the preferred annotation policy for prospective inspection. There is no
new model condition, synthetic data, loss/optimizer, pretraining, fitting or
winner. Validation/test partitions are inspected only for preservation and
coverage; there are no new predictions or predictive metrics. Historical
Presto 0.1.16 / Hitlist 1.78.0 targets, model results and failed IC50 quality screen
remain unchanged. All 9,053 donor/specimen/acquisition identities remain unknown;
source-study overlap and incomplete antigen/repertoire roles remain limitations.

Public verification: all 21 release assets were independently downloaded and
hash-verified through datacache; the downloaded verifier replayed every native
join, original table cell, applicability annotation and unchanged IC50 target.
See `results/public-verification.json` and `results/publication.json`.
