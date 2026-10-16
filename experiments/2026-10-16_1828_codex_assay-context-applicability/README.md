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
without importing Presto. Exact commit/runtime/hash receipts follow execution.

All raw and derived data stays in shared datacache generations. Public artifacts
will include both complete source-linked inspection tables, split/assay coverage,
policy, original source pins, immutable recipe, verifier and hash manifest.
