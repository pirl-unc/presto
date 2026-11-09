# Original availability-study assets

Verified immutable originals are installed here by datacache. They are source
inputs, not a normalized source registry or an admitted training dataset.
The shared normalized curation belongs to Hitlist.

The initial source audit's URL/size/hash receipts and reproduction command are in
`experiments/2026-10-22_1410_codex_availability-source-audit/reproduce/`.
Its checks fail on changed upstream bytes instead of silently refreshing a source.
Public API responses are retrieval snapshots; future API drift requires an
explicit new receipt. Large originals are ignored by git; quantitative workbook
assets are independently pinned to publisher file IDs and MD5 plus local SHA-256.

Small original API/SDRF snapshots are tracked under `snapshots/2026-10-22/` so
changing API counters and annotations do not prevent reproduction. These are
unaltered source bytes with source attribution in the audit receipts. The launcher
imports those local originals and downloads only the hash-pinned workbooks when
the immutable bundle is absent. This does not refresh the study to current data.
