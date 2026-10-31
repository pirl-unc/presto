# Qualified purified multi-assay cohort

168,210 observations; frozen original27,364-row IC50 cohort plus140,846 outcomes.
The adjacent manifest lists every local JSONL source and its SHA256. Large data
and the full preparation ledger are retained locally, excluded from Git.

Rebuild using the pinned source/cache locations and checksums in
`experiments/2026-10-27_2235_codex_shared-support-audit/reproduce/source/prepare_data.py`.
Run the independent verifier in that experiment before any fitting. Its README
records source, units, assay identity, complete human MHC partners and split policy.
Original IC50 source ordinals are one-based; new source ordinals are zero-based.
No original observations were rewritten to reconcile this convention.

The scale comparison excludes the10off-rate training-only rows. Its frozen
six-output manifests use168,200 rows:134,561training,17,503validation,16,136test.
All data are measured; there are no synthetic negatives or crop labels.
