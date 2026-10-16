# Honor and record Hitlist cap selection (#103)

The canonical Hitlist route ignores cap_sampling and retains all eligible record
objects before random.sample. Add explicit uniform_list/head/reservoir policies.
Keep the public Python loader default uniform_list for exact historical seeded
selection compatibility; canonical training passes its requested head/reservoir
mode. New head/reservoir collectors retain at most the requested cap per assay
family while still consuming all eligible rows for truthful pre-cap/drop counts.
This bounds record-object retention, not the already loaded projected dataframe.

Each family owns an independent Python Random stream initialized with the stated
sampling seed. Reservoir uses Algorithm R, head retains first eligible records,
uniform_list preserves the old random.sample order and selection. Record the
actual algorithm/version, mode, seed and per-family stream semantics in loader
stats/data funnel. None/zero/negative caps retain historical unlimited semantics.
Reject an unknown mode before reading inputs. Keep all existing endpoint/unit/
censoring, condition/MHC-role, mapping and source-lineage admission unchanged.

Verify actual selection, complete eligible/drop counts, repeatability and a seed
change, retained original identities/assay fields, bounded live object count,
legacy compatibility and canonical argument propagation. Run format/lint/full
suite and CI; bump0.1.12 and ship via a separate PR after predecessors. The
original human preflight remains an explicitly documented uniform-list sampling
result, despite its recorded reservoir flag; do not rewrite its history.
