# Integrate recovered DP/DM records into the shared source path

Date: 2026-11-10. Codex / GPT-6. Presto base 9a99bbf0; Hitlist base 43e2266.
Prior goal turn was progress: recovered evidence, verification and follow-ups
were committed/pushed. This step implements the qualified source, not a new
model architecture. The independent Atlas worker retains ownership of its fits.

## Contract

Use the exact PXD017154 native peptide–sample artifact (SHA256
daaa04962f690a184c553ac8ec55c8545f40bdfbecac61ce89a8997d94124beb).
Package ten biological-source sample CSVs and one separate capture-control
companion. Route observations through Hitlist scan_supplementary, contributor
capture, curated sample joins, generate_training_table and Presto ms_observation.
No custom model branch, synthetic affinity target, new score cutoff or current
cohort mutation. Preserve exact chain pairs, native PSM/forms/files and scoped
additional DM transduction. Register common experimental origin, retaining
unknown physical specimen/donor identity rather than assuming independence.

## Verification and outputs

1. Verify packaged candidates/control rows field-for-field against the pinned
   original extraction, preserving technical replicate support without inflation.
2. Materialize an isolated source-only Hitlist index plus real contributor
   contract and empty unrelated assay indexes. Export with the public common API;
   use an isolated HITLIST_DATA_DIR, never the user's existing dataset.
3. Join source contributor records independently. Admit each exported record
   through the existing shared Presto MS adapter, preserving predict/unpartitioned
   and not-train-eligible markers. Check actual model-context DM present/absent,
   all chain identities, source roles and count conservation.
4. Before training, compare source/peptide identities against prior frozen cohort
   identity indexes. Do not inspect designated held-out outcomes or predictions.
   Report independent additions and overlaps separately; freeze a prospective
   training/comparison contract only after these gates.
5. Run Hitlist format/lint/test gates; inspect package inclusion and version;
   publish a source PR with issue895 evidence and exact-head CI. Follow the
   repository's release workflow when qualified; do not mark the biological goal
   achieved based on software gates or data availability.

Experiment family: experiments/2026-11-10_1535_codex_dp-dm-shared-integration/.
Record source snapshots, exact environment, raw artifact hashes, all exclusions,
wall time and peak RSS where available. No GPU requested, no training or model
selection in the source integration stage, so no predictive validation/test
metrics or prediction dumps apply. Shared/source-specific future limitations
stay in Hitlist895 and Presto155; scientific parent PR131 remains draft.

Attempt01 stopped after source export because the verifier compared a pandas
nullable numpy.bool scalar by Python object identity. The saved export has
21,547 explicit false modification flags and no missing values; compare values,
preserve the failed attempt and run a new output root. No data policy changes.
