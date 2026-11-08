# BA projection lineage and complete-development embargo continuation

Date: 2026-11-08
Owner: Codex / GPT-5
Status: specified after the frozen primary common-loader preflight failed; no GPU fit launched

## Trigger

The frozen primary preflight for experiment
`2026-11-08_1800_codex_unified-parity-retrain` passed the exact reference
manifest on 1,560,770 rows, then failed at byte zero of the first BA file:
`missing original measured-outcome lineage; prepare row dispositions first`.
The projection preserved physical archive lineage but omitted the canonical
`provenance.measurement_lineage` mapping required by `ObservationIndex` to
identify and reject duplicate measured outcomes across sources.

This is a projection defect. Do not weaken the common loader or synthesize a
protocol identity.

## Correction

1. Add a deterministic original-measurement identity to every projected BA
   observation:
   - namespace: the exact source archive SHA-256;
   - measurement ID: SHA-256 of the archive-native source key
     `(archive, peptide, identity, context)`;
   - replicate ID: empty because the archives expose no replicate identifier.
2. Retain the full physical member/fold/line-number lineage already recorded in
   `archive_lineage`; identical rows collapsed across folds remain one logical
   measurement.
3. Extend the development embargo from the affinity-only file to the complete
   exposed Stage B development union:
   - affinity development: exact peptide and shared 9-mer core;
   - presentation development: exact peptide, shared 8-mer core, or same-length
     Hamming distance one, matching the existing Stage B separation contract.
4. Reproject from the same pinned archives and catalog into a new immutable
   `prepared-v2` directory. Never edit the completed v1 artifacts in place.
5. Preserve v1 plus the failed common-loader receipt as audit evidence. Record
   v1-to-v2 dispositions and hashes.

## Verification and continuation

- Unit-test measurement-lineage stability across duplicate source folds,
  distinct archive namespaces, exact/core embargoes, and presentation
  core-8/Hamming-1 embargoes.
- Run and independently verify the corrected projection on Modal CPU with no
  model, outcome, prediction, TEST, locked cohort, network, or GPU access.
- Freeze a new retrain-preflight continuation contract pointing only to the v2
  BA hashes; do not mutate the already executed primary launch contract.
- Rerun both full canonical loader passes and independent verification. H100
  training remains prohibited until both corrected passes succeed.
