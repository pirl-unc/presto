# Molecular MHC species and source-role provenance — first part of #115

## Problem and release boundary

The frozen 0.1.16 fit includes eight rat-host/HLA observations. Canonical MHC
species supervision currently reads processing_species, and coverage prioritizes
host over molecular allele identity. The native source already preserves reviewed
presenter/host/transgene roles, but the consumer drops those fields. This changes
an auxiliary label and makes human inspection misleading.

This independently reviewable PR corrects the molecular auxiliary and carries all
native role evidence end to end. It does not claim to resolve source-protein
attribution or redesign processing/repertoire conditioning. Those broader parts
of #115/#46 remain explicit; no historical fit, public asset or contract is edited.
No performance claim or new optimization is part of this PR.

## Design

1. Introduce a small species-provenance module with a named mapping from native
   Hitlist columns to inspectable consumer fields. Preserve reported host,
   antigen organism, native MHC identity/basis/disagreement, and every reviewed
   species_context/presenting/in-vivo/lineage-host/introduced-MHC/candidate/evidence
   field. Keep unknown values empty and keep original names/notes/citations.
2. Append a species_context dictionary to typed assay records and PrestoSample,
   preserving positional constructor compatibility. Populate it on Hitlist
   ingestion and pass it through the existing record-to-sample and generated
   lineage routes. Flatten its declared fields in collated lineage, held-out CSV
   and inspection reports. Never feed provenance identifiers/notes into a model.
3. Derive the MHC species bucket from explicit molecular allele identity via the
   existing mhcgnomes resolver. A same-species genotype bag can resolve one bucket;
   missing, unparseable or conflicting identities remain masked. Do not use
   host, processing or antigen organism as a substitute. Preserve fine identities
   in provenance even though the existing head has six coarse buckets.
4. Add a dedicated collated molecular-species field, preserve it on device
   transfer and MIL parent prediction exports, and make the canonical row supervision target
   and mask read it. Use the same resolver in MHC sequence coverage so a rat host
   expressing HLA is counted as human MHC. Keep the host/processing role unchanged
   in this bounded correction and document the remaining conditioning limitation.
5. Include all retained axes in frozen consumer fingerprints and support context
   facets. Prospective artifacts must record changed auxiliary labels and source
   schema, not silently accept an old qualification fingerprint. Membership and
   measured endpoint values/units/operators remain unchanged.

## Verification and actual data

- Human HLA with rat host, ordinary human, same-species multi-allele bags, mixed
  species, unparseable and sequence-only unidentified MHC cases.
- Record/sample/collator/device/MIL/prediction round trips; source-field retention,
  no provenance feature leakage, generated-parent preservation and fingerprint
  sensitivity to one changed axis.
- Source-linked data-only replay of all 9,053 published selected observations from
  pinned native parquet and membership through datacache. Verify exact IDs/order/
  partitions and all 2,614 measured IC50 labels; publish old/new molecular labels,
  distinct host/antigen/presenter/MHC axes, citations, context gaps and hashes.
- Preserve the original 0.1.16 run and declare remaining processing/antigen
  attribution limits. No new fit or reused-test model selection.
- Version 0.1.19; rebase onto merged #119 before freezing the executable audit.
  Format/lint, relevant regressions, full CI, review, merge, clean-main GitHub
  distribution and public-wheel verification are required.
