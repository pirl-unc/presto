# Native presenting-cell species for processing context — next part of #115

## Problem and scope

The molecular auxiliary is independent in 0.1.19, but canonical row and MIL
processing inputs still read sample.species, populated from native recorded host.
Hitlist's arm-scoped primary-literature registry separately supplies
presenting_species. In the frozen 9,053-row cohort it supplies 109 human and 15 rat
presenters, leaving 8,929 unknown. Host, in-vivo host, earlier lineage host, MHC
origin and antigen source must not be substituted for that role. Some native
presenter facts survive agreement across candidate arms even when arm-specific
review identity/status is empty; a missing review ID does not negate that explicit
per-field consensus. Preserve the native evidence and its attribution separately.

This PR fixes the native consumer's processing context. Antigen-source and
repertoire-relative foreignness, independent repertoire/TCR schema, legacy sampler
host grouping and source biological independence remain explicit limitations.
No source measurement, historical fit or past qualification is relabeled.
#123 tracks first-substring species parsing; add a conservative bounded-alias
single-organism helper for native roles while retaining the legacy parser contract.

## Design

1. Add one processing-context resolver alongside the existing molecular resolver.
   Native records use only their explicit presenting_species; empty, unsupported
   or ambiguous values use the existing unknown model context. No host, MHC or
   antigen fallback. Preserve a fine reported presenter plus the normalized model
   bucket, basis and status in inspection/lineage. Legacy records lacking the
   native role dictionary retain their explicit processing/species API behavior;
   disclose that compatibility basis rather than presenting it as reviewed evidence.
2. Use the same resolver for row model inputs, MS MIL instances and T-cell MIL
   instances. Keep it on device transfer and prediction exports. Do not alter
   sequence encoders, molecular labels, raw source host fields or assay targets.
3. Generated records retain per-field parent consensus for processing context,
   independently of any swapped allele. Conflicting or unknown parents stay
   unknown. Store that parent-context basis in the existing lineage dictionary
   and ledger so it cannot be invented during later collation. Retain all parent
   identities and original native roles; do not infer presenter from a donor MHC.
4. Include derived processing context/basis in final membership and support
   fingerprints, and bind a named prospective policy in schema 4 and source
   contracts. Reject omitted/stale policies even if a caller omits them from both
   manifest and source contract. Update the authoritative I/O status and training
   specification together, with the remaining antigen/repertoire limits explicit.

## Verification and actual evidence

- Native human-HLA/rat-presenter, human presenter/mouse in-vivo host, missing
  presenter, invalid or mixed presenter and legacy explicit context cases.
- Row/MIL/device/export parity, generated-parent consensus and conflict, no host/MHC
  fallback, unchanged molecular/IC50 targets and source-role fingerprint sensitivity.
- A separate registered data-only audit from pinned public sources and original
  membership: verify all 9,053 row IDs/order/partitions and 2,614 IC50 raw/effective
  targets; publish old/new model context, exact source evidence, role gaps and hashes
  through datacache. No new predictions, optimization or held-out selection.
- Base the final clean audit on merged 0.1.20; freeze its actual commit before
  execution. Version 0.1.21, formatting/lint/docs/focused/full CI, review, merge,
  exact clean-main release and independent public-wheel verification remain gates.
