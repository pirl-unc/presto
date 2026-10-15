# Frozen observation partitions and generated-parent lineage — Presto #71, part 2

## Integration order

This branch follows the frozen-admission PR (base 0aa7cc2). Land and release that
PR first; rebase this dependent branch on its squash merge. Keep the user checkout
untouched and MHCflurry out of scope. The first PR supplies measured-observation
bundle/contributor identities but does not yet qualify generated controls/splits.

## Assign before generating controls

After modality/target and MHC-resolution filtering, assign measured observations
to train/validation/test. Accept a hash-verified frozen assignment artifact, or
create and publish deterministic assignments with the requested seed/fractions.
Use stable observation identities and connected components, joining repeated
peptides, source contributors and known reviewed experimental origins; add
specimen/donor/study axes when requested by the declared independence policy.
Resolve aliases with the bundle's frozen registry. Unknown identity must remain
unknown; never infer specimen/donor from PMID, display label, cell line or genotype.
Do not split a component to hit a size target. Report actual component/partition
sizes and fail when the requested nonempty partitions cannot be formed.

Apply an external assignment to the selected subset, require every selected
observation to be assigned exactly once, and reject overlaps/policy violations.
Preserve the source assignment digest and record dropped membership after filters.
A generated split must not be advertised as independently held-out experiments
unless the actual biological audit can establish it. Default peptide-disjoint
qualification is narrower; unresolved biological axes stay explicitly inconclusive.

## Generated records

Carry source_partition as host metadata on measured records/samples. Generate
binding, elution and cascade controls separately inside each assigned partition.
Select all mismatch donors from that same partition and retain their observation
identity, not only the peptide string. Record immediate parents and root measured
parents for every generated row, plus generation kind, deterministic generated ID
and seed/draw identity. Do not copy a measured assay ID into a generated label.
Preserve distinct real/proxy/generated support accounting. The canonical frozen
path excludes unpinned auxiliary data and canary fabricated modalities.

Implement a small partition-aware augmentation wrapper shared by existing
augmentation calls; retain legacy non-frozen behavior explicitly. Cascades must
retain both parent generated IDs and root observations. Empty partitions do not
trigger cross-partition fallback. Existing control-policy choices remain visible.

## Actual dataset and artifacts

Construct the final dataset with retained partition and parent metadata. Split
indices follow the frozen assignment, not a second random split. Write the actual
membership after all dataset-level filtering and generated transformations, with
stable sample/generated IDs, bundle/observation/parent links, label kind and scope.
Never count alternate mappings or generated controls as new measured observations.

Audit both actual model-sample peptide overlap and measured/root-parent lineage.
Use Hitlist audit_splits with the bundle registry for observation/source/publication,
experimental origin, specimen and donor axes. Report pMHC ambiguity separately.
Reject cross-partition parent edges and any overlap required by the policy; an
unresolved required biological axis fails qualification rather than passing.
Write hashes, policy, exclusions, counts, coverage, overlap examples and a readable
README alongside reusable assignments in datacache. Run reports/predictions refer
to those exact shared artifacts. Prediction exports retain parent/partition fields.

## Verification

Use real-schema tiny frozen bundles and controlled generators to cover duplicate
mappings, cross-PMID reused origins, contributor reuse, pooled/unknown specimens,
external assignment mismatches, too few components, all generated parents, nested
cascade lineage, partition-local donor selection, final dataset drops and actual
prediction exports. Ensure IDs never enter model features. Compare legacy behavior
where its contract is unchanged. Version, format/lint, full CI/docs/package checks,
merge and verified GitHub release are mandatory. Close #71 only when this full
contract is demonstrated; real-corpus qualification remains Hitlist #680/#678.
