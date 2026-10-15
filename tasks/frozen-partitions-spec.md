# Frozen observation partitions and generated-parent lineage — Presto #71, part 2

## Integration order

This branch follows the frozen-admission PR #76 (merged as 2e6d923). Its clean-main
0.1.2 release is published and downloaded/hash-verified. The dependent branch is rebased onto that merge. Keep the user checkout
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

Use one host-metadata helper to identify generated records from their complete
generated payload, generation kind, seed, draw index and immediate parent IDs.
Retain the transitive measured roots separately. Preserve the chosen peptide donor
record instead of reducing donor pools to strings. Where an allele is sampled
from a deduplicated pool, explicitly select the first supporting record in stable
input order as its metadata donor; this preserves the existing random sequence
while making the derivation inspectable. Known-negative allele donors must come
from the matching peptide's negative evidence. The policy and parents are explicit,
not an assertion that mismatched alleles are measured negatives.

Frozen generation rejects missing root lineage, mixed bundle identities or mixed
partition membership. Legacy generation may remain unresolved, with that status
recorded explicitly. Never copy a parent's assay ID or measured value metadata
onto a generated target. Cascaded controls name the generated immediate parent
and its measured roots, and generated sample IDs remain stable through loaders.

Review reproduced a single-allele elution control that exactly duplicates a known
positive pair with the opposite label. Reject generated elution pairs overlapping
known positive candidate restrictions and count rejected attempts. For binding
controls retaining a real MHC, reject already measured peptide/allele pairs;
mutated/absent MHC controls remain a distinct input contract. Include hard-pair
and promiscuous-peptide cases. Requested generation attempts do not guarantee an
admitted row count. This source-validity correction accompanies parent tracing.

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
