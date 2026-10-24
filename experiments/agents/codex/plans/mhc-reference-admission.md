# Admit actual single-groove MHC species annotations

Prospective data-only preparation, 2026-10-21; Codex / GPT-6. This supplies a
real source for the now-supported single-chain auxiliary path. It performs no
optimizer updates, predictions, model selection or biological qualification.
Any later pretraining recipe/baselines/transfer screen require registration.

## Pinned source and deterministic curation

Use `/Users/iskander/.cache/mhcseqs/mhc-full-seqs.csv`, SHA256
`ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1`.
This is a pinned existing reference export, not a claim of current/latest producer
version. Preserve source/source_id, named allele and gene, raw taxonomy, class,
chain and quality metadata in row dispositions and annotation aliases.

Freeze these exact source-string mappings before preparation:
`Homo sapiens` -> `human`; `Mus musculus (Mouse)` -> `mouse`;
`Rattus norvegicus (Norway rat)` -> `rat`; `Macaca mulatta` -> `rhesus_macaque`.
Exclude other raw taxa; do not collapse unresolved `sp.` strings into a species.
No label-dependent sampling caps, class balancing or row resampling in this audit.

Require source/source_id, `groove_status=ok`, empty groove flags, functional=True,
null/questionable/pseudogene=False, and one of (I,alpha), (II,alpha), (II,beta).
Reject anomalous B2M/other chain assignments. Class I needs both actual domains
from that same reference chain; class II alpha/beta needs its actual single
first/second groove and an empty opposite slot. Each used groove must have
60–130 canonical residues, agree with its declared length and occur verbatim
within the source protein; class-I domains must be in source order without
overlap. These are mechanical/source-annotation checks, not evidence that every
reference protein is a peptide-presenting molecule. Nonclassical genes remain
explicit and are not automatically admissible to binding assays.

Explode each eligible reference chain into its actual domain annotations. For
species pretraining, retain one observation per unique (groove slot, sequence,
species), with all source aliases. This avoids repeatedly supervising a shared
domain once per allelic alias. If an identical groove sequence has conflicting
species among eligible rows, defer every source row containing it instead of
choosing a majority species. Keep every excluded/deferred/aliased row disposition.
Single-chain class metadata remains provenance; no paired class/pairing target
or missing partner is generated.

## Grouping and readiness

Connect annotations that share an exact groove sequence (including across slots)
or originate from the same reference protein/allele row. Hash sorted component
sequence keys; assign the whole component by SHA256 of
`presto-mhc-reference-v1:<component>` modulo 100, boundaries70/15/15. Preserve all
component members and parent links. The index must admit the resulting cohort
with no repeated identity chain across splits. This grouping prevents exact
sequence/original-molecule overlap; it is not a homology-cluster qualification.

Write a species-only modular manifest with the four frozen categories and
canonical observation lineage. No class/assay head, artificial sequence, negative
pair or assay label is allocated. Report per-species/class/slot/source/split
counts, aliases, conflicting sequences and connected-component size distribution.
Report a prospective readiness flag: every species in every split must have at
least20 domain annotations and at least3 connected components. Failure preserves
the cohort/report and prevents treating it as a ready four-species comparison;
do not tune the salt, thresholds or vocabulary to obtain a passing result.

Independently replay admitted observations against the pinned source rows:
sequence, source membership, slot, taxonomy mapping and label must all agree.
Verify every raw row has exactly one disposition, all aliases trace to eligible
nonconflicting rows, every admitted domain has an alias and each eligible source
row accounts for exactly its genuine domains. Run canonical preflight; no model
forward or test predictions. Preserve the full cohort as a hashed generated data
artifact, compact audit tables, actual source/code/env pins and invocation.
Validation/test partitions exist for future use, but predictive metrics/dumps
are inapplicable to this data-only preparation. Close the family and canonical
registry even if readiness fails; the full model goal remains open.
