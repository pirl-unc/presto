# Library namespace amendment after the ID hypothesis failed

The first lineage audit is retained unchanged. Its scored/aligned feature-ID
join yields 49,646 identity-consistent rows with exactly equal decimal intensity;
10,939 aligned rows have no feature ID and 3,485 IDs are absent from the deposited
score exports. No biological training followed these results.

The attempted equality of library TransitionGroupId and scored/aligned precursor
numbers fails on all but 20 aligned slots and 20 scored rows. These are distinct
identifier namespaces; a numeric coincidence is not provenance. Keep the failed
namespace test and every original exception. Do not replace IDs, silently call
them equivalent, or reuse this failed predicate in the shared compiler.

For library membership, use exact (ModifiedPeptideSequence, PrecursorCharge,
Decoy) equality with (FullPeptideName, Charge, decoy) for scored records and
(sequence, charge, '0') for aligned records. This is an exact chemical-form
membership assertion, not feature-ID lineage or a physical sample relation.
For aligned rows additionally crosscheck literal PeptideGroupLabel against
group_label. For scored rows crosscheck literal PeptideSequence against Sequence.
If the tuple has multiple library groups, preserve ambiguity; do not select one.
Preserve both original namespaces and count the mappings between them.

This amendment is prospective for the new membership stage, after the failed
namespace result was inspected. Do not modify the original lineage code or its
outputs. Independent verification must replay both the failed hypothesis and
the new membership mapping, including every exception and all original cells.

No score thresholds, input rows, numeric comparison tolerance, standard panel,
source selection or training decision change. Nominal-dose training is still
unadmitted; original native outer partitions remain closed.
