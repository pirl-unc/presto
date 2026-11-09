# Exact lineage contract after structural inventory, before joining

Inventory found 76,561 scored rows, 891,222 library transitions and 64,070 aligned
slots. There are no duplicate scored (filename,id), library transition IDs or
aligned (run,precursor) keys. Scored filenames match their assets under the fixed
normalization. The library has 12,851 precursor/form memberships. Its nine
observed intended standards use the same UniMod:695 forms as the aligned/scored
tables. The tenth bare standard does not occur in the library inventory.

Join each aligned (run,feature_id) to (filename with terminal '.mzML' removed,id).
Keep IDs as text; missing literal tokens never join. Independently check exact
sequence/FullPeptideName, charge/Charge, precursor/transition_group_id and target
decoy==0. Output every aligned slot and its original source line, link status,
candidate count/locator and each crosscheck. Missing, absent, ambiguous or
disagreeing links remain explicit; no sequence/RT/intensity fallback.

Join both aligned precursors and every scored candidate to library
TransitionGroupId; independently verify full form, charge, target/decoy state,
and aligned group_label where available. Preserve every exception. These joins
do not assign physical specimen identity or identification acceptance.

Compare intensity/Intensity, RT, leftWidth, rightWidth, peak_group_rank and m_score
only for unique scored-ID matches, recording raw values, signed decimal delta
and relation: same_literal, decimal_equal, within_numeric_tolerance, different,
both_missing, aligned_missing or scored_missing. Tolerance is fixed here as
abs(delta) <= 1e-12 + 1e-12 * max(abs(a),abs(b)); the label means only numeric
closeness, not proof of common integration or rounding. No biological/FDR cutoff
or replacement value is introduced. Report all differences, including matched
rows with identity disagreement. IDs and identity crosschecks never use tolerance.

Derive expected heavy forms and group-label forms from the original ten literal
protocol standards by exact substitution of '[L(13C6; 15N)]'. Reconcile intended
position, full form and charge, separately counting target/decoy memberships,
scored candidates and aligned slots. No inferred rows for a missing standard.

Publish a deterministic compressed TSV containing all aligned comparisons and
small JSON summaries/standard rows/library and candidate exceptions. Independent
verification uses pandas literal parsing and rational rather than decimal
comparison, rebuilding all joins and all source inventories without importing
the production audit. No fits, held-out predictions or native outer access.

Pipeline evidence: pinned main.nf lines817-821 applies parameterized q-value
filters during PyProphet export. Therefore absence from a deposited scored TSV
cannot establish absence of all internal candidates. Actual study parameter
overrides are not yet established. Missing feature IDs/scores also do not identify
the alignment algorithm's reason for a zero or NA.
