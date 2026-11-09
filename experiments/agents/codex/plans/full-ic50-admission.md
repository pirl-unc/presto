# Full eligible purified IC50 admission for matched controls

Prospective specification, 2026-10-21 UTC; Codex / GPT-6. The assay-only control
must not wait for successful auxiliary pretraining. This stage prepares real
source data and paired exact-only/exact-plus-censored training manifests. It
does not train a model or qualify affinity, annotation, APC, MS or TCR predictions.

## Frozen sources and supported population

Use the pinned Hitlist v4 binding parquet from the existing source inventory:
`/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/2026-10-16T07-20-43Z/binding.parquet`,
SHA256 `2b3f6ea2dd647e7fc54d214eef420feaebc4e81fab97ffc0565a2de1f8695ba0`.
Use the existing human IMGT records of the pinned mhcseqs catalog SHA256
`ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1`.
The catalog producer version is unknown; record runtime parser versions separately.
Freeze historical membership SHA256
`8959e60c0d9b8049aec25095b9c354d456b48f97ba8fb31d7b786e2861113e99`
and the committed 32-observation memorization JSONL SHA256
`145957e201c596ba82db4c900eb3715c34c81605ffadf887d0811f6f1d282ad5` before execution.

Admit all eligible direct IEDB/CEDAR IC50 measurements with an original assay IRI,
an explicitly purified-MHC assay method and explicit human HLA restriction.
Other endpoints, cellular assays and other molecular species remain separate
cohorts rather than being relabelled. Preserve each source's raw fields.
Require canonical unmodified peptide sequence, explicit `has_ptm=False`, no
reported modifications, no non-peptide-ligand flag, and lengths supported by
the model: 1–512 for class I, 9–512 for class II. Class I uses the entire peptide;
class II needs a complete nine-residue register. Report all excluded lengths.

Accept finite positive values in the model's declared concentration units
(`pM`, `nM`, `uM`, `mM`, `M`) without clipping. Preserve the original value/unit.
Explicit `=`, `<`, `<=`, `>`, `>=` are supported; absent/empty inequality follows
the declared source-export exact convention and retains its original missingness.
Require the raw numeric text to agree with the numeric value. An optional leading
inequality in the text must agree with the explicit source qualifier; no suffix,
range or approximate-value interpretation is invented. No biological label is
derived from the 500 nM evaluation threshold.

MHC references require source IDs, human IMGT taxonomy, functional/non-null/
non-questionable/non-pseudogene flags, clear groove status/flags, canonical
60–130-residue genuine domains and source-protein membership. Preserve all
identical catalog aliases for a two-field name. If that name has multiple
different admissible groove assignments, defer resolution rather than choose
the last row. Class I requires both domains of one heavy chain. Class II requires
an explicitly named alpha and beta chain; no DRA or other partner is imputed.
Verify each used assignment against current strict groove extraction. Report
nonclassical genes separately; a reference annotation alone does not qualify
peptide presentation, although an admitted assay is still a measured endpoint.

## Original-measurement conservation and partitions

Reuse canonical `prepare_observations`. Original lineage is the actual assay IRI
under one URI namespace, independent of export source/row ID. Exact copies get
one vote and retain their aliases. Conflicting exports of one original outcome
are deferred together. Distinct original assays remain separate observations,
including disagreeing same-input measurements; no averaging or consistency
filter from the 32-row capacity diagnostic is carried over.

Before deduplication, connect eligible observations sharing an exact peptide or
original assay IRI. Hash sorted component peptide strings, without using target
values. Preserve historical train/validation/test assignments for every peptide
appearing in the frozen membership. Keep all 32 memorization peptides in train.
If a connected component has incompatible historical assignments, defer the
whole component; never move an old held-out peptide into training. Assign entirely
new components using SHA256(`presto-direct-ic50-v1:<component>`), modulo 100,
boundaries 80/10/10. Freeze the salt, policy and old memberships before any fit.

This is a peptide/measurement-disjoint diagnostic partition, not a study/donor/
specimen independence certificate. Old held-out data have already been inspected;
record their reused membership and distinguish new rows. Shared studies and
MHC sequences, unknown donor/specimen/acquisition identity, and known prior
exposure must be reported. A new split does not certify unseen-allele performance.

## Outputs and readiness

Retain one final disposition for every raw binding row, stage-specific reasons,
all catalog aliases used, lineage conflicts, independent-input disagreement
reports, canonical source files and exact SHA256s. Independently replay each
admitted raw value, unit, qualifier, peptide, restriction and resolved groove,
plus source/measurement lineage and frozen grouping. Verify both source bytes
and source membership, not only the normalized JSONL.

Write common train-exact, train-censored, validation and test files by source.
The two manifests differ only in inclusion of train-censored files; their entire
validation/test observations are identical, with both exact and censored rows.
Allocate direct IC50, class-I/II branches, supplied human molecular species/class,
no annotation tasks, APC/MS/TCR modules, synthetic examples or variance reward.
Do not use the memorized checkpoint. The later fit starts fresh and fits scaling
only on its own training rows. Data-only preflight uses the canonical defaults
but is not authorization to launch an unregistered fitting recipe.

Report counts by split, class, qualifier, source, allele and study; target ranges,
500 nM support including ambiguous bounds, component-size concentration, source
duplicates/conflicts and historical assignment/exposure. Mechanical readiness
requires at least 50 exact measurements and 20 distinct peptide groups per class
in every split, at least 10 confidently binding and 10 confidently nonbinding
exact measurements per class/held-out split, and at least one multi-register
class-II training peptide. Failure preserves all evidence and prevents treating
the cohort as ready; do not adjust the salt, rows or thresholds to pass.

Register the data-only family before execution, freeze code/environment/invocation,
and close its README and canonical log regardless of readiness. Local CPU, one
Torch/BLAS thread, no GPU/Modal. No predictive metrics or dumps apply because no
model is evaluated; both held-out partitions are retained for a separately
registered matched baseline/training comparison. The full architecture and
biological qualification goal remains active.
