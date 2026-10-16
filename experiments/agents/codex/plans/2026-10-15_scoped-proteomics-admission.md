# Bounded published-scope admission audit

Question: does the actual published PXD013455 archive pass the new adapter,
retaining scientific provenance and honest task masks after global occurrence
collapse? This is a data-admission audit, not a training comparison or evaluation.

Input: the Hitlist-pinned pxd013455-hela-tryp46-v1 public archive; manifest SHA-256
441f3c66c6e6bc6a90c7cb8c54d0131676c3cb4e7a04ddbd9346061ad8f9d84b.
Reanalysis scope Hela_1_Bekker-Jensen_Tryp_46Fractions. Both source publications,
search FASTA/settings, cell/assay conditions and unresolved identities remain
in the immutable bundle. Datacache owns all reusable inputs and audit artifacts.

Admit a deterministic reservoir of 10,000 peptides (seed 42) after full occurrence
collapse. Compare full and admitted counts for MS/MS, matching and scoped absence.
Use real PrestoDataset, PrestoCollator, audit_split_support and output census;
assert 10,000 valid scoped labels, complete bundle/search/source linkage, unique
sample IDs, missing biological replicate/depth/excision labels, and masked
ambiguous contexts. Explicitly distinguish source-observation traceability
(positives only) from candidate-sidecar linkage (all records).

CPU only, one Torch/OpenMP thread, 15-minute wall timeout. Record OS/processor,
peak RSS, runtime, import paths/versions, git commit/dirty state and environment.
No GPU, model fitting, synthetic generation, held-out split or predictions.
Validation/test metrics are not applicable because this checks admission; the
next canonical trainer experiment must publish actual split and held-out evidence.

Publish admitted.tsv/parquet, coverage/policy/source receipt, complete canonical
support reports, hashes and readable README under shared datacache. Commit only
small summary/receipt JSON and the reproducibility launcher to this experiment.
Register before launch; capture the exact code commit. Close with counts, any
failure and corrective evidence, canonical experiment_log entry, and clear next
steps. Do not promote a model baseline or claim training readiness.
