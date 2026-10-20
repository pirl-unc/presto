Presto is adding a separate paired full-chain TCR–pMHC functional-specificity wrapper. Its existing native Hitlist route returns no receptor-specific evidence; `generate_training_table(include_evidence=...)` currently supports ms, binding and both. VDJdb confidence-score filtering is not the requested ground truth.

The intended source is Messemaker et al., “A functionally validated TCR-pMHC database for TCR specificity model development” (PMCID PMC12132471; preprint DOI 10.1101/2025.04.28.651095). The author repository https://github.com/schumacherlab/TCRvdb now lists 2026-10-01_TCRvdb_ttd_gil_ylq_nlv_glc.csv and publication DOI 10.1016/j.immuni.2026.09.007. Access is by request; redistribution requires explicit permission. No source CSV is attached to this issue and a public VDJdb score is not a substitute.

Request a separate receptor/functional evidence schema and local-file adapter after authorized access, without placing restricted data in public package/release/CI artifacts.

Acceptance:
- Inspect the actual authorized source schema; pin source version/hash and source-row/receptor/pair/assay identities. Keep paired alpha/beta evidence from the same receptor; never pair unrelated chains sharing an epitope.
- Preserve supplied full-chain sequences versus CDR3/V/J-only annotation and explicit reconstruction provenance. Missing/full-length-unverified sequences remain visible; do not label reconstruction as experimentally observed full length.
- Preserve pMHC identity, tested assay context, raw functional readouts/statistics, QC/expression status, replicate aggregation and author validation policy. Keep validated, tested-not-validating, indeterminate, failed-QC and unassayed separate. Non-validation is assay-scoped, not universal molecular non-binding. No shuffled pair or unobserved database entry is a measured negative.
- Retain original VDJdb/source-study links for leakage audits; receptor/pMHC/study overlap and near-duplicate receptor grouping must be inspectable. Report epitope/MHC coverage, without claiming arbitrary-epitope generalization.
- Export a reproducible local bundle/coverage receipt respecting the source's terms; use artificial fixtures in public tests. Keep this evidence separate from MHC affinity and positive-only MS elution.

Related #678/#680. This requests a data adapter/contract, not a dataset mirror or a trained model. The complete author data are not currently available locally; GitHub search located analysis code and a small derived subset, not a verified full release.
