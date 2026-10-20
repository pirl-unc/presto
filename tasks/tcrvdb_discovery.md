# TCRvdb source discovery — 2026-10-20

The user's source is PMCID PMC12132471 / PMID 40463065, Messemaker et al.,
DOI 10.1101/2025.04.28.651095. This is independent functional re-testing,
not filtering VDJdb by confidence score.

Checked the official schumacherlab/TCRvdb repository, its main tree and release
references; schumacherlab/schumacherlab_publications_code notebooks; the
repseq/2026-vdjdb-db-validation tree and matchmakers README; Biohub/DecoderTCR
benchmark documentation; gloriagrama/data-noise-mitigation-methods-for-bap;
and danielgchen/reproducibility_tarpon. GitHub code searches included TCRvdb,
CSV/TSV names, the 2025 filename and the exact 2026 filename. Repository search
for TCRvdb returned the author repository. Search absence is not proof that
no copy exists anywhere.

Findings:

- [Author repository](https://github.com/schumacherlab/TCRvdb) main tree contains
  README.md only. It lists 2026-10-01_TCRvdb_ttd_gil_ylq_nlv_glc.csv and the
  older 01_05_2025_TCRvdb.csv. Current citation is
  https://doi.org/10.1016/j.immuni.2026.09.007. Access is requested at
  https://tcrvdb.nkiai.nl/; redistribution requires explicit permission.
- [Validation analysis](https://github.com/repseq/2026-vdjdb-db-validation/tree/master/matchmakers)
  points back to author access and explicitly does not provide the source CSV.
- [DecoderTCR](https://github.com/Biohub/DecoderTCR/blob/main/results/README.md)
  publishes benchmark results/figures and references TCRvdb, not a complete
  author dataset in its inspected tree.
- [Derived subset repository](https://github.com/gloriagrama/data-noise-mitigation-methods-for-bap)
  contains a selected 15-record non-validating table and a VDJdb join. These
  are not a complete, independently verified TCRvdb release. No data rows were
  copied into Presto, and no permissions were inferred from its presence.
- The large structural derivatives linked from downstream repositories are
  generated models, not substitutes for the author's full functional/QC table.

Decision: no complete provenance-verified public download found. Do not claim
that confidence-filtered VDJdb or these selected derivatives satisfy the user's
request. Build the independent wrapper and normalized contract, and inspect an
authorized author file before implementing source-column conversion or training.
User confirmed they do not currently have a local copy. No author request was
sent on the user's behalf.
