# Quantify native MS observations with source-qualified condition contrasts

Date: 2026-11-10. Codex / GPT-6. Source code base: `038ffbcc` (PR156).
Previous goal turn was progress: the shared adapter fix, metadata audit, PR156
and scientific follow-up #155 were published. This family advances #155 toward
actual data selection rather than treating software tests as scientific evidence.

## Frozen inputs

Use the already admitted native Hitlist 2.1.3 export on `presto-checkpoints`:
`2026-11-07_1645_codex_unified-data-audit/admission/canonical-ms-export.parquet`,
112,357,418 bytes, SHA256
`4a739794ab7a21ceb633df6e4cde4c01054f3e53213a5a8dcdc4d78604dde58f`,
5,440,019 rows. Download to a fresh local artifact directory and verify before
reading. Resolve MHC with the original admission's cached catalog, SHA256
`ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1`.
Do not overlay the newer Hitlist YAML onto older observations.

## Questions and procedure

1. Inspect schema and project only fields required for literal source condition
   qualification and the existing positive-MS/candidate-MHC adapter. Preserve
   canonical source row ordinal and source hash. Do not read binding tables,
   prediction dumps, any designated TEST or confirmation file, or the 18-GB
   admission database. This is an unsplit positive-source inventory, not a
   TRAIN-only cohort or evaluation; it accesses native positive-MS records.
2. First pass: collect source study factor vocabularies with the shared adapter,
   only from qualified, unique-arm annotations. Count attribution/status/evidence
   and condition availability by class. Missing annotations remain missing.
3. Second pass: apply the shared MS adapter to records with at least one admitted
   factor, including the existing DM/DO/cytokine families and the four newly
   preserved fields. Keep original chemistry, restriction and complete candidate
   set gates; use the existing declared human DRA reference assumption. Count
   admission exclusions and distinct measured row IDs/peptides separately.
4. Summarize admitted positive observations by study, condition, class and factor
   state. Find studies with explicit present/absent contrasts. Label these only
   candidate contrasts until molecule, cell/material, capture and independent
   sample mapping are reviewed. Counts of identifications are not quantities or
   treatment effects, and observed-only data do not provide measured negatives.
5. Freeze the next cohort/evaluation specification from these source facts,
   preserving all existing study/sequence/initializer embargoes. Do not start a
   biological fit or score an existing development/TEST cohort in this audit.

## Artifacts and verification

Family: `experiments/2026-11-10_1431_codex_ms-condition-contrasts/`. Preserve
launch/source pins, source-schema inventory, grouped counts, exclusions, source
identity references, and runtime. Use local CPU with bounded Arrow batches,
one process and no model/GPU. Raw export stays outside Git; compact summaries
and reproducibility go in the family and canonical log. No losses, synthetic
labels, pretraining, predictive validation/test metrics or prediction dumps
apply. Test semantic grouping/unknown handling on artificial records and replay
key counts independently from original projected columns before closing.

If the audit shows only unpaired/confounded arms or missing annotations, retain
that finding, attach concrete evidence to the relevant existing upstream issue,
and pursue a source that supplies the required contrast. Do not invent controls
or stop the broader binding/MS/processing objective at the audit.

## Source-review amendment after the literal audit

The audit admits 36,495 class-II condition-bearing rows. Its only class-II
present/absent candidate is CIITA (PMID33592498), whose source notes qualify
parental detections as probable background. Do not fit a positive-ligand contrast
from this count alone. 305,607 otherwise condition-qualified class-II records
are deferred for unspecified peptidoform status; identify their source distribution
and review the current upstream preservation fix before filing a new curation gap.

Review original metadata for PXD017154 / PMID33025005, found by source discovery
and absent from the current Hitlist literal-accession search. Allow up to twelve
public metadata/article/file-list requests, each at most 10 MiB, to establish
processed-file availability and exact study/condition mapping. Do not download
raw spectra or fit from this new source during this admission audit. Preserve
retrieval receipts and link a new issue only if existing ownership is absent.
