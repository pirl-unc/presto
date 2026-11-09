# PXD024809 scoring and query lineage before calibration training

Follow the source0813 audit and Hitlist#845. The preceding goal turn was progress:
all-pair native response failure was published and the new nominal-dose source
was independently audited. Do not expand the stopped native response model.

## Fixed question and source scope

Determine which aligned quantities correspond to original scored features, how
zeros/NA/missing scores arise, and whether all ten intended heavy standards were
in the master query library. These are observation-likelihood questions. No
calibration slope, biological predictor or model-selection score is fitted here.

Reuse0813's exact68-file inventory, full64,070-row aligned quantity table, project
protocol, original article and10-standard panel by hash. Register a new family
before reading additional source tables. Download all ten deposited per-run
`.mzML.tsv` scoring files and `Dilute_library_merged_decoy.tsv`, chosen by source
role and complete acquisition coverage, not response quality. No RAW/mzML search,
author contact, private access or new environment packages. Keep originals once
under a shared data source directory and freeze URLs, declared sizes and SHA256s.

Numerical stages retain20min/3GiB RSS/1GiB disk reserve+32MiB startup margin,
50ms process-tree monitoring and OSpeak. Stream public downloads in1MiB chunks;
check declared/exact received sizes and a512MiB total cap before starting. Keep
failed partial bytes/receipts and never silently retry or overwrite a completed
asset. The standard file retrieval socket timeout is60s with a20minute total
stage limit. No dataset or model threshold changes follow a retrieval failure.

## Admission and reconciliation

1. Inventory original schemas, dimensions, literal missingness, identifiers,
   sequence/modification/charge fields, decoy flags and source filenames before
   defining joins. Preserve exact headers and row positions. Unexpected columns,
   identity multiplicity or invalid values require a recorded interpretation,
   not a guessed alias or exclusion.
2. Link run names through the explicit0813name-normalization rule and record its
   filename-derived evidence. Keep physical specimen/aliquot identities unknown
   unless supplied. Ten technical acquisitions are not ten biological samples.
3. Join aligned rows to scored candidates using original feature IDs where the
   sources permit, independently crosschecking run/form/charge. Never use nearest
   intensity, RT or sequence similarity to manufacture a match. Report unmatched,
   ambiguous and disagreeing cases with source locators. Do not erase rows.
4. Compare reported scores, RT/integration boundaries and quantities only where
   their definitions match. Determine whether quantity sums/integration changes
   are source-explained; preserve raw/aligned quantities separately if they differ.
   Missing m_score is not automatically a failed identification or a censor bound.
5. Inventory full target/decoy library membership at precursor and molecular-form
   levels. Preserve transition multiplicity and target/decoy identity. Reconcile
   all ten intended standards, including the absent0813standard, to target library
   and each run's scored candidates. Absence from the library is not nondetection;
   a decoy-only match is not an intended target. Keep cross-dilution library
   construction and selection conditioning explicit.
6. Verify the declared isotope annotation and position against source group labels
   and UniMod695. Do not collapse labeled standards with endogenous peptides.
   Nominal solution concentration remains distinct from injection amount,
   certified molarity, native availability and MHC capture/recovery.
7. Independently replay every admitted join, source count and numerical comparison
   with a separate parser/implementation before shared ingestion or training.

## Shared implementation and next decision

Use current Hitlist main in an isolated feature branch. Reuse existing evidence
snapshot, streaming Parquet sink, catalog/store/query and calibration contracts.
Add the smallest source-specific adapter and registered tables needed to preserve
these original measurements and relationships; no parallel metadata system or
Presto-only training loader. Freeze the concrete schema after inspecting actual
source headers, before writing the compiler. Source preservation may ship with
explicit unresolved claims; training admission needs a separate recipe with
complete dispositions and a justified endpoint/likelihood.

Run meaningful altered-source/identity/missingness tests and complete required
Hitlist formatting/lint/fulltests. Independently reconcile the real source output,
publish a PR/version bump and finish merge/tested deployment before calling the
shared implementation complete. Never modify the shared Python installation.

If source relationships support a bounded nine-standard nominal-dose diagnostic,
register its model, controls, held-out dose/sequence grouping, zero/missingness
policy and identifiability limits before fitting. If they fail, publish the
specific missing evidence and choose the next observation anchor; do not fit a
surrogate label. Either outcome is short of the full precursor-relative APC
availability and matched differentiable/discrete objective.
