# MHC-II MS/context source admission

Codex / GPT-6, 2026-10-26. Parent: `tasks/mhc2_improvement_spec.md`.
Register before acquiring/analysing the corrected human training export. This
family prepares actual model inputs; it does not claim predictive improvement.

## Input acquisition and limits

Use the canonical `human-cohort-corrected-v1` Hitlist release, first verified in
the1803release inventory. Fetch `training.parquet` (188,175,209bytes), `manifest.json`
and `SHA256SUMS` initially, plus the qualification report declared in launch.json
before acquisition. Verify every downloaded content file against published
checksums and retain original release metadata. New-download ceiling256MiB;
local one-thread analysis300s/3GiBRSS per invocation. Preserve the existing CPU
trainer/controllers, root dirt and shared environment. Raw bytes live under
`/private/tmp/presto-mhc2-20261026/ms-source/`; never replace the old v4 export.
No network literature expansion or public raw-MS download in this family.

## Admission questions and policy

- Inspect the actual schema and release qualification before claiming any
  molecular, material or context coverage. This Oct16 cohort is distinct from
  current Hitlist2.0.42 source; new curation is not silently broadcast onto it.
- Use source-class-II, human, unmodified canonical peptide observations under
  an explicit MS modality. Missingness/unobserved peptides are never measured
  negatives. Preserve source row/assay/study and observation-process identity.
- Resolve only evidence-supported candidate molecular pairs via the pinned human
  catalog. Preserve every candidate in a sample-level set; do not select a best
  predicted binder. Record partial-chain/candidate ambiguity as dispositions.
- Consume canonical source-context assertions at their actual scope. A uniquely
  matched arm can supply its reported intervention; a pooled/unmatched arm cannot.
  Use Hitlist's token semantics for concurrent conditions and preserve provenance.
  No added DM is not a component-loss measurement. Do not infer active processing
  pathways from class, cell type, protein origin, expression or intervention alone.
- Preserve paired source and candidate-parent evidence where available. Report
  missing contributor, parent, exact specimen or preparation evidence explicitly;
  do not manufacture specimen independence from run/sample labels.
- Determine which genuine source contrasts survive. A single-context study can
  support ligand ranking but cannot train an intervention effect. Separate the
  novel pathway/intervention inputs from cell identity and detector covariates.

## Next predictive family, not authorized by an unstated default

After admission, freeze a separate actual training/evaluation contract comparing
binding-only, sequence/MS and context-enabled models, with matched contrasts,
source-aware train/validation/test separation, actual per-example dumps and
independent metrics. If using generated candidate controls, name their sampling
distribution and limited contrast estimand; never call them observed negatives or
claim unconditional detection calibration. Retain class-I binding checks.
No parameters, losses, generated controls or test selection are set by this
admission plan. Do not begin fitting before that contract exists.

## Closure

Use `experiments/2026-10-26_1825_codex_mhc2-ms-admission/` for input/reproducibility
receipts, source/schema metadata, all dispositions and qualification results.
Independently verify source IDs, pairs, sample attribution and exported context
against original canonical rows before training admission. Preserve failed
attempts; update README and canonical experiment log when materially informative.
No predictive split/metric/dump applies until a predictive family is registered.
