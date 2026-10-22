# Endogenous availability sources: admission before biological fitting

October 22, 2026; Codex / GPT-6. Active availability goal; prospective audit,
not an admitted dataset or a claim of concentration identifiability. The user
asked which data will be used. Public paper/data-availability records have been
checked; downloadable source tables and exact pairing remain to be inspected.

## Intended sources and distinct roles

1. **Wolf-Levy et al. 2018**, proteasome-associated fragment measurements and
   parallel standard proteomics: https://pmc.ncbi.nlm.nih.gov/articles/PMC8897557/
   Start with the cell-line peptide table (Supplementary Data 2) and corresponding
   proteome/run mappings. Public PRIDE series PXD009092, PXD009360, PXD007263,
   PXD009181, PXD009438, PXD009112, PXD009508; map actual accessions/files to
   experiments instead of assuming all belong to the initial cohort. This MAPP
   assay captures proteasome-associated endogenous fragments; it is distinct
   from MHC-II-associated MAPPs. The paper reports 4,983 enriched peptides from
   1,004 proteins and parallel proteomics, but these counts are paper claims
   until source reconstruction. Initial HEK293 data do not validate professional
   APC generalization. Primary PBMC measurements are mixed-cell samples, not
   resolved APC measurements. Defer patient data when a cell-line cohort suffices.
2. **PXD057588**, quantitative cell-type-specific macrophage/tumor
   immunopeptidomics with global proteomics:
   https://pmc.ncbi.nlm.nih.gov/articles/PMC12616400/
   Closest candidate for APC-specific downstream validation. Audit Supplementary
   Tables 7–12, run/channel metadata, species, cell type, co-culture attribution,
   source proteins and matching global proteome. The published processing discards
   missing-channel peptides and uses a reported q-value threshold of .05; do not
   silently call this high-confidence complete coverage. Inspect pre-filtered
   identifications, confidence and missingness before deciding admission/curation.
   Quantifies bound ligands, not the complete intracellular product pool.
3. **Stopfer et al. 2020 / PXD017407**:
   https://pmc.ncbi.nlm.nih.gov/articles/PMC7265461/
   Relative pMHC data and calibrated copies-per-cell for selected targets;
   Supplementary Data 1–4, file maps 1–3/5, mass list 6 and source data. Reserve
   calibrated targets for a downstream quantity check when nonoverlapping units
   permit it. Melanoma cells are not professional APCs. RNA-seq (GSE144373) is
   not measured precursor protein abundance; preserve that distinction.
4. **Existing PXD013455 HeLa Trypsin/P Hitlist scope**, documented in
   `docs/scoped_proteomics.md`: already available software path for binary
   within-search non-MHC MS observation. Use only for the observation component;
   no native processing/absolute concentration labels. Its 46 fractions are not
   biological replicates and its negatives are scoped unobserved candidates.

Keep existing curated affinity/stability/elution labels in their declared heads.
Do not merge intensities, counts, copies/cell and affinity into one response.
The four source roles above are a staged stack, not one already-joined cohort.

## Homogeneous metadata prerequisite

The user's October 22 clarification requires source/sample distinctions in one
shared curation system. Follow `docs/study_curation_contract.md` and the existing
Hitlist 2.0 issues733/736/737. Do not create another Presto-specific source registry
or leave critical facts only in study notes. Freeze normalized source metadata,
assertion evidence/scope, actual material links and task-eligibility decisions
before fitting. Ingestion may preserve unresolved records; training must not
silently convert unknowns to defaults. The complete v2 path remains pending.

## Audit deliverable and gates

Create a dated experiment/audit directory with README, reproduction commands,
download inventory/URLs/hashes, actual file schemas and machine-readable coverage.
Keep raw reusable source assets under `data/`, with immutable paths/hash receipts.
Read only public data; no private access request or author contact is needed.

- Establish licensing, source version, study/run/sample/channel identities,
  measured pool, species/cell type, perturbation and control provenance.
- Inspect actual peptide identities, modifications, parent mapping and search
  reference; resolve coordinates without choosing arbitrary ambiguous parents.
- Preserve detection evidence, confidence thresholds, blanks, contaminant and
  decoy status, matching-between-runs status, complete-case filtering and missing
  values. Do not relabel missing/filtered values as measured zeros.
- Determine what is quantified: raw/normalized intensity, ratio, amount or
  copies/cell; preserve source transformations and calibration support.
- Verify real sample pairing with precursor protein measurements. Transcript
  abundance and unpaired population protein estimates are separately tagged.
- Determine whether same-peptide condition comparisons can reduce fixed response
  confounding, and whether instrument/capture changes invalidate that assumption.
- Define source-protein and specimen/study holdouts; report overlapping peptide
  sequences and source ambiguity. Do not infer unseen-study generalization from
  random peptides within a single study.
- Publish accepted/rejected rows and reasons, admitted endpoints and unresolved
  calibration. No biological fit before this contract is frozen.

## Scientific constraints on the next fit

Proteasome-associated intensity measures a captured pool, not directly production
flux or free intracellular concentration. MHC-bound quantities additionally depend
on binding/retention and capture. A matched precursor measurement helps but does
not eliminate peptide-specific response or pool/residence-time effects.

Start with the endpoint the tables support (likely relative observed fragment
signal or within-peptide changes), and compare a precursor/context baseline with
sequence-dependent predictions under identical splits. Do not normalize arbitrary
MS intensities and then call them calibrated molecule probabilities. If availability
and observation factors are not independently anchored, fit their identifiable
combined response and report that scope rather than interpreting separate factors.

Specify the next exact-vs-sampled objective only after data admission; preserve
the same observable target in both estimators. Add machinery-loss comparisons
only from verified matched loss/control cohorts. None of these initial sources
alone validates class-II lysosomal processing or all APC knockout effects.
The full concentration goal remains open until independent quantity evidence
supports its stated precursor-normalized interpretation.
