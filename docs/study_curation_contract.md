# Shared study and sample curation contract

This is a consumer requirement for the [Hitlist 2.0 catalog](https://github.com/pirl-unc/hitlist/issues/733),
not a second Presto source registry. It extends the
[material/protocol work](https://github.com/pirl-unc/hitlist/issues/736) and
[cross-modality work](https://github.com/pirl-unc/hitlist/issues/737).
New biological availability datasets must be curated through that homogeneous
system before training. Experiment READMEs summarize the resulting frozen
metadata; they must not be the only place where a distinguishing fact survives.

Status: prospective acceptance contract. Existing Hitlist condition, species and
lineage registries supply parts of it; the complete normalized v2 implementation
and the new source-specific sample/run curations are not yet qualified.

## Identity and scope

Retain separate, joinable identities for:

- study/publication and deposited dataset/version;
- original source file/table/sheet/row and its contributor;
- donor/animal, specimen, cell preparation, aliquot and experimental arm;
- measured cell population and any coculture/feeding/host partner;
- assay/capture, multiplex channel, fraction and acquisition run;
- analysis/search/quantification scope and its input files;
- measured peptide/peptidoform/protein result, mapping alternatives and quantity.

One specimen can contribute several modalities; one acquisition can contain
several multiplexed samples. Neither a publication nor a cell-line name establishes
a shared specimen. Fractions, channels, replicate injections and independent
biological replicates remain distinct. Reanalysis and reuse keep their connection
to the original measurements rather than creating new independent evidence.

Use explicit supported joins to Hitlist's existing material, condition, species,
lineage and observation identities. Preserve unresolved/pooled/ambiguous identities
and alternative mappings. Never broadcast a study-wide fact to every sample unless
the source explicitly supports that scope.

## Metadata shared across all studies

| Category | Required curation coverage |
|---|---|
| Source and evidence | Citation/accession/version, access/license, original files and hashes, source locator, curator/recipe version and date, raw source assertion alongside normalized value |
| Biological material | Species by role, strain, cell type/ontology, cell line/clone, tissue/compartment, primary versus transformed material, disease/state, relevant demographic or animal metadata when reported, isolation/selection history |
| Experimental system | In vivo/ex vivo/culture, monoculture/coculture, participating populations and roles, measured/sorted population, direct contact versus separated culture when reported, purity/contamination evidence, antigen source and delivery |
| Conditions and controls | Concurrent genetic and environmental factors, target gene/component, KO/KD/overexpression/variant and validation, cytokine/drug/stimulus identity, dose/unit/duration/order, explicit control type and matched control relation |
| MHC context | Molecular species, class, independently typed genotype, presenting candidate set, introduced MHC, capture specificity and peptide-level restriction with its evidence; these are separate assertions |
| Measured pool | MHC-bound ligands, proteasome-associated fragments, native intracellular/extracellular peptides, digest peptides, bulk proteins or purified molecular assay; compartment, enrichment/capture and intended observable |
| Assay and acquisition | IP target/antibody, digestion versus native peptide recovery, enzyme/specificity, chemical labeling, sample input, enrichment/fractionation, instrument/mode/fragmentation, channel/fraction/run, biological/technical replicate and batch |
| Search and identification | Search software/version, reference/proteome hash and species scope, modifications, peptide-length/search constraints, confidence/FDR threshold and level, decoy/contaminant treatment, direct identification versus matching/imputation, peptide-to-parent ambiguity |
| Quantification | What is measured, raw text/value, parsed value, unit, linear/log encoding, relative versus absolute scale, normalization/reference/batch, calibration standards and scope, recovery correction, cell count or precursor denominator and its provenance |
| Missingness and selection | Not searched, searched/unobserved, below a stated detection/quantification bound, failed QC, missing channel, filtered row, imputed value and measured zero; complete-case and other selection rules, original values and reasons |
| Relationships | Exact shared material/aliquot or paired specimen versus condition-level, cell-line, parent-line or tissue surrogate; relationship evidence, direction, counterpart IDs, applicable comparisons and unresolved precision |

Controlled terms/ontology IDs should be retained where established, together with
the original label. Missing ontology mapping does not justify dropping a source
fact or assigning a guessed term. Existing Hitlist condition vocabulary remains
the source of truth for interventions, not a new Presto synonym list.

Each assertion must distinguish a reported fact, explicit curated normalization,
inference/default, unknown/unreported, explicit absence, not applicable,
unresolved/ambiguous and conflicting evidence. An empty intervention field cannot
mean wild type. Inference is never silently upgraded to a measured annotation.
Evidence includes the relevant section/table/row and the scope of the claim;
confidence in an annotation is distinct from confidence in sample attribution.

Use [SDRF-Proteomics](https://github.com/bigbio/proteomics-sample-metadata/blob/master/sdrf-proteomics/README.adoc)
for interoperable sample-to-data-file information where available. Keep a pinned
version and preserve its source names, factors, channels, fractions and run links.
SDRF does not replace the measured-pool, calibration, uncertainty and task-eligibility
relations above. Do not claim SDRF validation without running its validator.

Scope multiplex channel identities to an assay/plex: a channel label can refer
to different samples in different measurements. Preserve source-table blocks and
cell coordinates; independently sorted panels on the same spreadsheet row are
not joined observations. Repeated SDRF column names must remain distinct ordered
fields rather than being overwritten in a dictionary.

## PXD057588 acceptance case

The [primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12616400/) is a source
to curate, not an already admitted training cohort. Source-reviewed records must
support all of the following distinctions without parsing free-text study notes:

1. Macrophage versus tumor measured population, the partner population, and naive/
   monoculture versus coculture state. Mixed or unresolved attribution remains
   explicit; a coculture label cannot identify which cells supplied a result.
2. Biological species and source reference for each population and peptide-mapping
   candidate. A source-compatibility filter is not proof of cell-of-origin.
3. MHC-I immunopeptidomics versus global proteomics, with their distinct material,
   capture/digest and observation scopes. MHC-bound intensities do not label native
   intracellular peptide concentration or isolated processing/MS components.
4. Original quantitative channels, baseline/control grouping and actual pairing to
   global proteomics, with unknown specimen relationships preserved. RNA expression
   would be a different modality and cannot substitute for protein abundance.
5. Reported identification thresholds, removal of missing-channel peptides,
   normalization, and filters intended to reduce contamination/misattribution.
   The paper reports q≤0.05 and complete-case filtering: both must remain visible
   in eligibility decisions; do not call all rows high-confidence or fill absent
   observations with zero.
6. Selected targeted/calibrated measurements versus broad relative measurements,
   including which targets and samples their calibration actually covers.

Final sample/run/channel IDs, numbers of independent specimens, pairings and
admitted quantities require inspection of deposited tables/file maps. The paper
alone does not establish them. Ingest candidate source records with an explicit
pending status; they cannot become training examples through membership alone.

The [initial deposited-source audit](https://github.com/pirl-unc/presto/blob/codex/modular-presto-model-20261020/experiments/2026-10-22_1410_codex_availability-source-audit/README.md)
found these concrete acceptance cases: the community SDRF covers only six of fifty
raw files; eight of nine channel-to-sample mappings differ between Tables 9 and 11;
Table 10 describes a tumor-partner MHC-I loss, not loss in the measured macrophage;
and Table 11 contains digest-peptide rows despite its protein-expression title.
Source conflicts (Table 8's M2 description and Table 10's replicate caption),
unknown specimen links and assay-specific normalization must survive ingestion.
These are source-audit findings, not a qualified normalized dataset.

The [quantity reconciliation](https://github.com/pirl-unc/presto/blob/codex/modular-presto-model-20261020/experiments/2026-10-22_1601_codex_pxd057588-quantity-reconciliation/README.md)
reproduces all 200,724 primary log2 fold changes from the original intensity cells.
These derivations must be represented as shared quantitative evidence, with a
versioned recipe, input measurement IDs/cell locators, output quantity, denominator
membership, operation order, normalization universe and source-method citation:

- Tables 7–10 and 12: log2 of treated intensity divided by the arithmetic mean
  of the same peptide's named control intensities. Table 12 has different control
  groups for the two tumor lines.
- Table 11: first divide each channel by its total over all retained digest-peptide
  rows, then compute the same control-mean ratio. The common mean-total multiplier
  cancels. Recomputing totals after an arbitrary training subset changes the target
  and is not the published normalization.
- Keep original intensity, derived log2FC and calibrated copies/cell as different
  quantity types. A recipe operating on an unscaled table cannot silently inherit
  the global-proteome normalization. All 200,724 values disagree under the tested
  opposite policy.
- Preserve the 32 contrast definitions and exact channel maps in the frozen
  [contract](https://github.com/pirl-unc/presto/blob/codex/modular-presto-model-20261020/experiments/2026-10-22_1601_codex_pxd057588-quantity-reconciliation/reproduce/contract.json).
  The [cell comparison dump](https://github.com/pirl-unc/presto/blob/codex/modular-presto-model-20261020/experiments/2026-10-22_1601_codex_pxd057588-quantity-reconciliation/results/cell_comparisons.csv.gz)
  supplies acceptance evidence; it is not a separate production curation format.

Raw parent labels also need mapping provenance: the 663 shared labels between
Tables 9 and 11 are neither verified protein-reference identities nor specimen
pairs. Only 110 labels have multiple observed ligands and multiple digest peptides
(258 ligand rows). This support count is a cohort diagnostic, not a rule that
missing products have zero abundance. A normalized export must retain enough
evidence to recompute the diagnostics and distinguish a source fact from a
training recipe's eligibility decision.

## Cross-study acceptance cases

- **Wolf-Levy MAPP:** proteasome-associated capture and parallel global proteomics;
  no confusion with MHC-II-associated MAPPs, free peptide amount or production flux.
  Keep cell-line observations separate from primary mixed-cell observations.
  The [processed-source audit](https://github.com/pirl-unc/presto/blob/codex/modular-presto-model-20261020/experiments/2026-10-22_1630_codex_mapp-source-admission/README.md)
  supplies 18 run mappings, 10,673 peptide records and 160,095 quantity comparisons.
  Preserve `MAPPbeadFilt` quantities separately from source evidence intensities:
  9,276 reported zeros have positive underlying signal. Retain direct versus
  transferred identifications, peptide/PSM/protein confidence levels and the
  disagreement between project and paper peptide-FDR claims. Quarantine the
  36 source protein-group records without IDs; never overwrite a shared empty key.
  Fragment-derived protein-group LFQ is not independent precursor abundance.
  Source structural support for 2,272 UT/cytokine peptide comparisons is not
  endpoint admission; all remain deferred pending shared context and source review.
- **PXD017407:** relative elution quantities versus selected calibrated pMHC
  copies/cell. Retain calibration applicability; melanoma is not a professional
  APC annotation and transcript abundance is not measured precursor protein.
- **PXD013455:** bulk Trypsin/P search scope, direct MS/MS versus matching,
  46 technical fractions rather than biological replicates, unknown donor/specimen
  identities, and binary within-search observability rather than native supply.
- **PXD056426/HAP1:** component-loss/control arms and quantitative elution/proteomics
  with real clone/replicate/sample links; preserve the requirements already filed
  in [Hitlist #726](https://github.com/pirl-unc/hitlist/issues/726).

## Enforced admission and downstream use

Separate source ingestion from task eligibility. Ingestion preserves incomplete
evidence. A training recipe explicitly names its needed measured pool, endpoint,
quantity scale, context precision, confidence, matching/missingness policy and
join/split requirements. It emits per-row admitted/deferred/excluded decisions
and a coverage report naming unavailable facts. Unknown metadata must not pass
as an assumed control or exact match.

Freeze registry generation, input hashes, field-role definitions, normalization
and conversion policies, source joins, exclusions and split lineage in every
training bundle. Biological features, measurement-process covariates, targets,
grouping/eligibility metadata and provenance have separate declared roles. Retaining
an important fact does not automatically make it a predictive feature; study ID,
source row, batch outcome summaries and calibrant target identity must not leak
the label into a biological representation.

Required regression tests exercise correct scope attachment, concurrent factors,
unknown/control distinctions, mixed populations, MHC role separation, measurement
units/encoding, absent versus zero observations, rejected surrogate-as-exact joins,
technical-versus-biological replication, source-field reconciliation and frozen
metadata round-trip into downstream prediction exports.

A frozen comparison report must answer: which samples differ, on which supported
axes; which measurements are genuinely paired; what quantity is predicted; what
the search could observe; and which distinctions remain unresolved. This is an
acceptance requirement for the Hitlist/Presto integration, not a claim that the
pending v2 work has already passed it.
