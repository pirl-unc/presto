# Qualify independent ERAP1 measurements before another biological fit

October 24, 2026, Codex / GPT-6. Follows the completed
`2026-10-24_0810_codex_erap1-trimming` experiment and Hitlist issue809.
The full objective remains precursor-relative peptide availability in APCs.
This plan defines the next source qualification step; it does not substitute
biochemical curve fitting or elution prediction for that objective.

## Evidence and hypothesis boundary

The fixed HAP1 sequence baseline has inner-CV RMSE0.596370 on ERAP1, versus
0.596436/0.596417 with exact/sampled trimming. All allocated physical parameters
update and independent recurrence verification passes. This supports stopping
sampler expansion on this particular biological model. It does not establish the
cause of its failure or prove that mechanistic processing is unhelpful.

Primary Chang2005 (PMID16286653, DOI10.1073/pnas.0500721102) and Evnouchidou2008
(PMID18987748, DOI10.1371/journal.pone.0003658) document substrate-context effects
absent from the residue-only rate model. They are source leads, not admitted numeric
training datasets. The PNAS article lists a supporting index and two PDFs; PLOS
exposes article figures. Browser retrieval errors do not establish inaccessible
original data, and a figure is not a table of raw replicates.

## Qualification contract

- Inventory original article/supplement/deposit assets, hashes, license and access
  status through shared Hitlist source conventions. Begin with non-pathogen model
  substrates. Retain source identity, selection and original measurements.
- Separate individual substrates from mixtures and unresolved libraries. A mixture
  does not produce individually supervised peptide records.
- Distinguish initial rates, endpoint fractions, kinetic constants and product
  time courses, retaining units, assay context, enzyme species/construct/variant,
  substrate/product lineage, replication, calibration and error definitions.
- Preserve unmeasured, censored, reported-zero, qualitative and unavailable states.
  Never fill values from prose, treat plotted means as replicates, or infer a
  catalytic rate from an arbitrary intensity or single endpoint fraction.
- Report deposited numeric data, graph-only evidence and author-request data
  separately. No automatic figure digitization or author messaging in this step.
- Do not create a Presto source parser. New source admission and tests belong in
  Hitlist; Presto records the pinned source/recipe manifests as a consumer.
- Record every candidate's compatibility and limitations, including whether
  measured selection/detection factors are present. Isolated enzyme measurements
  are not APC concentrations or MHC elution labels.

## Gate before new model implementation

A new experiment requires a frozen, source-verified numeric cohort with units,
replicate/dependency grouping, sufficient substrate/context variation and an
explicit mapping from assay to output. A fitting plan must specify independent
anchors for any newly introduced rate dependence rather than choosing a more
flexible feature set solely because it improves the already-seen HAP1 inner CV.
If a source offers only inaccessible or graph-only values, record that limitation
under809 and assess another independently measured source; do not invent targets.

The next model comparison must keep direct sequence and simple physical controls,
separate measured enzyme dynamics from cellular transport/selection/detection,
and preserve short/nonbinding products. Exact enumeration remains the numerical
reference. Add sampled gradients only after a fixed estimator gate, using the
same biological objective and reporting score counts, runtime and held-out error.
Do not add clipping/reuse/KL merely to repair a biological model miss.

HAP1 outer validation/test remain unopened until a separate prospective broader
comparison is frozen. Previous MAPP/native/macrophage tests stay closed. Final
qualification still needs actual APC data and identifiable precursor-relative
amounts; success on an isolated enzyme assay is only a component validation.

## Deliverables and current status

- [x] Record the negative biological result and preserve all fixed comparisons.
- [x] File/read back the shared upstream source request (Hitlist809).
- [ ] Inventory recoverable original numeric assets and semantic compatibility.
- [ ] Admit any supported measurements through shared Hitlist relations/tests.
- [ ] Freeze a new biological comparison only after the source gate passes.

No new dataset, quantitative labels, model or fitting run is claimed by this plan.

## Additional source lead verified after the initial plan

Maben2021, DOI 10.1038/s41467-021-25564-w, supplies a public 479,402-byte
Source Data ZIP with two workbooks (main and supplementary figures). Original
SHA-256: a9d7bcbae52b25e55aec5f725578238fd0e6719720153dc829b999dd4bde5d2c.
The linked correction DOI 10.1038/s41467-023-37300-7 concerns an acknowledgment.
The archive was retrieved and inventoried, but workbook measurements have not
been interpreted, admitted or used for training. Companion inventory JSON records
the original URL, retrieval time and exact member names/sizes. This is a source
availability lead; inspect measured versus computed data, reporter versus peptide
assays, replication and non-pathogen model-substrate provenance before admission.

Upstream record: https://github.com/pirl-unc/hitlist/issues/809#issuecomment-6384618997
Original local archive: /private/tmp/hitlist-erap1-809-assets/maben2021-source-data.zip.
