# Assay sources and supervision

This is a descriptive guide to current training, subordinate to the
[model I/O contract](model_io_contract.md). For exact active task definitions
read `LOSS_TASK_SPECS` in [train_synthetic.py](https://github.com/pirl-unc/presto/blob/main/scripts/train_synthetic.py),
which the canonical unified trainer uses. Historical dataset counts belong in
the [experiment registry](https://github.com/pirl-unc/presto/blob/main/experiments/experiment_log.md), not an undated claim
about today's corpus.

## Source paths

| Source path | Current use | Limits |
|---|---|---|
| Merged TSV (default primary source) | Normalized binding, kinetics, stability, processing, elution, T-cell and receptor-evidence records when present | Context and lineage depend on exported columns; not equivalent to Hitlist |
| Hitlist (explicit alternative primary source) | Curated binding, kinetics, stability and elution with mapping/flank context | Not the complete T-cell/TCR/processing corpus |
| IEDB/CEDAR raw exports | Inputs to source parsing/merge | Inclusion and label support must be audited in the actual merged artifact |
| VDJdb/McPAS | pMHC-only receptor evidence when included | Not sequence-conditioned TCR matching; positive evidence is not balanced negative supervision |
| mhcseqs / MHC reference resources | MHC sequence resolution and molecular identity support | Reference availability does not prove every source allele is resolved |
| Optional bulk non-MHC MS | Excision/detectability proxy supervision under capture/digestion routing | Disabled by default; candidate negatives and validation incomplete |
| Other receptor/structure/B-cell resources | Auxiliary utilities or future integrations | Do not claim canonical receptor matching from their availability |

`merged_tsv` and `hitlist` are mutually exclusive primary selections, not
opportunistically unioned based on local files. Bulk-MS is separately enabled.
Use the run's normalized funnel, lineage/support audit and split metadata to
describe what was actually used.

Merged binding-format rows route by their declared measurement: `value_type`,
then `assay_type`, then an exact measurement label in `assay_method` when both
are absent. Supported concentration, rate and stability labels retain their
existing units and censor qualifiers. A missing scalar remains a missing label
in that family; it does not become elution. A scalar from structural or
qualitative-binding measurements does not become an nM affinity target.

Binding-format presentation rows require an explicit MS method or a recognized
MS acquisition term. Presentation measured by Edman degradation, coelution or
T-cell recognition, and presentation with an unknown method, remains outside
the MS objective. An explicit `record_type=elution` declares the existing elution
contract directly. Acquisition subtype detection happens after that distinction.

Qualitative binding, structure, equilibrium association constants and unknown
measurement types remain separate source buckets. They are preserved in assay
exports and counted as unsupported in the merged loader's `skipped_by_reason`.
That dictionary also reports missing supported labels and required context;
its mutually exclusive counts reconcile with the compatibility aggregate
`skipped_unroutable_or_missing_label`. The normalized funnel exports the detailed
partition instead of also counting the aggregate. These omitted observations
may support future objectives; they are not negative training labels today.

## Current target-to-output mapping

| Observations | Output/loss path | Caveat |
|---|---|---|
| Binding concentration | `assays.KD_nM` plus family-specific targets; censor-aware regression | Generic binding also pushes heterogeneous concentrations toward KD |
| KD/IC50/EC50/proxy labels | Corresponding assay outputs where labeled | Proxy keys can alias KD under merged grouping |
| Binding descriptor metadata | Selected `binding_assay_panel_*` column, qualifier-aware regression | Entries are per-axis binding tracks, not complete kinetics/stability panels |
| kon/koff | Log-rate outputs, MSE | Censor-aware kinetic handling remains pending |
| Dissociation half-life/Tm | Log-minutes/normalized-temperature outputs, censor-aware regression | Method-specific measurement separation incomplete |
| Processing labels | Processing BCE where supported | Sparse supervision; do not infer complete pathway identification |
| Elution/MS | Elution, presentation and MS-related objectives; bag Noisy-OR on supported channels | Presentation is proxy-supervised by observation, not separately measured on every row |
| T-cell response | `tcell_logit` and immunogenicity BCE | Shared labels do not independently identify recognition and immunogenicity |
| T-cell assay metadata + response | Selected fixed `tcell_panel_logits` column with response BCE | Metadata is a selector, not a classification target; per-axis approximation |
| Receptor evidence | `tcr_evidence_logit` BCE and method-category auxiliary | No individual TCR input |
| MHC identity | Class/species/fine-type auxiliaries | Some current source-derived targets still conflate host/molecular roles |
| Bulk-MS | Excision/detectability proxy objectives | Generated negatives are not measured non-detections |

Detailed output units and complete ordered vocabularies are maintained in the
I/O contract. Unlabeled tasks are masked, not assigned negative labels. A
supported tensor with zero rows is an unsupported prediction for that run.

## Process versus measurement

Binding assays do not share the same generative observation equation as
immunopeptidomics. Elution can depend on presentation and detection; bulk protein
MS also depends on source abundance, digestion and acquisition. Current code
implements only parts of this decomposition.

In-vitro protease P1 profiles can be pinned by configuration. A pinned rule is
a prior, not a finding learned from the data. The in-vivo excision implementation
has trainable profiles and a class-I presentation contribution; this does not
prove cleavage, abundance and detectability have been separated.

APC interventions/cytokines are biological facts, whereas assay measurement
descriptors stay output-side. Preserve original annotations so coarse current
categories can be replaced without fabricating missing state.

## Validation requirements

For each trained output report actual train/validation/test support, qualifier
counts, real/synthetic provenance and class balance. Reconstruct measurements
in their declared units; score censored and exact observations appropriately.
Preserve selected-checkpoint provenance and per-example source/mapping lineage.

T-cell selected-column extraction uses the same task resolver as the loss.
No model-quality claim follows from correct loss wiring alone. Complete corpus
integration, context-role harmonization, supported negatives, joint assay
configuration modeling and held-out validation remain tracked in
[issue #46](https://github.com/pirl-unc/presto/issues/46).
