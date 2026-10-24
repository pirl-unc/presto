# Complete the modular architecture and training contract

Authority: the user's active goal of 2026-10-20, frozen issue #130, and
`/Users/iskander/.codex/attachments/a773b5ef-eb76-4fb1-8985-25909bdb3eaf/pasted-text-1.txt`.
The attachment was read in full. It is the review gap list, not a replacement for
the broader goal. This dated addendum preserves the original frozen issue.

## Scope and completion evidence

Deliver one coherent model/preparation/training/inference family for all admitted
subsets: class-I/II affinity and stability; cellular presentation conditioned on
APC species, cell type, cytokines, delivery and independently overridden machinery
components; bulk digestion and MHC-elution observations sharing an MS-response
module; paired TCR–pMHC scoring. Early shared representations receive supervised
peptide-source, MHC class/species/pairing, TCR species/pairing and CD4/CD8-origin
pretraining. Predictions and effective supplied/overridden values stay distinct.

Completion requires real data admission, executable quality/coverage gates and
recorded fitting/evaluation evidence, in addition to software correctness. No
synthetic fixture, supported head name or nonconstant random output substitutes
for a qualified biological result. Missing external data remains explicitly
incomplete while independent implementation/data work proceeds.

## Implementation decisions

1. **Training support:** retain tensor gradient/update evidence and add input
   feature/category exposure plus gradient/update coverage. Track only active
   endpoint masks. Unobserved feature effects may not be activated at serving.
   Fixed padding and structurally inapplicable columns are identified explicitly;
   do not claim all elements in an embedding/weight tensor learned from one norm.
   Every run closes with a machine-readable pass/fail/incomplete gate. A failed
   coverage gate preserves artifacts and rejects normal successful closure.
   Diagnostic exceptions must be declared with reasons before running; their
   checkpoints cannot acquire validated support through that exception.
2. **Preparation:** define explicit source-measurement lineage independent of
   export-row ID. Deduplicate copied outcomes before partitioning; preserve true
   replicates and flag contradictory repetitions. Produce per-row admitted,
   duplicate, excluded and deferred dispositions, counts by target/class/source/
   unit/qualifier/split, and leakage/overlap reports. Reuse existing raw loaders and
   strict groove extraction, never the legacy clipped target path. Preserve
   sample/control/parent/clone/donor/study and reconstruction provenance.
3. **Early supervision:** add peptide-only multilabel source-category outcomes
   using Weirdo's actual dataset/labels (membership may overlap). Keep these labels
   distinct from exact species roles. MHC and TCR chain-pair tasks require explicit
   pairing evidence; reference co-occurrence/generated mismatches remain declared
   representation objectives, not measured biochemical compatibility. CD4/CD8
   origin uses phenotyped cells, not candidate MHC class. Supplied and override
   inputs affect downstream context, never overwrite stored predictions/labels.
4. **Cell context:** add declared cell-type and cytokine slots with missingness,
   independently controlled machinery states/activity, and sourced/versioned
   reference profiles. Cell type/species provide visible defaults, never evidence
   that a particular component is intact. Explicit sample annotations/overrides
   win. Cell-line identifiers remain grouping/provenance only. Every context field
   has an endpoint mask and support check; purified binding and supplied-pMHC
   compatibility remain invariant to irrelevant processing context.
5. **Measurements:** add quantitative bulk/elution, detector and digest/processing
   endpoints with explicit units, transformations and censor bounds. Avoid a
   universal intensity scale across incompatible normalization protocols. Add
   prospective digestion/preparation metadata separately from acquisition. No
   quantitative-to-binary coercion; no pseudo-processing targets from detection.
   Preserve supported peptidoform scope and reject unsupported chemistry.
6. **MS decomposition:** one shared detector function across native elution and
   bulk routes, with dedicated biological/digest branches and one observed loss.
   Identification of separate MS bias requires actual anchors/comparisons and
   transfer checks. Without these, report components as indirect, unqualified
   diagnostics. Retain anchor supervision when the encoder changes.
7. **Subset continuation:** support compatible declared task subsets/extensions,
   explicit frozen dependencies and exact resume (optimizer, RNG, sampler cursor,
   selection/coverage state and source pins). Changes to vocabularies/scales need
   explicit compatible migration. Freezing a head alone does not freeze its
   changing upstream function. The inference contract verifies all support and
   validation-domain requirements separately from training support.
8. **Evidence:** retain task loss shares, per-task encoder gradients, input
   sensitivity, register diagnostics, raw/canonical targets and allele/study
   identities in evaluation outputs. Implement identity/calibration, bounded
   memorization, mean/ridge comparisons, contraction, class-specific and retained-
   task noninferiority gates. Test-use policy and independent evaluation domains
   remain explicit. Register every real fit/benchmark under `experiments/`.

## Delivery sequence (all items remain in scope)

- Reproduce and fix the attachment's three software/data failures with regressions.
- Complete the observation/pretraining/context/output schemas and masked model
  paths, then exercise every supported subset through preparation and inference.
- Inspect latest Hitlist and Weirdo releases/data, pin actual sources, and prepare
  qualified cohorts with complete dispositions. Confirm external TCRvdb access
  rather than inventing its raw schema. Record confirmed upstream gaps.
- Complete continuation, freezing and evaluation/promotion gates.
- Register and run bounded real-data fitting and comparisons; debug failures with
  preserved evidence. Pretraining is part of the requested architecture, while
  supplied-metadata controls isolate assay fitting from auxiliary errors.
- Qualify matched APC comparisons, MS anchors/transfer and paired TCR scoring as
  their actual data permit; never replace these with an IC50-only success claim.
- Reconcile docs/issue/PR scope, run final software checks and fresh-process
  checkpoint verification, publish the updated PR and inspect exact-head CI.

## Initial authoritative state and continuation audit

The PR is draft/open at `6f935093`; its four CI checks succeeded. The isolated
worktree contains only the previous review's documentation edits. The original
checkout and shared Python environment will be preserved. The previous answer
was explanatory, with no implementation progress; this goal turn revalidated
the state and starts the next concrete changes. No active training job is assumed.

Verification: targeted failing-before/passing-after contract regressions, followed
by relevant complete regression and strict docs/lint. Real predictive checks have
their own frozen experiments; software fixtures do not qualify endpoints.

## Clarification after the identity/IC50 objection

"Identity plus IC50" is a diagnostic subset, not the architecture's purpose or
the full-goal completion criterion. Identity denotes independently supervised
sequence annotations, not record identifiers. The assay head always retains rich
sequence vectors. Successful annotation prediction is not evidence that the
encoder captures peptide-MHC interactions or that the affinity fit is useful.

For the affinity diagnostic, distinguish two experimental factors: auxiliary
pretraining on/off and auxiliary replay during assay fitting on/off. Hold the
assay cohort, partitions, supplied metadata, assay architecture, assay-update
budget and model seeds fixed; report additional auxiliary compute separately.
Supplied metadata isolates representation transfer from metadata inference.
Evaluate predicted metadata as a separate serving comparison after that test.
An assay-only supplied-metadata control must not wait for successful auxiliary
pretraining. These are prospective comparisons, not executed experiments; exact
recipes and thresholds must be registered before running.

Detached categorical probabilities prevent direct assay gradients into auxiliary
readout weights, but the shared encoder still changes under assay supervision.
Retained annotation quality therefore needs measurement; replay is a proposed
mitigation, not a guarantee. Keep the requested auxiliary pretraining interfaces
while measuring their benefit independently of the IC50 debug and the remaining
cellular, MS and TCR qualification.
