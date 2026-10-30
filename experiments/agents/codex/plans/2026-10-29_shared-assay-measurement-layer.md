# Restore the shared assay measurement layer

Date: 2026-10-29. Agent/model: Codex / GPT-6.
Status: implemented and software-verified; matched predictive comparison pending.

Implementation decision (authorized 2026-10-29): preserve quantity baselines and
compute all declared joint method tracks on each shared candidate latent before
the existing register pooling. Calibration uses one scalar per joint track;
factorized decoding uses a shared low-rank latent projection and quantity,
preparation, mechanism and detector embeddings. Zero-initialize the projection
so all variants start at the same baseline. Assay selection happens only after
the complete panel is computed. Conflicted/incomplete methods use the quantity
reference, not a fabricated protocol. Registry-free manifests retain their
quantity outputs; declared registries default to factorized decoding. Freeze v5
for historical inference and give the new contract a distinct model version.

## Verified omission

The user correctly identified a consolidation gap. The March 11 assay-head plan
proposed a shared binding latent plus factorized assay embeddings. The March 16
plan and historical panels require these descriptors to define fixed outputs,
rather than become per-example inputs to the biological encoder.

Canonical shared-v5 shares its encoder, residue interaction and pMHC fusion, but
literal IC50, KD, EC50 and proxy outputs each have an independent linear readout.
It has no preparation/mechanism/detector embeddings. `assay_method_context` enters
provenance only; legacy `AssayHeads` panels do not execute in new training.
"Assay-specific" was misleading shorthand for these quantity-specific readouts.

The current purified adapter excludes cellular assay methods. Cellular IC50 is
not silently present in this cohort. Presentation/MS uses a separate endpoint.

## Common architecture and data contract

1. Keep one encoder, residue interaction, register implementation and trainer for
   both MHC classes. Assay descriptors must not change biological encoding or
   candidate enumeration.
2. Define fixed joint output tracks by quantity, preparation, mechanism and
   detector. Use factorized descriptor embeddings and one shared decoder of the
   sequence-derived latent, with a quantity baseline and a small latent-dependent
   residual. The same biological inputs must produce the same complete panel
   regardless of which track was observed. Metadata selects supervision only.
   Per-axis predictions alone do not define a joint assay configuration.
3. Purified competitive radioactivity and fluorescence are distinct tracks sharing
   preparation/mechanism factors. Cellular binding, stabilization, refolding and
   functional response need their actual process definitions. Use the existing
   biological context path where required; do not route cellular IC50/EC50 into
   purified affinity merely because units or detector match. Do not force cellular
   and purified outputs to agree through a blanket consistency loss.
4. Retain quantity-specific units/transforms and IC50/KD/EC50/proxy distinctions.
   The quantity-only reference for unresolved method metadata is neither proven
   thermodynamic affinity nor a mathematical marginal over protocols.
5. Normalize descriptors once in common admission, retaining original/effective
   values and evidence. Missing is unknown; source tags are not verified protocols.
   Known conflicts, including Hitlist #871, must not supervise disputed factors.
   Eligible quantity supervision can remain. Source/study IDs are audit/grouping
   keys, not biological predictor embeddings.
6. Keep explicit and reviewed inferred bounds in the common one-sided loss.
   Calibration does not replace source/assay-specific censor review.
7. Version the joint-track registry and checkpoint readout architecture. Allocate
   declared combinations, not an unchecked Cartesian product. Audit factor and
   combination support, gradients and actual updates. Distinguish supported
   outputs, compositional extrapolation and untrained parameters. Preserve exact
   historical checkpoint inference; never fabricate trained method embeddings.

Pure additive offsets cannot change within-assay Spearman for a fixed model.
The main hypothesis therefore requires a latent-dependent residual, alongside a
calibration-only control. Before implementation, fix whether the decoder scores
candidates before pooling or reads a common pooled latent; retain one pooling
implementation and verify selector-invariant fixed output columns. Statistical
register attention must not be described as physical occupancy.

## Prospective comparison

Register a timestamped experiment family before launch, with frozen source,
manifests, cohort hashes, runtime pins and exact invocation. Compare matched seeds:

| Condition | Measurement model |
|---|---|
| Pooled control | Existing quantity readouts, methods pooled |
| Calibration control | Quantity reference plus assay scalar offsets |
| Factorized decoder | Shared latent plus descriptor embeddings and latent-dependent residuals |

Keep rows, endpoint exposures, optimizer, budget, validation cadence and checkpoint
selection matched. Freeze residual regularization and budget before examining
evaluation results; record any tuning separately. Record parameter/runtime costs.
Do not combine this contrast with source expansion, new sampling or LR changes.

Retain inspected development holdouts for continuity. Current IC50 validation is
radioactivity-tagged only and cannot qualify fluorescence or cellular prediction.
Freeze additional source-separated evaluation support before fitting those tracks;
a new split carved from previously trained data is retrospective, not blind.
Audit peptide/source leakage. Unsupported evaluation cells remain unavailable.

Save per-example predictions and loss, exact-value Spearman/Pearson/RMSE, bound
violation metrics, binding-threshold metrics where meaningful, and ligand
AUROC/AUPRC. Report quantity × method × class and study-macro results alongside
pooled scores, with paired study-aware uncertainty and source-dominance checks.
Do not promote on pooled Spearman alone. Protect ligand performance.

Use H100!, at most two concurrent GPUs including existing jobs. Record actual
hardware/memory. Registered family `2026-10-29_2303_codex_assay-measurement-layer`
uses 4,000 updates per fit and varies IC50 method tracks while retaining all ten
outputs. Additional fluorescence holdouts are selected using source metadata,
with cross-output peptide embargo; they are retrospective development holdouts.
The running v5 curation comparison stays frozen. Complete registry closure before
claiming a winner.

## Verification

- Round-trip descriptors and reject invalid/conflicting combinations.
- Supervise only the observed panel cell; observed selector changes must leave
  every fixed panel prediction invariant for identical biological inputs.
- Verify sharing across classes/tracks, both censor directions, unknown/conflicted
  metadata, unsupported tracks, checkpoint round trips and untrained reporting.
- Independently recompute units and stratified metrics from saved predictions.
  Software wiring tests do not establish predictive gains.

## Evidence

[Purified fluorescence competition](https://pubmed.ncbi.nlm.nih.gov/16156661/)
shows that assay concentrations and reference peptide affect IC50.
[The MHC-I exchange study](https://pmc.ncbi.nlm.nih.gov/articles/PMC6760266/)
distinguishes purified binding, cellular binding and stabilization methods.
These support measurement separation, not an unmeasured decoder benefit.
