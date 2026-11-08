# Assay modeling policy

This policy separates biological inputs from measurement descriptors. The
[model I/O contract](model_io_contract.md) inventories the historical model;
[the shared model](modular_model.md) documents the current training path.

## Implementation status

Shared-v7 uses one nonlinear affinity decoder for IC50, KD, EC50 and the
separately labeled KD-via-IC50/KD-via-EC50 proxies. For each candidate it
concatenates the shared biological latent with separate 16-dimensional vectors
for quantity, preparation, mechanism and detector, then applies
`Linear -> GELU -> Linear -> scalar`. There are no independent affinity heads
or small residual corrections. Existing train-only quantity scales and register
pooling remain in force. Assay metadata selects supervision after fixed-panel
computation; it never enters the biological encoder.

Factorized decoding is the sole new-training path. Registered categorical values
have named learned vectors; unknown/conflicted method slots are fixed zeros and
use the same decoder. This unknown-method reference is not thermodynamic KD.
Unsupported requested tracks fail. Actual gradient/update evidence and exposure
counts distinguish directly supervised combinations, indirectly trained
combinations whose factors learned elsewhere, and random/untrained factors.
Known complete methods must be declared in the manifest registry. Source tags
remain unreviewed unless independently verified. `database_curated` means the
literal method, response, and assay ontology identity were reconciled to the
official database export; it does not mean the primary protocol was reviewed.
Review evidence is provenance and never a model input. Detector categories do not
identify the full laboratory protocol.

The shared factorized measurement decoder is concentration-oriented: lower
continuous output means stronger binding. Its qualitative-binding readout has a
fixed negative polarity so stronger binding produces a higher binary logit.
Labels are never inverted, and the polarity changes neither concentration
outputs nor their inherited checkpoint tensors.

Frozen v5 and v6 inference preserve earlier checkpoints, including v6 pooled,
scalar-calibration and low-rank residual models. They do not run in new training.
The new decoder is an architectural decision, not a claim of measured superiority.

Shared-v7 can also allocate a separate fixed T-cell response panel. Each declared
joint `(family, response, method)` track has named family, response, and method
vectors. The fixed cellular pMHC representation and resolved biological context
are concatenated inside the new `Linear -> GELU -> Linear` positivity decoder;
no inherited fusion tensor changes shape and no assay descriptor enters that
biological context.
The same fixed panel emits a conditional ordered-intensity readout for graded
positive observations. Its two cumulative thresholds are monotonically ordered
by construction; negatives and ungraded positives never receive fabricated
intensity labels. The complete panel is computed for every eligible cellular
pMHC latent, and the row descriptor selects only the supervised output.

The T-cell decoder shares neither parameters nor output descriptors with the
affinity measurement decoder. It is allocated only when a checkpoint declares
T-cell tracks, so checkpoints without such tracks retain the same state dict and
inference. Fixed-parent training leaves only `tcell_assay_layer.*` trainable.
Per-track binary and conditional-ordinal exposure/update evidence is
stored separately, and serving rejects an unqualified panel. This is currently a
software capability: the public checkpoint predictor returns the complete
qualified positivity and conditional-intensity panel, including descriptor-free
counterfactual requests. No T-cell fit or PRIME superiority claim is implied.

Cellular binding,
stabilization and functional response require their own process definitions;
shared detectors or IC50 units do not make them purified binding measurements.

## The distinction

**Sequence-only encoding; biologically conditioned downstream predictions.**

- Residue encoders receive sequences and structural position/missingness only.
- Biological state can affect the appropriate downstream component.
- Measurement apparatus describes fixed output tracks and selects supervision,
  but never becomes an observed per-example input feature.

An APC knockout, APC cytokine exposure, presenting MHC set and repertoire-selection
MHC set describe biology. Fluorescence versus radioactivity describes how it was
observed. APC species, MHC origin, receptor species, repertoire-system species and
antigen species are distinct roles.

Do not classify a source column solely by its name. “Stimulation” can mean an APC
exposure or a T-cell assay protocol; “peptide format” can mix antigen delivery
with measurement setup. Normalize the biological facts separately from the
observation descriptors. Unknown is not an untreated control.

## Fixed outputs and supervision

A fixed assay column is the same function for every example. Assay metadata may
select which column is compared with the measured outcome. It may not redefine
the other columns. Predicting a method ID with cross entropy is not predicting
a response under that method.

Biological counterfactual tracks can coexist with conditioning on observed
state. Predicting across APM states does not prohibit taking a known APM
intervention as input to the relevant downstream biological component.

## Historical model details

The former model's coarse cellular categories and generic host inputs do not
implement the full role-specific design. Its T-cell panels are per-axis response
outputs, not a joint configuration registry. Their unknown-reference baseline is not a
mathematical marginal.

That model's input signature, assay vocabularies, transformed units,
aliases, proxy labels and pending capabilities are listed once in
[model_io_contract.md](model_io_contract.md).
[Issue #46](https://github.com/pirl-unc/presto/issues/46) tracks the gaps.
