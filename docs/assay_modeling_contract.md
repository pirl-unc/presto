# Assay modeling policy

This policy separates biological inputs from measurement descriptors. The
[model I/O contract](model_io_contract.md) inventories the historical model;
[the shared model](modular_model.md) documents the current training path.

## Implementation status

Shared-v6 adds a versioned registry of joint assay tracks to the common model.
Each declared quantity produces a fixed panel using preparation, mechanism and
detector embeddings plus a shared latent projection of the candidate features.
Metadata selects supervision after panel computation. Pooled and scalar controls
use this same training path. Missing/conflicted descriptors use the learned
quantity reference; unsupported requested tracks fail. Track-specific supervision
and parameter-update evidence are saved and audited.
Legacy factorized heads/panels do not run in the canonical trainer. The purified
adapter excludes cellular methods; this does not establish cellular assay
qualification elsewhere in the model.

The [restoration specification](https://github.com/pirl-unc/presto/blob/29fb857e/experiments/agents/codex/plans/2026-10-29_shared-assay-measurement-layer.md)
documents the fixed joint assay outputs and matched comparison. The software is
implemented and tested; predictive improvement is not yet measured. Frozen v5
fits retain pooled method outputs, with historical inference preserved.
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
