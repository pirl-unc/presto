# Assay modeling policy

The authoritative specification is [the model I/O contract](model_io_contract.md).
This page is a short policy guide, not a second normative inventory.

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

## Current limits

Current coarse cellular categories and generic host inputs do not implement the
full role-specific design. T-cell panels are per-axis response outputs, not a
joint configuration registry. Their scalar unknown-reference baseline is not a
mathematical marginal.

The complete implemented input signature, assay vocabularies, transformed units,
aliases, proxy labels and pending capabilities are listed once in
[model_io_contract.md](model_io_contract.md).
[Issue #46](https://github.com/pirl-unc/presto/issues/46) tracks the gaps.
