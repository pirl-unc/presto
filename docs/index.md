# Presto

Unified immunoinformatics for pMHC presentation and T-cell recognition.

## Start Here

- Architecture specification: `design.md`
- Canonical assay modeling contract: `assay_modeling_contract.md`
- Training specification: `training_spec.md`
- TCR encoder specification: `tcr_spec.md`
- CLI usage: `cli.md`
- Mouse overlay provenance notes: `notes/mouse_mhc_overlay_sources.md`
- Implementation-status audit: `../TODO.md`

## Summary

Presto is organized around one shared biological latent path:
- class/species inference from MHC sequence,
- class-specific processing,
- class-symmetric binding/stability latents with class-probability-calibrated class-compatible readouts,
- class-specific presentation,
- CD8/CD4 recognition and immunogenicity branches,
- output-side assay-specific readout heads.

The canonical production training strategy is unified mixed-source training with time-varying task/regularizer weight schedules.
The TCR-conditioned pathway is currently future work and is not active in canonical training/inference.
For canonical assay modeling, assay-selector metadata is not a model input.
