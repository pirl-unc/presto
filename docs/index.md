# Presto

Unified immunoinformatics for pMHC presentation and T-cell recognition.

## Start Here

- Single authoritative I/O design: [model_io_contract.md](model_io_contract.md)
- Current architecture: [design.md](design.md)
- Assay policy summary: [assay_modeling_contract.md](assay_modeling_contract.md)
- Assay source inventory and supervision map: `assay_learning_scheme.md`
- Training specification: `training_spec.md`
- TCR encoder specification: `tcr_spec.md`
- CLI usage: `cli.md`
- Repo inventory and retention guide: [repo_inventory.md](repo_inventory.md)
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

Sequence-only encoding permits downstream biological conditioning. The target
keeps presenting versus repertoire-selection MHC and all species roles distinct.
These richer context roles are not all implemented: see the I/O contract and
[issue #46](https://github.com/pirl-unc/presto/issues/46). Output availability
does not imply label support, calibration or a validated complete corpus.
