# Presto

Presto uses one shared sequence model and one observation-manifest interface for
new identity pretraining, MHC-I/MHC-II binding, cellular/MS endpoints and paired
TCR tasks. Endpoint configuration selects readouts within that same model.

## Start here

- [Shared model and observation contract](modular_model.md)
- [One training interface, migration and capability limits](unified_training.md)
- [CLI](cli.md)
- [Study curation](study_curation_contract.md)
- [Frozen Hitlist inputs](frozen_hitlist.md)
- [Scoped proteomics inputs](scoped_proteomics.md)
- [Repository inventory](repo_inventory.md)

The full ordered peptide supplies class-II flank context around every nine-residue
register; class I uses the whole peptide. Both classes share the encoder, fusion
and trainer. Shared software does not establish predictive transfer or improved
accuracy. The new checkpoint contract has not been scientifically qualified.

Historical default-model design, training and TCR-wrapper pages are retained with
scope banners. Their outputs and claims must not be treated as a description of
the current model. Existing experiment records retain their source/version pins.
