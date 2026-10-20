Presto's native Hitlist training route currently covers binding/kinetics/stability and MS elution; it explicitly returns empty processing, T-cell response and TCR-evidence record lists. Consequently these outputs cannot acquire measured direct supervision merely by increasing the native Hitlist sample cap. Other Presto source routes have loaders, but lack the same frozen native evidence/lineage contract.

Please inventory and expose measured cellular functional and antigen-processing evidence as distinct evidence families, preserving their actual assay meanings. This is separate from the paired receptor TCRvdb adapter and from treating positive immunopeptidomics as a processing assay.

Acceptance:
- Audit accessible IEDB/CEDAR T-cell and relevant processing source tables/supplements; report available endpoints, positive/negative/unknown counts, MHC restriction resolution and source-study coverage.
- Preserve assay method/readout, quantitative units and qualifier, effector/APC context, antigen format (peptide/protein), culture/stimulation context, and source-linked experimental identity where explicitly reported.
- Distinguish pMHC multimer binding, activation/cytokine/cytotoxicity, and direct processing/cleavage/transport measurements. Do not relabel absent entries, failed QC or unobserved MS peptides as measured negatives.
- Keep receptor-specific versus repertoire-level evidence distinct. Preserve measured restriction separately from donor genotype/candidate allele sets.
- Provide a versioned frozen export with all contributors, provenance and explicit unknowns, source-linked readable examples, per-family coverage and tests of target routing/units/censoring.

Related #678/#680 and pirl-unc/presto#48/#53. Start with an availability/coverage report; do not assert that all requested families exist in the present Hitlist inputs or that weak IC50 fit is caused by their absence.
