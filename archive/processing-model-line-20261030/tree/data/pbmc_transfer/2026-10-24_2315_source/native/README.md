# Hitlist portable CI corpus

This is a content-verified snapshot of one dataset generation. The original artifact contract and source hashes are retained. It is a software regression corpus, not a qualified training split.

| File | Meaning |
| --- | --- |
| observations.parquet | Positive MHC-MS evidence |
| binding.parquet | Binding endpoints, units and censoring |
| other_assays.parquet | Other, negative and unknown assay evidence |
| observation_contributors.parquet | Every retained source contributor and reference |
| peptide_mappings.parquet | Alternative protein occurrences, not independent observations |
| peptide_mappings_meta.json | Mapping contract, source index hashes and reference limitations |
| bulk_proteomics.parquet | Bulk observations; fraction counts do not establish detectability |
| line_expression.parquet | Cell-line expression evidence |
| observations_meta.json | Original build contract, source hashes and complete file inventory |

Cell types, conditions, MHC typing and assay restrictions retain their distinct roles. Private source paths are omitted. Mapping input hashes replace machine-specific mtime/path comparisons in the portable receipt; this does not make the downloaded snapshot a fresh local build. Actual cohort coverage, partitions, biological independence and training performance require separate qualification.
