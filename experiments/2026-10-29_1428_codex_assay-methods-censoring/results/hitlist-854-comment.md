The current Presto temporal-cohort audit quantifies the same gap in a newer full-export build (Hitlist 2.0.40, acquired 2026-10-28).

The canonical MS export has 5,440,009 rows: 4,904,983 IEDB and **535,026 supplementary rows**. Every supplementary row has null `has_ptm`. They therefore all fail the consumer's explicitly-unmodified peptidoform contract; all **1,641,063 original MS TRAIN observations** have raw source `iedb`. Downloaded/indexed supplementary data should not be reported as trained data.

Of those deferred supplement rows, 332,866 class-II and 9,954 class-I rows already have exact allele provenance plus monoallelic restriction evidence. These are promising metadata counts, not a claim that all are independent, eligible training examples. Source-backed peptidoform curation and source/condition/overlap checks remain necessary; do not blanket-fill missing PTM status with False.

Export SHA256: `0f6fe591e6f5c12975ec89bdbdc8e17b1b910b26e24e27a80ded09a6576e114e`.
TRAIN MS SHA256: `bc709276b36bd456a09451deba140aeb618c9614a79505981e7567d1a0b0e203`.

Reproduction, source ledger and independent reconciliation are in Presto's `experiments/2026-10-29_1428_codex_assay-methods-censoring/`; no held-out numerical outcomes were used and no source labels were changed.
