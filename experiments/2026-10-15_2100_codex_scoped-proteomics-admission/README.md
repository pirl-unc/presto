# Published scoped proteomics admission audit

Agent: Codex / GPT-6. Registered 2026-10-15 America/New_York.
Plan: [admission contract](../agents/codex/plans/2026-10-15_scoped-proteomics-admission.md).

This checks the actual public PXD013455 HeLa Trypsin/P archive through the new
adapter and canonical dataset/collation/support audit. It does not fit a model,
assign partitions or evaluate held-out performance. No validation/test metrics or
prediction dumps are expected; those require the next canonical trainer work.

Input manifest SHA-256:
`441f3c66c6e6bc6a90c7cb8c54d0131676c3cb4e7a04ddbd9346061ad8f9d84b`.
Select 10,000 unique peptides by reservoir seed 42 after complete occurrence
collapse. Direct MS/MS, matching and scoped absence are distinct proxy families.
No excision, biological replicate or depth labels. Multiple protein occurrences
mask flanks/parent while preserving a complete sidecar lookup. Citations,
cell/assay/search/reference conditions and unresolved identities remain pinned.

CPU only; one Torch/OpenMP thread; 15-minute wall limit. No GPU, pretraining,
synthetic data or loss optimization. Source code is committed before launch;
`reproduce/launch.json` captures its commit, paths and environment overrides.
All reusable audit data are published to shared datacache; results here contain
small summaries and hashes/locations, not duplicate data tables.

## Results

Admission passed against the public pinned archive, using code commit
`e4fe7e19ab3208fdc74e54b9dc2b16fcae35ac24`. All 10,000 selected peptides retain
candidate lookup and frozen scope identity. The canonical lineage audit reports
zero issues and zero duplicate sample IDs. The 1,114 positives additionally retain
original observation IDs; the 8,886 absent candidates do not invent measured IDs.
The canonical census's `untraceable_rows` refers to missing measured-observation
IDs, not missing candidate linkage. All labels count as proxy support; direct
intrinsic-detectability support and excision/depth labels are zero.

| Evidence | Full unique peptides | Admitted peptides |
| --- | ---: | ---: |
| Direct MS/MS identification | 143,216 | 960 |
| Match-between-runs identification | 20,778 | 154 |
| Not observed in this search | 1,287,164 | 8,886 |
| Total | 1,451,158 | 10,000 |

The source has 1,656,863 candidate occurrences and 172,033 observations, supported
by 12,287 leading-razor parents. Of the collapsed peptides, 153,980 have multiple
candidate occurrences and therefore mask parent/flank inputs. No invalid peptide
sequences were excluded. The 8,039 source observations outside the candidate
window remain in the source bundle, not as extra labels.

Admission plus canonical support audit took 74.38 seconds; measured process peak
RSS was 595,787,776 bytes (568.2 MiB), before audit-artifact publication. Hardware:
macOS arm64 CPU, one Torch thread, no GPU. Versions: Presto 0.1.4, Hitlist 1.75.0,
datacache 1.27.1, pyarrow 25.0.1 and Torch 2.14.1. The existing environment was
not modified; PYTHONPATH selected the isolated feature checkout and previously
verified public Hitlist/datacache wheel imports. Exact invocation/import paths
and environment overrides are in [reproduce/launch.json](reproduce/launch.json).

[Summary](results/summary.json) preserves the complete counts and limitations.
[Artifact receipt](results/artifacts.json) lists SHA-256, byte sizes and shared
locations for admitted TSV/parquet, source receipt, canonical support/coverage
JSON and CSV, and README. All reusable tables are datacache-owned. Source and
projection sidecars remain available through the pinned receipt.

This establishes adapter admission and collation, not training readiness. No
fitting, synthetic data, objective optimization, held-out partitions or prediction
dumps were produced. Validation/test metrics are intentionally not applicable.
One HeLa search cannot qualify experiment/donor/specimen-independent performance,
MBR is not direct MS/MS, and leading-razor support is not unique parent proof.
No model baseline is promoted.

Status: admission audit closed; public artifact publication follows PR #83 release.
Next Step: canonical trainer integration, frozen within-search membership, explicit
shared/unknown lineage audits, then bounded fitting and held-out baselines (#70).
Open Questions: cross-experiment detectability generalization remains unsupported.
