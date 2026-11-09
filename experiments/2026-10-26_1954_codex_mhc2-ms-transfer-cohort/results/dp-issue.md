PMID 35251023 (Laghmouchi 2022) describes 13 defined HLA-DP paired constructs in K562 populations. Current Hitlist curation collapses the study to one generic unperturbed/untreated arm, leaving the engineered MHC context unspecified.

Primary Methods: https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2022.831822/full (cell populations and HLA-DP ligand identification sections). This is a metadata/arm-lineage request; it does not invalidate the measured MS detections or require inference of an antigen-processing route.

### Current evidence

Read-only Hitlist 2.0.42, commit `13b91424295e01286db46e8e9818fcc2fe816834`, `hitlist/data/pmid_overrides.yaml`:

- One `ms_samples` entry: `cell lines (multiple HLA-DP genotypes)`.
- `condition_id: cell_lines_multiple_hla_dp_genotypes`.
- `condition: unperturbed`, `condition_control: untreated`.
- `mhc: HLA class II`; no source-specific engineering/paired-construct context in this arm.

The older Hitlist 1.78.0 human export already retains explicit paired restrictions and monoallelic/K562 evidence for 39,293 rows admitted by Presto's class-II MS adapter. That older export is not evidence that current arm curation distinguishes the populations. Export SHA256: `f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd`.

### Requested resolution

- Curate source-backed K562/introduced-HLA context and distinguish the reported paired constructs using actual primary sample identities.
- Preserve paired molecule, sample, control and contributor lineage through export. Join individual IEDB rows only where source evidence supports it; unresolved row-to-arm identity must remain explicit.
- Keep engineered cell context separate from peptide-specific restriction evidence and from processing-pathway assertions.
- Add regression fixtures for resolved and unresolved arm matches. Review any expression-reference assignment that depends on whether the sample is engineered, without inventing source-matched expression measurements.

#593 tracks sibling engineering-field drift in other studies; its current study list does not include 35251023. A repository search for this PMID found no existing issue. This single generic-arm gap can evade sibling-based drift checks.

Consumer evidence: Presto `experiments/2026-10-26_1954_codex_mhc2-ms-transfer-cohort/results/inventory/current-curation.json`, branch `codex/mhc2-improvement-20261026` (publication pending).
