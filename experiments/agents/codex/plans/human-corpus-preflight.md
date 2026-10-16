# Prospective human-corpus frozen-input preflight

Agent/model: Codex (GPT-6). Characterize actual source admission, frozen partitions,
lineage leakage and output support; do not construct or fit a model. Source is the
public human-cohort-baseline-v1, native manifest SHA-256
039efc7854098951b5c10b3fb5b38294452eeb425e367ed1fafdeebddf34b38d.
It contains 4,071,463 human-MHC/human-peptide-source observations, lengths 8–25,
produced by released Hitlist 1.76.0. The producer predates #694/#696 corrections;
this run must retain that limitation. The complete source lineage/metadata,
original endpoints, units, censoring, cell types/conditions and allele roles stay
available in the public native bundle. No new original-source claim is permitted.

Use the canonical Presto frozen Hitlist route, no merged/live assay inputs,
no flanks/protein-context backfill, no generated negatives or synthetic auxiliary
samples, no bulk/proteomics mixing or MHC-only/UniProt augmentation. Cap binding
and elution at 5,000 records each and stability/kinetics at 1,000 each, reservoir
sampling with data_seed=42 (canonical source seed offset retained). Both MHC
classes remain eligible; strict sequence resolution filters are explicit. Pin
mhcseqs 2.6.13 and torch CPU 2.14.1, retain installed dependency versions and pip
report. Freeze measured-root membership before conversion, val/test fractions
0.1/0.1, split seed42, peptide_disjoint policy. Preserve mirrored source records,
known reused experiments and acquisitions in components; do not break large
components to manufacture balanced partitions. Retain parent graph and filtered
root IDs even when they have no final model sample.

Run canonical data_preflight_only with traceable-lineage validation. Collect the
full per-output/family/source/split coverage matrix, actual assignments and
membership, data funnel, lineage/root/model-peptide audits and exact regularization
settings. No supported-output declaration or minimum is relaxed after looking at
counts. This preflight declares no output training-ready: label support and
negative/qualifier coverage are measurements to inform the next prospective fit.
All regularization weights are explicitly zero even though no optimizer runs.
No predictions/loss performance exist because no model is constructed.

Use one bounded Modal CPU job: eight CPUs, 64 GiB RAM, 512 GiB ephemeral disk (Modal minimum), no GPU, 50-minute function
limit and 45-minute work alarm, one Torch/BLAS thread, no data-loader workers.
The source already exists in the shared volume; verify native hashes before use.
MHC catalog is rebuilt with mhcseqs 2.6.13 from the two official commit-and-hash-pinned FASTAs in reproduce/mhc-sources.json. The complete raw/full catalogs and merge/validation reports are retained. Package pins alone do not identify these sequence inputs.
All reusable inputs, MHC caches, partitions and output bundles use datacache under
a separate shared root. Save failures, stdout/stderr, exact wheel/code/config
hashes, dependency versions, runtime and runner RSS. Publish complete outputs and
per-file inventories, with small summaries and closure in the experiment registry.
An inconclusive independence audit or unsupported target is a result, not a reason
to guess identities or change thresholds. #678/#680 and Presto #48/#53 stay open.
