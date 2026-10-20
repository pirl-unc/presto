# Canonical affinity target-policy audit — prospective plan

Use the public corrected-human IC50 source/context table (2,614 observations,
SHA-256 59c73feeec2ba00041fcc3f8a91b6701ef31bd3f5d3ce6db01de188c45f44583)
and archived model configuration, fetched and verified through datacache. Preserve
all source strings, sample IDs, measurement units/operators and published train,
validation and test membership. Use the candidate Presto 0.1.16 canonical collator
and OutputCoverageCensus, without constructing a model. Freeze a clean recipe
commit before execution; record its identity, imported source, dependency versions,
hardware, wall time and peak RSS. Use local CPU, one Torch/BLAS thread, no GPU,
no workers, a ten-minute alarm and batches of 128.

This is a label-only reconstruction: MHC class and allele identity are retained,
but MHC sequences are not fabricated or loaded. It does not certify full model
inputs, reproduce every auxiliary label or requalify the original manifest.
Only IC50 direct-objective results are compared with the independent published
posthoc audit; other objective exposures remain visible but are not an additive
source-observation count. No synthetic labels, optimization or predictions.

Acceptance: all 2,614 raw/effective values and original censor directions match
the released 0.1.13 audit; exact clipped sample identities match, with split counts
157/31/23 and clipped exact counts 39/8/7. Export all canonical audit rows, a flat
count table, full metadata for clipped direct observations, executable policies,
and a summary. Context table must preserve citations, assay, MHC, cell, condition
and unknown donor/specimen/acquisition fields verbatim. All outputs materialize
as one immutable datacache generation keyed by recipe and producer; publish a
hash inventory and reproducibility bundle. No predictive metrics or preferred
model exist for this audit. Existing lineage limitations remain unchanged.
