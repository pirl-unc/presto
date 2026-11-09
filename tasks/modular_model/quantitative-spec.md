# Quantitative measurements and digestion preparation

This extends the full goal's observation paths without claiming identifiable MS
bias, calibrated quantities or direct cleavage supervision from peptide lists.

## Observation contract

- Add namespaced endpoints `bulk_quantity:<domain>`, `elution_quantity:<domain>`,
  `ms_quantity:<domain>`, `digest_quantity:<domain>` and
  `cellular_quantity:<domain>`. The first two are combined observed quantities;
  the latter three require independent measured component evidence. Cellular
  quantity means pMHC availability, not an arbitrary biochemical cleavage assay.
  Direct cleavage/transport-specific admission remains a separate unfinished
  requirement; do not manufacture it from these endpoints.
- A declared `QuantitativeDomain` records measured quantity, canonical unit,
  identity/log2/log10 target transform, normalization and its source/reference.
  Each row carries this contract and must exactly match the model's declared
  domain. Units and normalization are checked on training and prediction.
- Preserve raw values, units and qualifiers, with explicit linear/log2/log10
  value encoding. Convert logs algebraically without needless exponentiation.
  Reject undefined/nonfinite conversions, missing quantities and invented zeros.
  Bounds use their real numerical detection/quantification limit. Missing rows
  remain missing; no quantitative-to-binary conversion.
- Domains distinguish measurement meaning/scale, not learned dataset features.
  Joint fitting can retain several incompatible scales with separate readouts;
  it must not claim their values are interchangeable. Direct component anchors
  need source-attested isolated/calibrated measurements, not database membership.

## Small model extension

- Reuse the shared encoder and existing digest, cellular and MS feature blocks.
  For each domain allocate only reachable bias-free width-128 scalar readouts.
  The same domain's MS readout is shared between bulk, elution and MS anchors;
  the digest and cellular readouts are shared with their admitted anchors.
- Sum required component scores plus one endpoint offset. Keep quantitative
  outputs in the declared transformed measurement space. Standardize residuals
  with training-only scales inside the loss, preserving a common component scale
  across endpoints with different target distributions. Legacy assay outputs
  retain their existing standardized internal representation.
- A quantitative MHC bag has one outcome and one loss. Deduplicate candidate
  molecules and use a statistical mean in linear space (implemented in the
  declared log base where relevant). This is not a physical occupancy model.
- Binary routes keep their existing readouts and behavior. Quantitative-only
  subsets must not allocate unused binary readout parameters. Capabilities and
  serving guards distinguish per-domain directly anchored versus indirect scores.
- Add explicitly declared numeric preparation fields/units plus missingness to
  digest features only (e.g. a documented prospective digest condition). They do
  not enter the shared detector, purified binding or APC presentation. Training
  preflight requires varied observed values for allocated preparation channels.

## Integration and verification

- Centralize endpoint metadata so source admission, deduplication, scalers,
  forward/loss, evaluation, checkpoint loading, subset initialization, support
  guards and public inference agree on routes and target units.
- Preserve endpoint/domain names in coverage and held-out records. Never apply
  the 500 nM binding threshold to a non-affinity quantity sharing a unit string.
- Test log conversions and censor bounds, incompatible normalization rejection,
  shared MS weights and context isolation, candidate-bag loss multiplicity,
  subset allocation, direct/indirect support, preparation feature gradients,
  streaming admission/evaluation and fresh checkpoint inference.
- Run focused regression and pinned lint/docs, then required broader validation.
  Software fixtures do not satisfy the real-data baseline/anchor/transfer gates.
