# Full-chain TCR specificity and IC50 diagnostic — 2026-10-20

## Scope and evidence

User requests a receptor-specific extension, using the functionally validated
VDJdb subset, and investigation of almost constant IC50 predictions including
Hitlist freshness and upstream data requests. Work in an isolated checkout from
`ee4ad60`; preserve the original dirty checkout.

The specified paper is Messemaker et al., PMID 40463065 / PMC12132471,
DOI 10.1101/2025.04.28.651095. Its TCRvdb resource is distinct from VDJdb score
filtering. The author repository now identifies a 2026-10-01 release and
DOI 10.1016/j.immuni.2026.09.007; access is by request and redistribution is
restricted. No substitute dataset or invented author column names.

## Implementation contract

1. Add a separate `TCRSpecificityModel` around an existing Presto instance.
   Inputs are paired alpha/beta amino-acid sequences and the existing pMHC input
   tensors. Preserve canonical Presto outputs/checkpoints and repertoire
   recognition semantics.
2. Encode both full supplied chains with position-aware sequence encoders and
   combine their representations with `pmhc_interaction_vec`. Use separate
   chain encoders to preserve alpha/beta identity. Expose a pair-specific
   functional-response logit, not a receptor KD or a validated probability.
   An untrained wrapper is explicitly unqualified.
3. Freeze the pMHC backbone by default, including its eval/dropout state.
   Permit explicit joint fine-tuning. Preserve gradients through the receptor
   encoder/scorer and through Presto only when requested.
4. Validate paired, nonempty, canonical amino-acid inputs; reject overlength
   chains rather than truncating them. A sequence is not certified full length
   by its character count. Preserve observed versus reconstructed sequence
   provenance in data, never infer missing chains or use outcome/source IDs as
   model features.
5. Provide an explicit normalized local CSV input contract and strict loader
   with source IDs, pMHC, paired chains, assay scope, sequence provenance,
   functional outcome and measurement/quality status. Unassayed, ambiguous or
   failed-QC outcomes remain excluded/unlabeled. No random mismatches promoted
   to experimental negatives. Do not claim to parse the author CSV until the
   actual authorized file is inspected.
6. Provide loss and checkpoint round-trip support and an executable usage
   example using existing pMHC preparation. Keep training dataset assembly
   separate from canonical pMHC-only evidence loaders.
7. Evaluate future models with receptor-cluster and pMHC/study-aware splits;
   a few assayed pMHCs cannot establish generalization to arbitrary epitopes.
   Dataset acquisition/training on TCRvdb is gated on authorized file access.

## IC50 and Hitlist investigation

1. Compare the executed frozen manifest, current published exports, local
   package version and actual local index generation. These are separate
   freshness claims. Never rewrite the historical run or refresh its inputs.
2. Trace raw endpoint/unit/censor labels through collation, model head and
   loss. Inspect published training-only diagnostics and gradient/optimizer
   evidence. A finite gradient does not prove useful sequence dependence.
3. Register any new computation beyond unit verification in a timestamped
   experiment directory before running it. Start with read-only artifact
   analysis and a bounded CPU training-only diagnostic if source artifacts
   are available. Preserve source/target identity and all failure outcomes.
   No paid compute or new held-out model-selection sweep.
4. Identify actual missing upstream fields/evidence. Search existing Hitlist
   issues, then file specific nonduplicate requests with schema, provenance,
   acceptance checks and downstream use. Do not blame data freshness for
   collapse without evidence.

## Verification

Test chain sensitivity, pMHC sensitivity, padding, invalid/missing/overlength
inputs, batch shape, frozen/training modes, gradient flow, loss masking and
checkpoint equivalence. Test strict data admission and provenance preservation.
Run relevant existing tests, pinned Ruff, strict docs and full suite before
publishing code. Record measured outcomes and remaining access limitations.
