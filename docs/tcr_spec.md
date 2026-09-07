# Receptor-specific pathway: future work

The authoritative desired design is [model_io_contract.md](model_io_contract.md).
This page records the boundary of a future extension, not an implemented API.

## What exists now

Canonical Presto has no TCR-sequence input, `tcr_vec`, `match_logit` or
receptor-specific affinity output. The unused standalone TCR implementation was
removed; its history can be inspected with `git log -- models/tcr.py`.
No checkpoint key-dropping migration should revive it implicitly.

VDJdb/McPAS and other receptor records can contribute **pMHC-only evidence**
and evidence-method supervision. That is an active objective, but does not
predict whether a specified receptor recognizes the pMHC.

`Predictor.predict_recognition` and `presto predict recognition` report
repertoire-level recognition/immunogenicity, not receptor-specific matching.
The current recognition path still lacks the complete presented-pMHC and
selection-system context required by the desired design.

## Intended extension

- Encode paired receptor sequences without species or assay metadata in the
  residue encoder. Preserve absent chains and pairing/source provenance.
- Keep individual receptor species independent of the species of the system
  whose repertoire is being modeled.
- Keep presenting APC MHC separate from repertoire-selection MHC.
- Explicitly distinguish population/repertoire recognition from a specified
  TCR–pMHC match; supplying a receptor must not silently change the meaning of
  the same output.
- Match against the presented pMHC representation. Repertoire-level recognition
  is not restricted to peptide plus foreignness.
- Add receptor-specific binding/functional outputs only with declared source
  evidence, negatives, loss semantics and independent held-out support.

Attention architecture, contrastive loss, missing-chain handling and multi-TCR
aggregation are implementation choices to settle in a new scoped specification.
An old cosine-similarity prototype is not a commitment to a validated matching
model. No multi-TCR bag or receptor-specific canonical training is claimed here.

See [issue #46](https://github.com/pirl-unc/presto/issues/46) for the unified
input/output program and acceptance gates.
