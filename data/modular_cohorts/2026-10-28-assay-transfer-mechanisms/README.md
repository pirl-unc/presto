# Auxiliary transfer development cohort

Only the earlier six-output TRAIN population is repartitioned. The immutable input
hashes, component assignments, output files and negative-control label permutation
are recorded in `preparation.json` and the experiment's `reproduce/data-pins.json`.
Measured observations remain unchanged except split/group metadata. Files ending
`shuffled.jsonl` contain explicitly synthetic auxiliary training labels; they are
negative controls and must not be admitted as measured biological outcomes.

See `experiments/2026-10-28_0250_codex_assay-transfer-mechanisms/README.md` for the
fixed experimental contract. Old validation/test examples are excluded. This is
development data, not an independent external biological qualification population.
