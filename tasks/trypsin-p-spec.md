# Separate the Trypsin/P search rule — Presto #82

## Evidence and scientific boundary

The frozen PXD013455 search contract declares MaxQuant Trypsin/P (public source
manifest 441f3c66c6e6bc6a90c7cb8c54d0131676c3cb4e7a04ddbd9346061ad8f9d84b).
Cox Lab's primary Andromeda documentation explicitly distinguishes Trypsin/P,
which permits cleavage after K/R even before proline, from the Trypsin rule that
excludes proline at P1-prime:
https://cox-labs.github.io/coxdocs/andromeda_enzymes.html .
This is a search-specificity contract, not proof every chemical cleavage occurs.
The current consumer silently maps both names to a proline-blocked rule.

## Vocabulary and controls

Append `trypsin_p` to the machinery vocabulary; preserve all existing indices.
Add the corresponding declared digest category, route explicit Trypsin/P strings
to it, and keep ordinary Trypsin distinct. The contradictory historical string
`Trypsin/P (cleaves K/R except before P)` must fail or remain explicitly identified
as unqualified legacy metadata; it must not silently establish either contract.
The qualified scoped adapter must propagate the correct category while retaining
the exact source protocol string. Rules: P1 K/R, no P1-prime proline exclusion for
Trypsin/P. Existing Trypsin behavior and proteasome-mixture components stay fixed.
Rule-derived controls retain their synthetic role; no new observed labels appear.

## Checkpoint compatibility

Version the excision machinery vocabulary in the model constructor/checkpoint
config. New models use the appended vocabulary. Existing checkpoints without a
version are inferred only from an exactly recognized head shape and constructed
with the original seven-entry vocabulary; all original parameters and predictions
remain unchanged. Do not pad learned weights or silently copy a Trypsin fit into
a newly claimed Trypsin/P fit. Legacy models reject the new category explicitly,
including tensor-index callers, rather than clamping it to an unrelated row.
New checkpoints record the version and load normally; unknown/contradictory
versions or head shapes fail clearly. Audit canonical loading and trainer resume
paths so compatibility failures are explicit before training.

## Verification and publication

Test K-P/R-P boundaries, ordinary Trypsin exclusion, digest/machinery encoding,
scoped record-to-model routing and synthetic-control consistency. Demonstrate
unchanged old-vocabulary forward outputs after loading an old-format checkpoint,
new-checkpoint round trips, rejection of a new category by a legacy model and
unknown/mismatched vocabulary versions. Run focused and complete suites plus
formatting/lint/docs. Version 0.1.6; separate PR, review, merge and verified public
GitHub distribution. Update the scoped projection/policy receipt when the consumer
code changes and repeat the actual canonical preflight with the new identity.
The existing admission artifact remains evidence for its exact historical code;
no retroactive relabeling or readiness claim. Canonical training integration and
held-out fitting remain #70/#48/#53 after this semantic correction.
