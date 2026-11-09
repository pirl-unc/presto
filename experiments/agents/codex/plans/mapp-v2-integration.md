# Integrate pinned MAPP evidence with the concurrent Hitlist 2.0 release

Prospective follow-up, October 23 2026. No implementation or new training has started. The repeatability experiment used Hitlist b94289d and immutable source/recipe manifests; PR761 subsequently merged to main at0dfa170. Draft758/760 must join the canonical interface before release.

## Contract

1. Read current main EvidenceStore, evidence bindings, schema7 reference successor and canonical evidence_recipe API. Diff both draft branches against main; preserve unrelated work and all original data/measurement identities.
2. Write an integration spec in Hitlist before changes. Rebase or prepare a clean successor worktree without rewriting pinned experimental receipts. Reconcile source-generation schema/version and catalog/bindings rather than keeping a second public query/materialization path.
3. Represent the MAPP three/two-peptide and complete-run admission rules through the canonical recipe interface with explicit source scope, observation pool, quantity/censor policy, contributors and connected-component partitioning. Preserve all dispositions, base run observations, mapping alternatives and no-observed-zero assumptions. If the generic API cannot express a rule, add a small shared extension and tests; do not bypass into a Presto workbook reader.
4. Independently compare every observation, contributor, normalized target, eligible set and split to the pinned outputs. Expected producer/provenance metadata changes are distinct from biological changes; any target/member change requires explanation and a new recipe version.
5. Exercise public API/CLI end-to-end on the actual shared generation. Run focused semantic tests, format/lint, normal full suite with intact memory guard, strict docs and final-head CI. Follow existing release workflow; do not modify the shared Python environment opportunistically.
6. Publish reconciliation evidence in the canonical experiments registry if a new materialization/audit is performed. Keep existing models and source-qualified calibration (#762), precursor (#757) and processing (#759) tasks separate.

## Review gates

No claims of using latest Hitlist until the new generation actually passes source/recipe replay. No additional fitting against the inspected MAPP test. Closure requires canonical interface accessibility plus byte/semantic conservation, not only passing software tests. Goal remains machinery-conditioned peptide availability modulo precursor abundance, not observed-set composition alone.
