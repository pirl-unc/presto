# Alternate binding-selector descriptor repair

Agent/model: Codex / GPT-6. Detailed implementation and experiment contract:
[`tasks/binding_selector_spec.md`](../../../../tasks/binding_selector_spec.md).

Compare actual panel and probe-bootstrap selectors before/after retaining
observed assay type/method and culture fields. Keep selection, units, qualifiers,
ordering, publication lineage and non-descriptor payloads identical. The fixed
allele-panel audit measures that declared subset, not all source observations.
No training, predictive evaluation, MHC filtering or source curation change.

Use the registered timestamped experiment, pinned merged input/environment,
explicit selector parameters and clean commit receipts. Preserve pre-rebase
audit commits when moving the eventual PR onto merged main. Close complete
field/column counts and payload reconciliation before claiming #62 resolved.
