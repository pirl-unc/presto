# Equal-study validation loss for staged presentation

Problem: the frozen presentation validation has 634/692/47,204 rows across three
independent study components. Pooled loss mostly selects for the largest study,
although the prospective benchmark reports equal-study performance. Preserve all
rows and the disjoint test; add an opt-in canonical selection reduction.

Implementation: `selection_reduction="study_mean"` with optional existing
`selection_endpoint`. Filter identity and frozen outcomes exactly as the existing
selection code does. For each eligible endpoint, compute arithmetic mean of
per-row losses within the canonical `group_id` study component, then give every
component equal weight; if multiple endpoints are selected, give endpoints equal
weight. Reject missing/empty study group IDs and nonfinite per-row losses. Existing
observation_mean/endpoint_mean behavior and default resolved contracts remain
identical. Affinity strict-Spearman selectors cannot be combined with this mode.

Preflight must accept primary endpoint+study_mean while retaining old conflicts.
The training selection call passes complete validation records only for the new
mode. Resume contracts already include training options/source hashes, so an
existing run cannot silently change this criterion. No backbone or source labels
change. Expose the criterion in checkpoint metadata/resolved config and document.

Verify hand-calculated unequal-study and multi-endpoint examples, frozen/identity
filtering, malformed-record rejection, complete canonical two-update selected-
checkpoint/reload and exact interrupted resume under the new criterion. Existing
default golden trajectory tests must remain unchanged; run focused controls,
continuation/resume tests, lint, then standard CI. Do not modify running affinity
workers or their frozen source. Freeze a new reviewed runtime for the later
presentation fit and rerun the software fixture under it before GPU allocation.

Verification: 102 focused selection/transfer/resume tests passed; complete suite
3023 passed,4 expected skips (300.78s). Repository lint/format passes523 files;
strict MkDocs build passes. Exact interrupted-resume and default trajectories
remain covered. New source will be frozen separately from active7a23/2dabb jobs.
