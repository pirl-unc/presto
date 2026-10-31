# Independent audit control-label coverage

The first audit passed its source-cell, quantity and search-row comparisons but
stopped in the acquisition-name check at `AlphaSyn_0min_R1`: the independent
verifier only listed mixed-enzyme/substrate and organ labels. The source plan and
adapter already preserve substrate-only and enzyme-only controls. Add the exact
`(?:AlphaSyn|20S)_\d+min_R[123]` label grammar, checking its full deposited RAW
suffix. Do not skip those rows, infer specimens or change canonical source bytes.

The original audit code, frozen source, log and failed resource receipt remain.
The amended `audit-controls` stage has its own start, log, snapshot and resource
receipt. All prior invariants and limits remain. This amendment changes verifier
coverage, not source interpretation, cohort membership or a biological result.
