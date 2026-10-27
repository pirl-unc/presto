# Shared transfer improvements

Status: preparing; no new results yet. Codex / GPT-6.

Prospective specification: [plan](../agents/codex/plans/2026-10-27_transfer-improvements.md).

Twelve fixed fits compare exposure, class balancing and shared residue cross-attention.
Original inspected development IC50 cohort; no biological qualification claim.
Requested H100!, at most two concurrent workers. Raw outputs: artifacts/2026-10-27_2100_codex_transfer-improvements/runs.

Next step: implementation verification and frozen launch.
Open questions: whether either hypothesis improves validation and cross-class transfer.

Prelaunch verification:136 software tests pass, including both-class interaction gradients,
exact balanced-sampling resume, common endpoint contracts and v2/v3/v4 inference replay.
Changed production code and explicitly included experiment scripts pass pinned Ruff0.16.0.
Prepared source/data upload132,308,870bytes; two frozen runtimes share the same trainer.
