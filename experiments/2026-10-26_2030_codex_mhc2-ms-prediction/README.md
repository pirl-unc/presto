# MHC-II MS permutation background — infeasible control contract

Status: terminal preparation failure, preserved. Codex / GPT-6,2026-10-26.
[Original plan](../agents/codex/plans/2026-10-26_mhc2-ms-prediction.md).

Source and partition preparation succeeded, but control generation stopped after
1,000 attempts for a length15 homopolymer. Its exact composition permits only
one sequence, so a distinct permutation is mathematically impossible. No positive
was silently dropped; no model was scored/fitted, and no holdout prediction exists.
The failed source hash and diagnostic are in `results/failure-diagnostic.json`.

DP retained18,906/3,853/4,062 peptides in train/validation/test, comprising
5,986/1,303/1,317 connected components. External DQ8 retained3,453 peptides/
1,189 components. These are input counts, not predictive outcomes. Source
qualification, exact paired molecules and initializer exclusions follow the plan.
No synthetic outcomes were successfully materialized; no loss or training applies.

The run took8.395seconds, peakRSS863,846,400bytes, localCPU/one numerical thread;
no Modal/GPU/install. It remained within the original resource limits. Source and
script hashes, git commit/dirt, invocation and original failure are preserved in
`reproduce/launch.json`, `launch.sh`, `source/` and compressed exact logs.
Six boundary tests passed before launch. The failure was anticipated by the
fail-closed homopolymer test and handled by stopping, as specified.

Next: [separate length-background family](../2026-10-26_2036_codex_mhc2-ms-length-background/README.md),
registered before predictions, changes the background distribution and explicitly
retains all original positive membership/partitions. The broader goal is active.
