# Finite partial observations and multiset distribution losses

Prospectively registered October 22, 2026; Codex / GPT-6. No fit outcomes yet.
The frozen [plan](../agents/codex/plans/partial-observation-multiset.md) specifies
60 fits: positive/blind observation weights × 32/256 counts × exact CE, exact L2,
corrected multiset L2 (16 or 64 draws), and naive histogram MSE × three seeds.
The naive method intentionally optimizes a different expected objective.

Data/model are the earlier generic integer-symbol experiment: 144 parent strings,
96/24/24 parent-disjoint splits, four contexts, 384/96/96 rows, 70 spans per row.
There are no biological sequences or assay measurements. Known observation weights
and finite empirical counts replace full-oracle training. The fitting interface
contains only features, starts, weights and counts. True distributions are used
by the simulator and post-fit diagnostics, never checkpoint selection.

All fits use Adam .003, no weight decay/pretraining, batch16, 1,440 updates, one
CPU thread, deterministic float32. Select minimum validation observed-count CE
every 120 updates. Count budgets are nested from fixed draws. Per-example train,
validation and test probability dumps, complete count identities and weights,
checkpoints, all training histories and provenance will be retained.

Run `bash experiments/2026-10-22_1317_codex_partial-observation-multiset/reproduce/launch.sh`.
The launcher refuses to overwrite a run. `reproduce/launch.json` pins all settings
and source/plan hashes; `reproduce/source/` freezes fit sources. Execution records
the actual commit, working-tree state, environment, runtime and memory. No Modal
or GPU. Molecular units, affinity cutoffs and binding-classification metrics do
not apply. Conditional counts do not identify absolute concentration.

Eight pre-fit tests pass, including exhaustive expected-cost and policy-gradient
checks for the corrected and naive objectives, finite gradients with zero
observability, invisible-mass nonidentifiability, nested observations and exclusion
of oracle labels from the fitting interface. See `launch_logs/tests.log`.
