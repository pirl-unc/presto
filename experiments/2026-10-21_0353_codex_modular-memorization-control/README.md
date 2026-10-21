# Real IC50 memorization with supplied MHC metadata

Status: prepared and checked; optimization has not started.
Agent/model: Codex / GPT-6. Plan:
[modular-assay-only-memorization.md](../agents/codex/plans/modular-assay-only-memorization.md).

One assay-only condition uses 16 class-I and 16 explicitly paired class-II
observations from pinned Hitlist v4. No auxiliary pretraining/replay, no synthetic
data, no held-out evaluation. All outcomes are raw exact log10(nM). This tests
optimization/capacity, not generalization, useful auxiliary transfer or the
completion of the holistic model. The full prospective contract and fixed
pass/fail/resource rules are in the linked plan; inputs/launcher/results will be
frozen in this directory and the shared cohort area before optimization.

No validation/test split is used intentionally: all 32 observations are fit
targets. Historical held-out peptides are excluded during preparation. Future
independent evaluation must account for these 32 newly exposed inputs.

Preparation scanned 814,268 raw rows and found 26,619 eligible candidates.
There were 841 contradictory exact-input groups and zero conflicting original
assay IDs among eligible candidates. Excluding those groups and collapsing
consistent repeated inputs left 24,146 possible diagnostic inputs. The fixed hash
selected 16 per class. All 32 source values replay exactly as unclipped log10(nM).
Target ranges are 0–5.89345 for class I and 0.93802–4.48144 for class II.

The selected inputs are in
[`data/modular_cohorts/2026-10-21-memorization-control`](../../data/modular_cohorts/2026-10-21-memorization-control).
The frozen manifest SHA-256 is
`74f6ff7e208a239bb5d83c295bcb950e07effd999f571d7a615ff1410a2f94ee`.
The curation/prelaunch receipts and selected catalog records are in `results/`.
Large source/candidate disposition files are under `artifacts/` and pinned by
`results/disposition-artifacts.json`. Original preparation code is preserved as
`reproduce/source/preparation-launch.py`; the later launcher only changes the
raw-artifact destinations before fitting, not selection or target policy.

Prelaunch gate arithmetic was checked with exact predictions and a deliberately
failed class-II spread condition; both behaved as intended. This is a software
check, not a measured model result. The application source `51b5f61` passes the
full regression (2,454 passed, four skipped), strict docs and pinned lint.
