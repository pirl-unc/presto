# Why do six assay outputs help MHC-I more than MHC-II?

Status: preparing the fixed twelve-fit comparison; no new predictive results yet.
Agent/model: Codex / GPT-6. Date: 2026-10-28 UTC.

Detailed pre-implementation contract:
[plan](../agents/codex/plans/2026-10-28_assay-transfer-mechanisms.md).
Extends [the closed scale family](../2026-10-27_2255_codex_shared-multi-assay-scale/README.md).

Inventory the earlier data and separate capacity, class exposure, IC50 exposure
and auxiliary label information through the common trainer. A new component split
uses earlier TRAIN rows only; old heldouts and fitted weights are excluded.
This remains a development experiment, not external biological qualification.

Freeze source/data/manifests/invocation under `reproduce/`; keep live validation,
per-example heldout dumps, independent metrics, plots, replay and support evidence
under `results/`. Large inputs/checkpoints live in repo-level `data/`/`artifacts/`.

Status: preparation. Next step: inventory, source split and sampler checks.
Open question: capacity, reduced IC50 exposure, class imbalance or useful auxiliary
information beyond groove-specific target distributions?
