# Broad Frontier Bakeoff (2026-03-12)

## Spec

- Goal: rerun the current broad-contract frontier candidates for 5 epochs on Modal under one explicit experiment family so the broad IEDB frontier is interpretable beyond 3-epoch snapshots.
- Fixed contract:
  - 7 class-I alleles: HLA-A*02:01, HLA-A*24:02, HLA-A*03:01, HLA-A*11:01, HLA-A*01:01, HLA-B*07:02, HLA-B*44:02
  - measurement profile: `numeric_no_qualitative`
  - included assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
  - qualifiers: `all` with censor-aware loss
  - no synthetics
  - no ranking / contrastive
  - 5 epochs
- Conditions to include: every broad-contract 3-epoch condition previously ranked from the experiment log:
  - Directness round-2 canonical/groove conditions that were recorded in the log ranking (`DP00`, `DP01`, `DP05`, `DG1`)
  - Positional composition `P00..P07` and `G00..G07`
  - Assay-head `A00..A07`
- Preserve family-specific architecture defaults where required:
  - canonical Presto conditions use warm start
  - groove-transformer controls keep the groove baseline path and batch size
- Deliverables:
  - experiment dir under `experiments/`
  - manifest and variants table
  - later: `options_vs_perf.*` and canonical log entry

## Execution

- [ ] Add spec to tasks/todo.md
- [ ] Add detailed plan to experiments/agents/codex/plans/
- [ ] Implement unified launcher for 28 frontier conditions
- [ ] Verify launcher syntax locally
- [ ] Launch detached Modal sweep and record app ids incrementally
