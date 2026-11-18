# Repository boundaries and retention

The supported implementation is the installed `presto` package. Start with
[Current Workflow](current_workflow.md) for the model, trainer, admission adapters
and commands. `legacy/` and archived launchers support historical reproduction;
new work uses the shared core.

## What belongs in git

Keep source, tests, package resources, current documentation, experiment decisions,
source/data hashes, launch contracts and compact result tables. Preserve existing
reference bundles until a deliberate migration supplies equivalent retrieval and
integrity checks. Model tensors and multi-GB observation exports belong in the
artifact store, with exact locations and hashes recorded in the experiment.

The three recent APC/presentation families have readable decisions and compact
results plus a checksummed `reproduce/history.tar.gz`. This preserves every
previously tracked file, including frozen runtime snapshots and intermediate
receipts, without presenting thousands of copies as active source modules.
`reproduce/history.json` records original paths, hashes and the source commit.
Restore an exact historical tree from the repository root with:

```bash
python scripts/restore_experiment.py \
  experiments/2026-11-17_1155_codex_shared-presentation-expansion/reproduce/history.json
```

The restore command verifies the archive and members before writing, and refuses
to replace different existing files. Raw tensors and untracked prediction dumps
remain at the original experiment's documented local/Modal locations. Compression
does not turn an incomplete historical record into a complete one.

Some regression tests deliberately load historical preparation/evaluation fixtures.
Those fixtures remain as plain files; they are not imported by the supported
runtime. Do not edit frozen snapshots to make them agree with new behavior.

## Local artifacts require an inventory

Ignored does not mean disposable. `artifacts/` can contain isolated source branches
and recovery copies; `modal_runs/` can contain the only local checkpoints or
prediction dumps. Before deletion, identify the active consumer and verify another
copy's hash. Keep unrelated edits and unmerged branches during cleanup.

Temporary build directories and completed pytest temporary directories are
regenerable. Retain the verification logs and receipts before removing them. On a
space-constrained machine, run tests with
`./test.sh -o tmp_path_retention_policy=none` so per-test checkpoints are discarded
as their tests finish. Do not remove another process's active temporary directory.
