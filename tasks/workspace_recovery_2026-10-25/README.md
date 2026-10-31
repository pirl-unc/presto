# Active worktree recovery — 2026-10-25

The restoration completed and passed independent verification. The active worktree
now resides at `/Users/iskander/code/presto/.worktrees/presto-modular-model`.
`/private/tmp/presto-modular-model` remains a compatibility symlink, so recorded
absolute artifact paths still resolve. Git performed the relocation without
copying the scientific data.

At approximately 04:10 UTC, the temporary worktree's `.git` pointer and 1,396
tracked files were found missing. The registered Git index, branch and required
objects remained intact. The cause is unverified. This discovery occurred after
the separately documented storage cleanup and its successful preservation checks;
do not attribute it to a particular process or person from timestamps alone.

Only the missing tracked paths were restored: 482,416,345 logical bytes, through
969 verified independent APFS clones and 427 Git blob writes. Existing files were
not overwritten. Recovery retained the 1 GiB free-space reserve plus startup
margin. It did not reset the index, rewrite history or modify shared Python.

The [independent verification](verification.json) checked:

- All 1,396 restored files against their recorded Git blob IDs, sizes, executable
  modes and independent inodes; no tracked deletions remain in the active worktree.
- Unchanged primary-checkout HEAD, tracked status, working diff and staged diff.
- Unchanged surviving active-worktree tracked status, apart from the restored
  deletions. This status check is not a historical byte-level snapshot of every
  untracked file or a claim about other temporary worktrees.
- All 51 artifacts in the current source audit and all 231 artifacts in the
  completed temporal comparison against their manifests.
- All 46 current source assets at their original absolute paths against hashes
  and sizes, including the preserved failed retrievals.
- The durable worktree registration and old-path compatibility symlink.

The exact executed restoration and verification scripts are retained in
`executed-source/`. The working helpers pass Ruff 0.16.0 lint and format checks;
[AST equality](format-verification.json) confirms formatting did not change their
behavior. The restoration is deliberately non-repeatable on existing files. Do
not rerun it or delete its journal. The verifier also refuses to overwrite its
completed result.

Local pre-recovery primary-checkout patches and status snapshots contain unrelated
user work and must remain unpublished. They support fingerprint comparisons;
this report does not include their contents. No model fitting or new artifact
cleanup ran as part of this verification. Later changes in shared-volume free
space must not be attributed to the earlier cleanup's savings.
