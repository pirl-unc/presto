# Preserve task data while recovering local training reserve

Current free space fell below the existing1GiBlaunch reserve after publishing
the completed experiments. Do not relax launch/full-suite gates, remove source
data/checkpoints, modify shared Python, or garbage-collect shared Git history.

The concrete reviewed candidate inventory is
`tasks/modular_model/intact-checkout-cow-plan.json`:195identical tracked files
(about269MBlogical) in four clean, task-created Hitlist worktrees. Each matches
the clean release checkout at the same relative path by full SHA256. No .git,
untracked, small, symlink, differing-content or user-primary-checkout file qualifies.

Use APFS clonefile directly, with no full-copy fallback, to create a temporary
independent inode sharing storage with the matching release file. Before atomic
replacement, verify exact source/target hashes and that target inode/metadata have
not changed; require ordinary single-link files, identical modes and xattrs.
Restore destination timestamps. Verify final content hashes, independent inodes,
unchanged Git state and all frozen experiment/runtime manifests. On discrepancy,
stop and preserve evidence; never unlink an original before its verified clone is
ready. Only the temporary destination is replaced. Record per-file evidence and
before/after free space, which may also reflect unrelated shared-volume activity.

This is storage maintenance, not a new biological experiment. Resume local runs
only if observed free space again exceeds the unchanged floor plus startup margin.
Full Hitlist suite still requires its separate2GiBprerequisite.

Initial helper stops before any clone/replacement because this macOS interpreter
does not expose os.listxattr. Preserve its source at
`/private/tmp/presto-verify-checkout-cow-initial.py`; compare complete extended
attributes with the native read-only `/usr/bin/xattr -lx` command instead.
Content/metadata/inode checks and all guards remain unchanged.

First pass verifies all195independent clones and unchanged Git/content state;
observed free space increases241,741,824bytes but remains982,929,408bytes, below
the unchanged reserve. Extend the same preflight and verified-clone procedure to
three clean task-created earlier checkouts: hitlist-lymph-20261024,
hitlist-lymph-recipe-20261024, and hitlist-mapp2025-20261024. The separate follow-up
plan/result preserves the first pass and its inventory. No extra source or
experiment data is removed. Shared-volume movement prevents attributing every
byte of observed change solely to this operation.
