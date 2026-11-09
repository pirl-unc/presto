# Presto storage cleanup — 2026-10-25

The avoidable growth was mostly repeated inputs and experiment artifacts. The
read-only inventory found 7.86 GiB of
byte-identical duplicate copies (269 groups). Plain `.log` files total only about
110 MiB. JSONL includes observations and prediction dumps as well as logs, so it
was not treated as disposable text.

Examples: the same 479,550,307-byte source table appeared three times in recipes;
a 5,725,138-byte training split appeared 64 times across conditions/worktrees.
The main checkout accounted for about 18.2 GiB including Git, the current worktree
17.8 GiB, and two older worktrees 2.5 GiB. APFS extent sharing can make directory
accounting double-count physical storage; these are not independent totals.

| Action | Verified result |
|---|---:|
| Removed redundant range/partial-download files | 145 files; 601.4 MiB |
| Removed ignored regenerable caches | 1967 files; 38.7 MiB allocated |
| Shared storage using independent APFS clones | 529 files; 6.60 GiB logical |
| Skipped files whose clone changed macOS metadata | 24 originals preserved |
| Unique scientific files deleted | 0 |

Observed available capacity increased from **1.20 GiB**
at inventory to **5.52 GiB** at final verification
(a **4.32 GiB** net increase).
Other jobs were using the same volume: this is an observed capacity change, not
an exact accounting of physical blocks exclusively reclaimed by this process.
Space had fallen to 0.65 GiB before the first removal.
The existing local-experiment startup reserve is now available; no new training
was started by cleanup.

## Preserved evidence and verification

All scientific paths and unique source data, checkpoints, logs, predictions,
reports and Git history remain. Every replaced file was whole-file SHA256 verified
before and after, with independent inodes and original mode/owner/group/flags,
extended attributes and modification time. All 24 skipped originals retain their
original inode, contents and metadata. Git HEAD, tracked status, working diff and
staged diff match the pre-cleanup snapshot in all four worktrees, including the
user's dirty primary checkout. The pending experiment's four pinned input hashes
also match. Two tiny Ruff cache metadata files regenerated normally afterward.

Both retained assembled archives passed gzip CRC validation. All 145 removed pieces
were independently matched to exact byte ranges in those archives before deletion,
and independently reconstructed and rehashed afterward without recreating files.
A first-pass interruption stopped before replacing an original CSV when macOS
changed the temporary clone's provenance attribute. The recovery skips such
files; no provenance attributes were rewritten. No hard links or Git garbage
collection were used. Shared Python and other projects were not cleaned.

[Independent verification](verification.json), [initial inventory](inventory.json),
[exact cleanup plan](cleanup-plan.json), [fragment mapping](fragment-plan.json),
[first-pass journal](cleanup-journal.jsonl), [recovery journal](recovery-journal.jsonl),
[interruption](first-pass-interruption.json), [recovery result](recovery-result.json).
Executed maintenance code is preserved in `executed-source/`; it should not be
blindly rerun against already processed paths.

## Restoring the redundant pieces

The assembled archives are the retained copies. From the worktree root, inspect:

```sh
/Users/iskander/code/shared-virtual-env/bin/python -B tasks/storage_cleanup_2026-10-25/restore_fragments.py
```

Add `--apply` only to recreate the removed pieces. The helper verifies archive and
piece hashes, refuses overwrites and requires enough capacity plus 1 GiB reserve.
Restoration is not required for model training or source replay.

## Preventing recurrence

- Keep immutable inputs once per content hash and reference them from each run.
  When a standalone evidence package needs the same files, use copy-on-write copies
  where supported instead of physically duplicating complete source tables.
- Remove range-download fragments after assembled archive validation; retain the
  download manifest, hashes and complete archive.
- Keep canonical reports, reproducibility configs, held-out prediction dumps and
  selected/resumable checkpoints. Reserve full step-by-step checkpoint histories
  for experiments whose registered diagnostic actually needs them.
- Periodically remove stale ignored caches. Check clean branches, untracked evidence
  and live use before retiring worktrees; no whole worktree was removed here.

## Subsequent workspace recovery

The preservation checks above describe the completed cleanup at its verification
time. Later, the active worktree's `.git` pointer and 1,396 tracked files were found
missing from temporary storage; the cause is unverified. All recorded missing
tracked files were restored from verified Git content, and the active worktree
was moved into the durable project directory. Primary-checkout fingerprints,
282 recent experiment artifacts and 46 source assets passed independent checks.
See the [recovery report](../workspace_recovery_2026-10-25/README.md). This later
recovery is separate from the cleanup and its measured capacity change.
