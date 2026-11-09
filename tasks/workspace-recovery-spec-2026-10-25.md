# Recover the active worktree without losing edits or source paths

At04:10UTC publication verification discovers that the active temporary worktree's
`.git` pointer and1,396tracked files are missing. The common Git registration,
index, branch and all required blobs remain; HEAD is still8454ef9 and the index
has no staged changes. Surviving work, including0344source evidence and unrelated
task edits, remains present. The cause is unverified; do not attribute deletion
to any person or process from timestamps alone.

The read-only manifest records every missing path, mode, blob ID and size:
482,416,345bytes total. It also records primary-checkout HEAD/status/working and
staged diff fingerprints. Do not publish these deletions, reset the worktree,
overwrite surviving files, delete Git metadata or touch the user's tracked work.

## Recovery

1. Revalidate registration, HEAD, empty index and source artifact hashes. Preserve
   current source/data paths and the existing missing-file inventory.
2. Restore only the missing `.git` pointer from the surviving registered gitdir.
   Move the worktree with Git to the durable ignored directory
   `/Users/iskander/code/presto/.worktrees/presto-modular-model`. Add only that
   dedicated directory to the repository's local `.git/info/exclude`, preserving
   existing exclusions. This changes no tracked primary-checkout content.
3. Create `/private/tmp/presto-modular-model` as a compatibility symlink to the
   durable location so frozen absolute source paths and launcher references keep
   resolving. No scientific file is copied merely to relocate the worktree.
4. Restore the exact missing tracked blobs from the surviving index/HEAD, preserving
   executable modes. The full482MB fits the observed capacity; require at least
   its remaining bytes plus1GiBreserve and32MiBmargin before starting. Recheck the
   reserve during restoration. Prefer a verified identical primary-checkout APFS
   clone when available, with independent inode, otherwise stream the Git blob.
   Never overwrite an existing file or follow an unexpected parent symlink.
   Journal each restoration; a resource failure leaves completed work intact and
   does not relax the guard.
5. Independently verify all restored blobs against Git IDs, all51source artifact
   entries and their original-data asset hashes, compatible absolute paths, and
   unchanged primary HEAD/status/diffs. Confirm no tracked deletions remain and
   unrelated task changes are preserved. Run applicable lint/static checks before
   selectively committing only owned source-report and recovery changes.

Do not store a second copy of bulk raw data, reset user work or change shared
Python. Recovery is complete only after these checks; the full scientific goal
and canonical-source implementation remain separate unfinished work.
