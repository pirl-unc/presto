# Presto storage audit and bounded cleanup

The user explicitly asks whether development history creates waste and authorizes
cleaning old Presto logs/artifacts. Audit the actual footprint, then reclaim
disposable or duplicate storage without losing unique scientific evidence.
This is maintenance, not a predictive experiment. Existing training resource gates
remain unchanged. Work in the modular worktree so primary-checkout edits/staging
are preserved.

## Scope and classification

Inventory the four existing Presto worktrees and `/private/tmp/presto*` entries.
Record logical/allocated sizes, file age, large-file categories and exact duplicates.
Do not follow symlinks into unrelated projects. Do not scan or clean shared Python,
other projects' temporary directories, Git object storage or user backups.
Directory allocation totals can double-count APFS shared extents; report measured
filesystem capacity separately and do not attribute concurrent disk movement solely
to cleanup.

Preserve unique raw inputs, completed/failed experiment evidence, per-example
predictions, selected/resumable checkpoints, frozen source/configs/manifests and
canonical reports. An old timestamp alone does not justify deletion. Identify
files still open by running processes and exclude them from mutation. Preserve
modified/staged tracked files and tracked source generally.

## Allowed first pass

1. Remove only explicitly inventoried ignored Python/test/lint caches and stale
   generated site/build outputs whose inputs remain. Verify tracked/untracked
   status and age; exclude live/open files. No broad recursive cleanup command.
2. For large byte-identical immutable files, use APFS copy-on-write clones to share
   storage while retaining every path and independent inode. Hash entire files,
   require regular non-symlink single-link files on the same device, compare mode
   and extended attributes, and preserve destination timestamps. Recheck content,
   identity and metadata immediately before atomic replacement. No full-copy
   fallback, hard links or changed file contents. Freeze an explicit candidate
   inventory before mutation and retain per-file results.
3. Delete redundant downloaded/build artifacts only where a complete retained
   byte-identical copy is verified and references can remain valid. Otherwise use
   clone sharing. Do not remove whole worktrees until cleanliness, branch retention,
   running-process use and untracked records are verified.

The inventory also finds143range-download parts for two retained assembled MAPP
archives, plus two partial-download files. Remove only pieces independently matched
byte-for-byte at their original offsets in a complete gzip-verified archive. Keep
archive hashes, piece hashes, offsets and restore commands, so no unique bytes or
source provenance are lost. Preserve any nonmatching/in-use piece.

If these do not recover useful space, assess lossless archival of large closed
text logs with recorded original hashes and restoration commands. Make a concrete
inventory before considering deletion of any unique historical results.

## Verification and closure

Record before/after capacity and exact reclaimed candidates. Verify retained hashes,
metadata, independent clone inodes, worktree status/staging and relevant artifact
manifests. No numerical training is launched as part of cleanup. Write a concise
report with footprint categories, actions, recoverable storage and a simple future
retention recommendation. Do not mistake regeneration caches for canonical data or
claim that directory size alone is reclaimable physical space.

## Preserved first-pass interruption

The first pass removed1967ignored cache files and145verified redundant download
pieces, then shared three large duplicates. It stopped before replacing an old
primary-worktree CSV because macOS assigned the temporary clone a different
`com.apple.provenance` value. Content hashes agree; the original CSV is unchanged.
Do not change or remove provenance attributes to force sharing. Preserve the
failure receipt and original helper. Resume only unprocessed manifest entries;
if a new temporary clone has different attributes, record it, discard that newly
created redundant temporary, and leave the original untouched. Retain all other
content/metadata checks and verify completed operations rather than replaying them.
