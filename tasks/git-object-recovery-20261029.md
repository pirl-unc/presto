# Git object-cache recovery, 2026-10-29

The shared-training checkout borrowed objects from a temporary checkout that
was removed externally. Preserve working files, index edits and all refs.

- [x] Record the missing alternate and index-object inventory.
- [x] Restore exact hash-matching blobs from existing working files.
- [x] Fetch the remaining 425 missing current-index blobs by object ID.
- [x] Replace the vanished alternate with the durable main repository object store.
- [x] Commit the independently verified assay audit (cabf653d).
- [ ] Restore remaining historical commit/tree/blob objects; verify full connectivity.
- [x] Verify published audit branch and preserve recovery evidence.

The initial all-history refetch stalled and was stopped. Targeted object fetch
restored the active index. A connectivity check still identifies one historical
commit, two trees and three blobs. Use an isolated filtered object fetch for
commit ancestry, then restore remaining named objects; never rewrite history,
clear refs, reset the index or discard local changes to hide missing data.
Recovery logs/receipts are currently under /private/tmp/presto-*-recovery*.

Audit cabf653d was pushed successfully. Current index and branch content are restored. Commit ancestry was recovered through an isolated tree-filtered fetch; a bounded exact-object recovery controller is now resolving remaining historical trees/blobs. Durable logs, controller and status: /Users/iskander/code/presto/artifacts/git-object-recovery-20261029/.
