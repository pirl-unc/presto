# Restore headroom without changing MAPP evidence

Problem: the shared APFS data volume has about1.1GiB free. Required tests hit the
unchanged1GiBprovenance reserve. Stop large materialization and suite retries until
headroom is verified; do not delete user work or lower a guard.

Scope: only duplicated, task-created MAPP package snapshots. Replace an existing
source-table/source-asset snapshot with an APFS copy-on-write clone of an already
verified identical source. This preserves independent file semantics and all bytes;
never use hardlinks or overwrite original source data. No biological data, target,
model, manifest, schema or experiment policy changes.

Procedure:
- [x] Enumerate approved pairs from bound manifests and verify SHA256/byte equality.
- [x] Test clonefile in owned scratch space, including mutation isolation.
- [x] Clone to a new same-directory path with native clonefile(no fallback copying),
      preserve the destination's permission/timestamps, verify bytes, then atomically
      replace only the owned duplicate. Record each transition immediately.
- [x] Rehash all package manifests/files after replacement and verify independence
      (distinct inode/linkcount1), free-space change and no source mutation.
- [ ] Rerun the unresolved required suite only if free space exceeds2GiB.

Abort on any unsupported clone, changed source, existing scratch path, mismatch or
insufficient headroom. Already completed identical replacements may remain because
no logical data changes. Receipts and exact maintenance script belong under tasks.

Outcome:62package files rehashed unchanged; clone mutation isolation passed. Free
space remains1.128GiB, so existing snapshots were already largely CoW/shared.
The2GiBsuite-headroom gate fails. No full-suite retry or large source append.
Continue with read-only PBMC qualification and the existing hosted CI handles.
