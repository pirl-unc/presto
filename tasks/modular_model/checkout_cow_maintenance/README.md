# Preserved storage-maintenance helpers

The `executed-source/*.py.txt` files retain the exact bytes of the helpers used
for the recorded storage operation, including the initial helper that stopped
before making a clone. Their hashes are in `source-preservation.json`.

The corresponding `.py` files have formatting-only changes to satisfy repository
lint. AST equality was verified for all three pairs; the helpers were not rerun.
No file, experiment result, resource guard or cloning behavior changed.

CI40152836732 caught style errors in these task-level files. The isolated
experiment check did not cover them. The corrected tracked checkout passes both
Ruff checks and formatting, while unrelated untracked historical source snapshots
remain unchanged. Original plan/results are in the parent directory.
