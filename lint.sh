#!/usr/bin/env bash
set -euo pipefail

# scripts/ was missing from this list until 2026-09-01. It is the largest
# directory in the repo and held 383 of the 638 long lines the widened config
# found, none of which CI could see. `experiments/` stays out on purpose:
# those are frozen snapshots of scripts as they ran, and editing them to
# satisfy a linter would falsify the record.
ruff check *.py cli/ data/ inference/ models/ scripts/ training/ tests/
