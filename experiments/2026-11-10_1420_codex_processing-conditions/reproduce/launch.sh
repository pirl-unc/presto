#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd "$(dirname "$0")/.." && pwd)"
"${PRESTO_AUDIT_PYTHON:-/Users/iskander/code/shared-virtual-env/bin/python}" \
  "$experiment_dir/reproduce/source/audit.py" \
  --hitlist "${PRESTO_HITLIST_CHECKOUT:-/Users/iskander/code/hitlist}" \
  --output "${PRESTO_CONDITION_AUDIT_OUTPUT:-$experiment_dir/results}"
