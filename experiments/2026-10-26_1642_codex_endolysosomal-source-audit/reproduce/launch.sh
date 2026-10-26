#!/usr/bin/env bash
set -euo pipefail
PRESTO_AUDIT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
PRESTO_AUDIT_PYTHON=${PRESTO_AUDIT_PYTHON:-/Users/iskander/code/shared-virtual-env/bin/python}
# Offline replay requires the original pinned assets at the paths in the ledger.
# No network request, model fitting, held-out scoring or automatic retry occurs.
exec "$PRESTO_AUDIT_PYTHON" "$PRESTO_AUDIT_DIR/code/verify.py"
