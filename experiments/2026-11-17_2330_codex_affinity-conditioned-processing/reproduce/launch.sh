#!/usr/bin/env bash
set -euo pipefail
export PRESTO_MODAL_GPU="${PRESTO_MODAL_GPU:-H100!}"
EXP_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# Core 0c21bc4da1066ebfadf713dca478d655c31d4b8b; model-v2 wrapper snapshot.
if [ "$#" -eq 0 ]; then
    set -- --arms continued,paired,paired-junction --seeds 42,43 --steps 2000
fi
exec "${PRESTO_PYTHON:-/Users/iskander/code/shared-virtual-env/bin/python}" \
    "$EXP_DIR/code/dns_transport.py" --module modal run --detach \
    "$EXP_DIR/code/launch.py" "$@"
