#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PRESTO_MODAL_WHEEL="$PWD/presto-0.1.22-py3-none-any.whl"
export PRESTO_MODAL_GPU="H100!"
python -m modal run --detach modal_workflow.py --request request.json
