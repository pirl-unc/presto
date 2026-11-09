#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
stage="$1"
shift
exec /Users/iskander/code/shared-virtual-env/bin/python -B "code/$stage.py" "$@"
