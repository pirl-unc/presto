#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec /Users/iskander/code/shared-virtual-env/bin/python "$TASK_ROOT/code/launch.py" "$@"
