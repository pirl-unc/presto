#!/bin/sh
set -eu
TASK_FAMILY=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec /Users/iskander/code/shared-virtual-env/bin/python "$TASK_FAMILY/code/package.py"
