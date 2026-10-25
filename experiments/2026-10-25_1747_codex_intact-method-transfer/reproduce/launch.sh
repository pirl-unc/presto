#!/bin/sh
set -eu
export PYTHONSAFEPATH=1
exec /Users/iskander/code/shared-virtual-env/bin/python -B "/Users/iskander/code/presto/.worktrees/presto-modular-model/experiments/2026-10-25_1747_codex_intact-method-transfer/code/launch.py" "$@"
