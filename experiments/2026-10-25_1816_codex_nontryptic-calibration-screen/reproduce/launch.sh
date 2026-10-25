#!/bin/sh
set -eu
export PYTHONSAFEPATH=1
exec /Users/iskander/code/shared-virtual-env/bin/python -B "/Users/iskander/code/presto/.worktrees/presto-modular-model/experiments/2026-10-25_1816_codex_nontryptic-calibration-screen/code/launch.py" "$@"
