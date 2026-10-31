#!/bin/sh
set -eu
export PYTHONSAFEPATH=1
exec /Users/iskander/code/shared-virtual-env/bin/python -B /Users/iskander/code/presto/.worktrees/presto-modular-model/experiments/2026-10-26_0236_codex_nontryptic-source-package/code/launch_v2.py "$@"
