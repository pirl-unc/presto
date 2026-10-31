#!/usr/bin/env bash
set -euo pipefail
# source revision 7083bd899a29e8b251924079ca2941238e396784
cd /Users/iskander/code/presto/.worktrees/shared-training/presto
export PRESTO_MODAL_GPU='H100!'
exec /Users/iskander/code/shared-virtual-env/bin/python -m modal run --detach /Users/iskander/code/presto/.worktrees/shared-training/presto/experiments/2026-10-30_1154_codex_direct-factorized-affinity/code/launch.py
