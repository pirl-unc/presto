#!/bin/sh
set -eu
export PRESTO_MODAL_GPU="H100!"
exec modal run --detach "/Users/iskander/code/presto/.worktrees/shared-training/presto/experiments/2026-10-28_1540_codex_temporal-multi-output/code/launch.py" "$@"
