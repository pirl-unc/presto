#!/bin/sh
set -eu
export PRESTO_MODAL_GPU="H100!"
exec modal run --detach "/Users/iskander/code/presto/.worktrees/shared-training/presto/experiments/2026-10-29_1956_codex_curated-data-retrain/code/launch.py" "$@"
