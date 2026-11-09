#!/bin/sh
set -eu
cd /Users/iskander/code/presto/.worktrees/shared-training/presto
exec /Users/iskander/code/shared-virtual-env/bin/python -m modal run --detach experiments/2026-10-30_1725_codex_preparation-throughput/code/launch.py
