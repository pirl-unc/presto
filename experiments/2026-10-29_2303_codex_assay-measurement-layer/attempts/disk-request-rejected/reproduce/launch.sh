#!/bin/sh
set -eu
cd /Users/iskander/code/presto/.worktrees/shared-training/presto
PRESTO_MODAL_GPU=H100! /Users/iskander/code/shared-virtual-env/bin/python -m modal run --detach /Users/iskander/code/presto/.worktrees/shared-training/presto/experiments/2026-10-29_2303_codex_assay-measurement-layer/code/launch.py
