#!/bin/sh
set -eu
cd "/Users/iskander/code/presto/.worktrees/shared-training/presto"
PRESTO_MODAL_GPU=H100! /Users/iskander/code/shared-virtual-env/bin/modal run --detach "/Users/iskander/code/presto/.worktrees/shared-training/presto/experiments/2026-10-27_2255_codex_shared-multi-assay-scale/code/launch.py"
