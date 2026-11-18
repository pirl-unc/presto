#!/usr/bin/env bash
set -euo pipefail
# Prepared but not launched.
# Base commit: d79853d
# Branch: claude/restore-unmapped-masking
# Dirty: yes; see the working-tree diff for the data-trust remediation.
cd /Users/iskander/code/presto
modal run experiments/2026-09-04_1121_claude_groove-corrected-baseline/code/launch.py "$@"
