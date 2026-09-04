#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: caa57abe249660c09765323e5bb45336a5095dc7
#   branch: claude/flank-context-and-data-fixes
#   dirty: yes
cd '/Users/iskander/code/presto'
'/Users/iskander/code/shared-virtual-env/bin/python3' '/Users/iskander/code/shared-virtual-env/bin/modal' 'run' '--detach' 'experiments/2026-09-03_1746_claude_flank-context-fixes/code/launch.py'
