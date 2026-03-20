#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: bf7fcbfcc2cfdcbaa32f6424b7f7ad0e09ceb857
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
'/Users/iskander/code/shared-virtual-env/bin/python' 'experiments/2026-03-19_1236_codex_cpu-vs-mps-focused-pf07-minitrain-mpssafe/code/launch.py' '--dry-run' '--out-dir' 'experiments/2026-03-19_1236_codex_cpu-vs-mps-focused-pf07-minitrain-mpssafe'
