#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: bf7fcbfcc2cfdcbaa32f6424b7f7ad0e09ceb857
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
export PRESTO_MODAL_GPU='H100!'
'/Users/iskander/code/shared-virtual-env/bin/python' 'experiments/2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun/code/launch.py' '--out-dir' 'experiments/2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun'
