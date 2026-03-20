#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: e01eea1e91eb3b07ac5a8d75e65956ef688cccfb
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
export PRESTO_EXPERIMENT_AGENT='claude'
export PRESTO_MODAL_GPU='H100!'
'/Users/iskander/code/shared-virtual-env/bin/python3' 'benchmark_distributional_ba_heads.py' '--epochs' '20' '--batch-size' '256' '--agent-label' 'claude' '--prefix' 'dist-ba-v1'
