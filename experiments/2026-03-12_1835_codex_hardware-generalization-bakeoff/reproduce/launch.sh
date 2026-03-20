#!/usr/bin/env bash
set -euo pipefail
# Git state at launch:
#   commit: e01eea1e91eb3b07ac5a8d75e65956ef688cccfb
#   branch: main
#   dirty: yes
cd '/Users/iskander/code/presto'
'/Users/iskander/code/shared-virtual-env/bin/python' 'scripts/benchmark_hardware_generalization.py' '--out-dir' 'experiments/2026-03-12_1835_codex_hardware-generalization-bakeoff'
