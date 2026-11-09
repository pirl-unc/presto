#!/bin/bash
set -euo pipefail
cd /private/tmp/presto-modular-model
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_0645_codex_controlled-mapp-signal/code/signal.py
