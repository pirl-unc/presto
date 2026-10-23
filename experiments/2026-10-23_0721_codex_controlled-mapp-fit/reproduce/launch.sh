#!/bin/bash
set -euo pipefail
cd /private/tmp/presto-modular-model
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_0721_codex_controlled-mapp-fit/code/train.py
