#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_1138_codex_digestion-temporal-signal/code/signal_check.py
