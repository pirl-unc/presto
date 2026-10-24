#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for script in provenance_and_plot closure publish_predictions; do
 /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_2244_codex_mapp2025-baseline-prediction/analysis/$script.py
done
