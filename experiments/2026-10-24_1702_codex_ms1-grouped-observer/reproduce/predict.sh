#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/private/tmp/hitlist-proteometools-ms1
observer_root=experiments/2026-10-24_1702_codex_ms1-grouped-observer
observer_python=/Users/iskander/code/shared-virtual-env/bin/python
"$observer_python" "$observer_root/code/admit.py" > "$observer_root/launch_logs/admission.log" 2>&1
"$observer_python" "$observer_root/analysis/verify_admission.py" > "$observer_root/launch_logs/source-verification.log" 2>&1
"$observer_python" "$observer_root/code/fit.py" > "$observer_root/launch_logs/fit.log" 2>&1
"$observer_python" "$observer_root/code/evaluate.py" > "$observer_root/launch_logs/evaluation.log" 2>&1
"$observer_python" "$observer_root/analysis/verify_predictions.py" > "$observer_root/launch_logs/prediction-verification.log" 2>&1
"$observer_python" "$observer_root/analysis/plot_results.py" > "$observer_root/launch_logs/plot.log" 2>&1
