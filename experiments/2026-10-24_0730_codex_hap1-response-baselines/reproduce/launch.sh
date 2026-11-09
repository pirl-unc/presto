#!/usr/bin/env bash
set -euo pipefail
# Registered against Presto6045dd2 / Hitlist2f4bbe0 plus frozen recipe source.
# Use fresh experiment and recipe destinations: launchers never overwrite attempts.
ROOT=/private/tmp/presto-modular-model
EXPERIMENT="$ROOT/experiments/2026-10-24_0730_codex_hap1-response-baselines"
RECIPE="$ROOT/data/hap1_response_recipes/2026-10-24_0730_v1"
SOURCE="$ROOT/data/hap1_quantitative_source_admission/2026-10-24_0612_codex_hap1-quantitative-admission/canonical-package-v2"
PYTHON=/Users/iskander/code/shared-virtual-env/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/private/tmp/hitlist-hap1-recipe
cd "$ROOT"
"$PYTHON" "$EXPERIMENT/code/admit.py" --experiment "$EXPERIMENT" --source "$SOURCE" --output "$RECIPE"
"$PYTHON" "$EXPERIMENT/analysis/verify_admission.py" --recipe "$RECIPE" --output "$EXPERIMENT/results/independent-admission.json"
"$PYTHON" "$EXPERIMENT/code/fit.py" --experiment "$EXPERIMENT" --recipe "$RECIPE"
"$PYTHON" "$EXPERIMENT/analysis/verify_fits.py" --experiment "$EXPERIMENT"
"$PYTHON" "$EXPERIMENT/analysis/report.py" --experiment "$EXPERIMENT"
