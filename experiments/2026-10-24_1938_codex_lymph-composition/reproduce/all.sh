#!/bin/sh
set -eu
# Use a fresh experiment copy and explicitly relocated recipe/prepared destinations.
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
sh experiments/2026-10-24_1938_codex_lymph-composition/reproduce/launch.sh
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_1938_codex_lymph-composition/analysis/verify_admission.py
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_1938_codex_lymph-composition/code/qualify.py
sh experiments/2026-10-24_1938_codex_lymph-composition/reproduce/fit.sh
sh experiments/2026-10-24_1938_codex_lymph-composition/reproduce/evaluate.sh
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_1938_codex_lymph-composition/analysis/verify_predictions.py
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_1938_codex_lymph-composition/analysis/report.py
