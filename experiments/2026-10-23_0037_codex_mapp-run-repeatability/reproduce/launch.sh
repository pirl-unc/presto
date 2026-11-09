#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
PYTHONPATH=/private/tmp/hitlist-mapp-recipe /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_0037_codex_mapp-run-repeatability/code/admit.py
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_0037_codex_mapp-run-repeatability/code/verify_admission.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_0037_codex_mapp-run-repeatability/code/analyze.py
OPENBLAS_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_0037_codex_mapp-run-repeatability/code/verify_analysis.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_0037_codex_mapp-run-repeatability/code/fit_linear.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_0037_codex_mapp-run-repeatability/code/verify_linear.py
MPLCONFIGDIR=/private/tmp/presto-mapp-matplotlib OPENBLAS_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-23_0037_codex_mapp-run-repeatability/code/report.py
