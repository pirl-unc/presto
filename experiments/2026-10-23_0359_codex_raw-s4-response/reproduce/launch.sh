#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export PYTHONPATH=/private/tmp/hitlist-ms-calibration OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-23_0359_codex_raw-s4-response/code/admit.py
/Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-23_0359_codex_raw-s4-response/code/verify_source.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-23_0359_codex_raw-s4-response/code/fit.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-23_0359_codex_raw-s4-response/code/verify_fit.py
