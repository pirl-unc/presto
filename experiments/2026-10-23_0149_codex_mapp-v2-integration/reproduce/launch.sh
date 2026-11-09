#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
PYTHONPATH=/private/tmp/hitlist-mapp-recipe OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-23_0149_codex_mapp-v2-integration/code/launch.py
OPENBLAS_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-23_0149_codex_mapp-v2-integration/code/verify.py
