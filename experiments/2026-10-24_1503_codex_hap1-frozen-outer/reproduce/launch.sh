#!/bin/sh
set -eu
cd '/private/tmp/presto-modular-model'
export PYTHONPATH='/private/tmp/presto-native-import'
export OMP_NUM_THREADS='1'
export OPENBLAS_NUM_THREADS='1'
export MKL_NUM_THREADS='1'
/Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-24_1503_codex_hap1-frozen-outer/code/launch.py --phase prepare
/Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-24_1503_codex_hap1-frozen-outer/code/launch.py --phase evaluate
/Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-24_1503_codex_hap1-frozen-outer/code/verify.py
