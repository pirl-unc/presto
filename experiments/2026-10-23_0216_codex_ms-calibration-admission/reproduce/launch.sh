#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
OPENBLAS_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-modular-model/experiments/2026-10-23_0216_codex_ms-calibration-admission/code/launch.py
