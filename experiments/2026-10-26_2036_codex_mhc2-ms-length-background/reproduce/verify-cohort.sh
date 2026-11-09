#!/bin/sh
set -eu
cd /private/tmp/presto-mhc2-20261026/presto
env PYTHONPATH=/private/tmp/presto-mhc2-20261026:/private/tmp/presto-pr-completion-20261026/dependencies OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-mhc2-20261026/presto/experiments/2026-10-26_2036_codex_mhc2-ms-length-background/code/verify_cohort.py
