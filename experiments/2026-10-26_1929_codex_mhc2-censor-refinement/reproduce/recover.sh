#!/bin/sh
set -eu
cd /private/tmp/presto-mhc2-20261026/presto
env PYTHONPATH=/private/tmp/presto-mhc2-20261026:/private/tmp/presto-pr-completion-20261026/dependencies OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 /Users/iskander/code/shared-virtual-env/bin/python /private/tmp/presto-mhc2-20261026/presto/experiments/2026-10-26_1929_codex_mhc2-censor-refinement/code/recover.py
