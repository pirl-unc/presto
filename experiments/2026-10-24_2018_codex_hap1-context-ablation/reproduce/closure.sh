#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
/Users/iskander/code/shared-virtual-env/bin/python -u experiments/2026-10-24_2018_codex_hap1-context-ablation/analysis/verify.py
# Successful final plot used the existing writable temporary cache below.
MPLCONFIGDIR=/var/folders/qm/qf74v01n2s12j3w0h1yrx4l00000gn/T/matplotlib-nt0aky8w /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_2018_codex_hap1-context-ablation/analysis/report.py
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_2018_codex_hap1-context-ablation/analysis/audit_binding_headers.py
