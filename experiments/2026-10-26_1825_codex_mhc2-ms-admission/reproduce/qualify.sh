#!/bin/sh
set -eu
# Run from the registered Presto checkout after launch.sh has acquired raw bytes.
# Existing profile/admission outputs are intentionally refused, not overwritten.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=/private/tmp/presto-mhc2-20261026:/private/tmp/presto-pr-completion-20261026/dependencies
TASK_PYTHON=/Users/iskander/code/shared-virtual-env/bin/python
TASK_FAMILY=experiments/2026-10-26_1825_codex_mhc2-ms-admission
"$TASK_PYTHON" "$TASK_FAMILY/code/profile_inputs.py"
"$TASK_PYTHON" "$TASK_FAMILY/code/admit.py"
"$TASK_PYTHON" "$TASK_FAMILY/code/verify_admission.py"
"$TASK_PYTHON" "$TASK_FAMILY/code/check_supplement_chemistry.py"
