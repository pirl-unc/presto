#!/usr/bin/env bash
set -euo pipefail
family=experiments/2026-10-16_0110_codex_scoped-sampling-qualification
test -z "$(git status --porcelain)"
build_dir=$(mktemp -d /tmp/presto-scoped-sampling-wheel-XXXXXX)
python -m build --wheel --outdir "$build_dir"
export PRESTO_QUALIFICATION_WHEEL="$build_dir/presto-0.1.11-py3-none-any.whl"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
modal run "$family/code/launch.py"
