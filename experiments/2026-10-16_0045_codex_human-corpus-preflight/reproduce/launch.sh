#!/usr/bin/env bash
set -euo pipefail
family=experiments/2026-10-16_0045_codex_human-corpus-preflight
test -z "$(git status --porcelain)"
build_dir=$(mktemp -d /tmp/presto-human-wheel-XXXXXX)
python -m build --wheel --outdir "$build_dir"
export PRESTO_QUALIFICATION_WHEEL="$build_dir/presto-0.1.10-py3-none-any.whl"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
modal run "$family/code/launch.py"
