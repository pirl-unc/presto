#!/usr/bin/env bash
# First end-to-end run on real data after the assay-panel scale fix.
#
# Local CPU, deliberately small: the point was to prove the pipeline learns at
# all after the panel loss was corrected, and to get honest held-out numbers
# fast. It is NOT a performance run -- see README for what the numbers do and
# do not support.
set -euo pipefail

python -m presto train unified \
  --data-source hitlist \
  --hitlist-allele "HLA-A*02:01" \
  --max-binding 2500 \
  --max-elution 2500 \
  --max-tcell 800 \
  --max-vdjdb 800 \
  --max-stability 400 \
  --epochs 8 \
  --batch_size 32 \
  --d_model 64 \
  --n_layers 2 \
  --n_heads 4 \
  --run-dir "${RUN_DIR:-./out/panel-scale-fix-validation}" \
  --checkpoint "${RUN_DIR:-./out/panel-scale-fix-validation}/model.pt"
