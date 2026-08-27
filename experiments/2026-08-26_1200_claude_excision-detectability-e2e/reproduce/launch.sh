#!/usr/bin/env bash
# End-to-end validation: every branch of the model in one unified training run.
#
# IMPORTANT: run from a neutral directory. Putting ~/code on sys.path (as cwd or
# PYTHONPATH) makes ~/code/mhcseqs/ shadow the installed mhcseqs package as an
# empty namespace package, which silently disables all MHC sequence resolution.
set -euo pipefail

PRESTO_DIR="${PRESTO_DIR:-$HOME/code/presto}"
OUT_DIR="${OUT_DIR:-./out/liberation-detectability-e2e}"
mkdir -p "$OUT_DIR"
cd /tmp

python -u -m presto train unified \
  --data-dir "$PRESTO_DIR/data" \
  --data-source hitlist \
  --hitlist-mhc-class I \
  --hitlist-allele "HLA-A*02:01" \
  --latent-topology expanded \
  --bulk-ms --bulk-cell-line HeLa --max-bulk-ms 1500 \
  --max-binding 1200 --max-elution 1200 --max-tcell 300 --max-vdjdb 200 \
  --max-processing 200 --max-kinetics 50 --max-stability 200 \
  --cap-sampling head \
  --epochs 1 --batch_size 32 --d_model 32 --n_layers 2 --n_heads 4 \
  --run-dir "$OUT_DIR" \
  --checkpoint "$OUT_DIR/model.pt" \
  2>&1 | tee "$OUT_DIR/run.log"
