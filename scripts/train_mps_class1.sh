#!/usr/bin/env bash
# Full MHC Class I training on Apple Silicon (MPS) using the L2 recipe.
#
# L2 = dag_prep_readout_leaf + assay_heads_only + lr=3e-4 warmup_cosine
#       + d128 + pretrained
#
# This is the 7-allele bakeoff winner (Spearman 0.806, F1 0.820, probe 2502x).
# See: experiments/2026-03-17_1042_claude_7allele-bakeoff/README.md
#
# Prerequisites:
#   1. pip install -e .   (from repo root)
#   2. Pretrain checkpoint at modal_runs/pulls/mhc-pretrain-20260308b/mhc_pretrain.pt
#      (fetch from Modal volume if missing: modal volume get presto-checkpoints mhc-pretrain-20260308b/ modal_runs/pulls/)
#   3. data/merged_deduped.tsv and data/mhc_index.csv must exist
#      (run: python -m presto data merge --datadir data)
#
# Usage:
#   bash scripts/train_mps_class1.sh                    # default: 50 epochs, seed 42
#   EPOCHS=10 SEED=43 bash scripts/train_mps_class1.sh  # override epochs/seed

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Configurable parameters (override via env vars)
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-3e-4}"
D_MODEL="${D_MODEL:-128}"
CHECKPOINT="${CHECKPOINT:-modal_runs/pulls/mhc-pretrain-20260308b/mhc_pretrain.pt}"
RUN_ID="${RUN_ID:-mps-class1-L2-e${EPOCHS}-s${SEED}}"

# Validate prerequisites
if [ ! -f "data/merged_deduped.tsv" ]; then
    echo "ERROR: data/merged_deduped.tsv not found. Run: python -m presto data merge --datadir data"
    exit 1
fi
if [ ! -f "data/mhc_index.csv" ]; then
    echo "ERROR: data/mhc_index.csv not found."
    exit 1
fi

CHECKPOINT_ARGS=""
if [ -f "$CHECKPOINT" ]; then
    CHECKPOINT_ARGS="--init-checkpoint $CHECKPOINT"
    echo "Using pretrain checkpoint: $CHECKPOINT"
else
    echo "WARNING: Pretrain checkpoint not found at $CHECKPOINT — running cold start"
fi

echo "============================================"
echo "  Full Class I Training on MPS (L2 recipe)"
echo "  epochs=$EPOCHS  seed=$SEED  d_model=$D_MODEL"
echo "  lr=$LR  schedule=warmup_cosine"
echo "  run_id=$RUN_ID"
echo "============================================"

python scripts/focused_binding_probe.py \
    --device mps \
    --mps-safe-mode auto \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --seed "$SEED" \
    --split-seed "$SPLIT_SEED" \
    --d-model "$D_MODEL" \
    --n-layers 2 \
    --n-heads 4 \
    --affinity-assay-residual-mode dag_prep_readout_leaf \
    --affinity-loss-mode assay_heads_only \
    --affinity-target-encoding mhcflurry \
    --lr "$LR" \
    --lr-schedule warmup_cosine \
    --weight-decay 0.01 \
    --alleles HLA-A*02:01,HLA-A*24:02,HLA-A*03:01,HLA-A*11:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02 \
    --train-all-alleles \
    --train-mhc-class-filter I \
    --probe-peptide SLLQHLIGL \
    --extra-probe-peptides FLRYLLFGI,NFLIKFLLI \
    --measurement-profile numeric_no_qualitative \
    --qualifier-filter all \
    --peptide-pos-mode concat_start_end_frac \
    --groove-pos-mode concat_start_end_frac \
    --binding-core-lengths 8,9,10,11 \
    --binding-core-refinement shared \
    --kd-grouping-mode split_kd_proxy \
    --max-affinity-nm 100000 \
    --no-synthetic-negatives \
    --binding-contrastive-weight 0 \
    --binding-peptide-contrastive-weight 0 \
    --probe-plot-frequency off \
    --design-id "L2-class1-mps" \
    --merged-tsv data/merged_deduped.tsv \
    --index-csv data/mhc_index.csv \
    $CHECKPOINT_ARGS \
    "$@"
