#!/usr/bin/env bash
set -euo pipefail

repo_root="/Users/iskander/code/presto"
run_dir="${repo_root}/experiments/2026-09-05_0957_codex_pr45-integrity-e2e/results/run"

cd "$repo_root"

if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing run directory: $run_dir" >&2
  exit 1
fi
mkdir -p "$run_dir"
git rev-parse HEAD > "$run_dir/git_commit.txt"
git branch --show-current > "$run_dir/git_branch.txt"

env HITLIST_DATA_DIR=/Users/iskander/.hitlist \
  python -m presto train unified \
  --data-dir data \
  --data-source hitlist \
  --hitlist-allele 'HLA-A*02:01' \
  --source-mapping-policy mask_unresolved \
  --max-binding 96 \
  --max-kinetics 2 \
  --max-stability 24 \
  --max-elution 96 \
  --cap-sampling reservoir \
  --data-seed 42 \
  --exclude-target kon \
  --exclude-target koff \
  --exclude-target tm \
  --synthetic-pmhc-negative-ratio 0 \
  --synthetic-elution-negative-ratio 0 \
  --synthetic-cascade-elution-negative-ratio 0 \
  --synthetic-cascade-tcell-negative-ratio 0 \
  --synthetic-class-i-no-mhc-beta-negative-ratio 0 \
  --synthetic-processing-negative-ratio 0 \
  --mhc-augmentation-samples 0 \
  --uniprot-negative-ratio 0 \
  --val-frac 0.2 \
  --test-frac 0.2 \
  --split-mode peptide_group \
  --require-split-target binding \
  --require-split-target t_half \
  --require-split-target elution \
  --require-traceable-lineage \
  --forbid-fake-null-sequences \
  --epochs 1 \
  --batch_size 16 \
  --d_model 32 \
  --n_layers 1 \
  --n_heads 4 \
  --max-batches 2 \
  --max-val-batches 1 \
  --max-mil-instances 16 \
  --device cpu \
  --num-workers 0 \
  --no-pin-memory \
  --no-amp \
  --no-compile \
  --no-balanced-batches \
  --no-uncertainty-weighting \
  --no-profile-performance \
  --no-track-probe-affinity \
  --no-track-probe-motif-scan \
  --no-track-pmhc-flow \
  --no-track-output-latent-stats \
  --seed 42 \
  --checkpoint "$run_dir/model.pt" \
  --run-dir "$run_dir" \
  2>&1 | tee "$run_dir/train.log"

test ! -e "$run_dir/holdout_error.json"
test -s "$run_dir/val_predictions.csv"
test -s "$run_dir/test_predictions.csv"
test -s "$run_dir/val_summary.json"
test -s "$run_dir/test_summary.json"
