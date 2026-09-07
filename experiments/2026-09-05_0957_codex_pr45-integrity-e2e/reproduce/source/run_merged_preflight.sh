#!/usr/bin/env bash
set -euo pipefail

repo_root="/Users/iskander/code/presto"
run_dir="${repo_root}/experiments/2026-09-05_0957_codex_pr45-integrity-e2e/results/merged_preflight"

cd "$repo_root"

if [[ -e "$run_dir" ]]; then
  echo "Refusing to overwrite existing run directory: $run_dir" >&2
  exit 1
fi

mkdir -p "$run_dir"
git rev-parse HEAD > "$run_dir/git_commit.txt"
git branch --show-current > "$run_dir/git_branch.txt"

python -m presto train unified \
  --data-dir data \
  --data-source merged_tsv \
  --max-binding 96 \
  --max-kinetics 24 \
  --max-stability 24 \
  --max-processing 24 \
  --max-elution 96 \
  --max-tcell 96 \
  --max-vdjdb 24 \
  --cap-sampling reservoir \
  --data-seed 42 \
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
  --require-traceable-lineage \
  --forbid-fake-null-sequences \
  --device cpu \
  --num-workers 0 \
  --no-pin-memory \
  --no-compile \
  --no-balanced-batches \
  --no-profile-performance \
  --no-track-probe-affinity \
  --no-track-probe-motif-scan \
  --no-track-pmhc-flow \
  --no-track-output-latent-stats \
  --seed 42 \
  --data-preflight-only \
  --run-dir "$run_dir" \
  2>&1 | tee "$run_dir/preflight.log"

test -s "$run_dir/data_funnel.json"
test -s "$run_dir/data_funnel.csv"
test -s "$run_dir/split_support.json"
test -s "$run_dir/split_support.csv"
