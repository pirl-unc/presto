#!/usr/bin/env bash
set -euo pipefail

repo_root="/Users/iskander/code/presto"
experiment_dir="${repo_root}/experiments/2026-09-05_2213_codex_pr45-lineage-funnel-closure"
expected_commit="15003dacf310de3534325cb2db7fd6f3e2e89481"

cd "$repo_root"
if ! git cat-file -e "${expected_commit}^{commit}"; then
  echo "Missing recorded code commit $expected_commit" >&2
  exit 1
fi
if ! git diff --quiet "$expected_commit" -- \
  data/loaders.py scripts/train_iedb.py training/data_support.py; then
  echo "Production files do not match recorded code commit $expected_commit" >&2
  exit 1
fi
if [[ -e "${experiment_dir}/results" ]]; then
  echo "Refusing to overwrite existing results" >&2
  exit 1
fi

mkdir -p "${experiment_dir}/results/hitlist_preflight"
python "${experiment_dir}/reproduce/source/verify_class2_lineage.py" \
  > "${experiment_dir}/results/class2_lineage.json"
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
  --run-dir "${experiment_dir}/results/hitlist_preflight" \
  2>&1 | tee "${experiment_dir}/results/hitlist_preflight/preflight.log"
python "${experiment_dir}/reproduce/source/verify_outputs.py" "$experiment_dir" \
  > "${experiment_dir}/results/verification_summary.json"
