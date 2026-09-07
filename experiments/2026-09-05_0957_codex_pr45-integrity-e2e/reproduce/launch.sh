#!/usr/bin/env bash
set -euo pipefail

repo_root="/Users/iskander/code/presto"
source_dir="$(dirname "$0")/source"

cd "$repo_root"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Refusing to launch from a dirty worktree" >&2
  exit 1
fi

"$source_dir/run_merged_preflight.sh"
"$source_dir/run_smoke.sh"
