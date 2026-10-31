#!/bin/sh
set -eu
# Pass a fresh artifact directory as $1; refuses existing per-variant outputs.
root=/Users/iskander/code/presto/artifacts/2026-10-30_1725_codex_preparation-throughput
for variant in baseline candidate; do
  /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-30_1725_codex_preparation-throughput/reproduce/source/software_parity.py --runtime "$root/$variant" --output "$1/software-$variant"
done
