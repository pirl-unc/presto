#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
for asset_kind in project sample datasets; do
  /Users/iskander/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python \
    experiments/2026-10-24_0938_codex_native-apc-source-search/code/retrieve.py \
    --url "http://www.inbirg.com/ppd_static/exports/${asset_kind}-download.xlsx" \
    --out "data/native_apc_source_search/2026-10-24_0938_peptidomicsdb/${asset_kind}-download.xlsx"
done
/Users/iskander/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python \
  experiments/2026-10-24_0938_codex_native-apc-source-search/code/inventory.py \
  --data-dir data/native_apc_source_search/2026-10-24_0938_peptidomicsdb \
  --out experiments/2026-10-24_0938_codex_native-apc-source-search/results/inventory-v1
/Users/iskander/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python \
  experiments/2026-10-24_0938_codex_native-apc-source-search/code/verify.py \
  --data-dir data/native_apc_source_search/2026-10-24_0938_peptidomicsdb \
  --out experiments/2026-10-24_0938_codex_native-apc-source-search/results/verification.json
/Users/iskander/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python \
  experiments/2026-10-24_0938_codex_native-apc-source-search/code/review_cases.py \
  --data-dir data/native_apc_source_search/2026-10-24_0938_peptidomicsdb \
  --out experiments/2026-10-24_0938_codex_native-apc-source-search/results/reviewed-source-cases.json
