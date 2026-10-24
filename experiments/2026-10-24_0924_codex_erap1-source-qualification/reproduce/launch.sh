#!/bin/sh
set -eu
# Presto e6ce3b2477e34ae636012d416f2a9e092a304e4b + frozen audit sources/task dirt.
cd /private/tmp/presto-modular-model
/Users/iskander/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python \
  experiments/2026-10-24_0924_codex_erap1-source-qualification/code/launch.py \
  --archive /private/tmp/hitlist-erap1-809-assets/maben2021-source-data.zip \
  --out data/erap1_source_qualification/2026-10-24_0924_maben-v1
/Users/iskander/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python \
  experiments/2026-10-24_0924_codex_erap1-source-qualification/code/retrieve.py \
  --url https://www.ebi.ac.uk/europepmc/webservices/rest/PMC8421391/fullTextXML \
  --out data/erap1_source_qualification/2026-10-24_0924_maben-v1/article.xml
/Users/iskander/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python \
  experiments/2026-10-24_0924_codex_erap1-source-qualification/code/verify.py \
  --archive /private/tmp/hitlist-erap1-809-assets/maben2021-source-data.zip \
  --inventory-dir data/erap1_source_qualification/2026-10-24_0924_maben-v1 \
  --out experiments/2026-10-24_0924_codex_erap1-source-qualification/results/verification.json
/Users/iskander/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python \
  experiments/2026-10-24_0924_codex_erap1-source-qualification/code/classify.py \
  --inventory-dir data/erap1_source_qualification/2026-10-24_0924_maben-v1 \
  --out experiments/2026-10-24_0924_codex_erap1-source-qualification/results
