#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH=/private/tmp/hitlist-dp-dm-source-20261110:/private/tmp/presto-conditions-import
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-11-10_1535_codex_dp-dm-shared-integration/reproduce/source/verify_shared-attempt02.py --native /private/tmp/presto-dp-dm-source-20261110/native-extraction-v2/native-peptide-samples.jsonl.gz --catalog /Users/iskander/code/presto/artifacts/2026-11-07_1645_codex_unified-data-audit/upload/mhc-full-seqs.csv --output /private/tmp/presto-dp-dm-shared-20261110/attempt02
