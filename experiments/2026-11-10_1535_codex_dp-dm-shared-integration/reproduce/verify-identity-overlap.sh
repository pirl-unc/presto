#!/usr/bin/env bash
set -euo pipefail
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-11-10_1535_codex_dp-dm-shared-integration/reproduce/source/verify_identity_overlap.py --summary experiments/2026-11-10_1535_codex_dp-dm-shared-integration/results/identity-overlap-summary.json --membership /private/tmp/presto-dp-dm-shared-20261110/identity-overlap-v1/candidate-identity-overlaps.parquet --output experiments/2026-11-10_1535_codex_dp-dm-shared-integration/results/independent-overlap-replay.json
