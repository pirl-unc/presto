#!/usr/bin/env bash
set -euo pipefail
FAMILY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PRESTO_PYTHON:-/Users/iskander/code/shared-virtual-env/bin/python}"
SOURCE_DIR="${PRESTO_DP_DM_SOURCE_ROOT:?Set to the pinned raw source directory}"
OUTPUT_DIR="${PRESTO_DP_DM_OUTPUT:?Set to a fresh extraction output directory}"
SDRF_PATH="${PRESTO_DP_DM_SDRF:?Set to the pinned PRIDE source SDRF}"
"$PYTHON_BIN" "$FAMILY_DIR/code/extract_mzid.py" \
 --inventory "$FAMILY_DIR/results/full-file-inventory.json" --sources "$SOURCE_DIR" \
 --sdrf "$SDRF_PATH" --constructs "$FAMILY_DIR/source-review/constructs.json" \
 --output "$OUTPUT_DIR/mzid"
"$PYTHON_BIN" "$FAMILY_DIR/code/verify_mzid.py" --sources "$SOURCE_DIR" \
 --extraction "$OUTPUT_DIR/mzid" --output "$OUTPUT_DIR/mzid-verification.json"
"$PYTHON_BIN" "$FAMILY_DIR/code/extract_native.py" --native-results "$SOURCE_DIR/native-results" \
 --samples "$OUTPUT_DIR/mzid/source-samples.json" --output "$OUTPUT_DIR/native"
"$PYTHON_BIN" "$FAMILY_DIR/code/verify_native.py" --native-results "$SOURCE_DIR/native-results" \
 --extraction "$OUTPUT_DIR/native" --samples "$OUTPUT_DIR/mzid/source-samples.json" \
 --output "$OUTPUT_DIR/native-verification.json"
