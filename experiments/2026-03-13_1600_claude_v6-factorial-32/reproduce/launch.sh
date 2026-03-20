#!/usr/bin/env bash
# Launch all 32 v6 factorial conditions on Modal.
# 16 condition specs × 2 content_conditioned = 32 runs.
# GPU: H100! (default), 50 epochs, batch_size 256.
set -euo pipefail

# cc0: content-independent (omit flag)
for cond_id in $(seq 1 16); do
    echo "Launching v6 cond ${cond_id} cc0..."
    modal run --detach scripts/train_modal.py::distributional_ba_v6_run \
        --cond-id "${cond_id}" \
        --epochs 50 \
        --batch-size 256
done

# cc1: content-conditioned (include flag)
for cond_id in $(seq 1 16); do
    echo "Launching v6 cond ${cond_id} cc1..."
    modal run --detach scripts/train_modal.py::distributional_ba_v6_run \
        --cond-id "${cond_id}" \
        --content-conditioned \
        --epochs 50 \
        --batch-size 256
done

echo "All 32 conditions launched."
