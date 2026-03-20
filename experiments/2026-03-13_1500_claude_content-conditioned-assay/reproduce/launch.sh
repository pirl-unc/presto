#!/usr/bin/env bash
# Launch all 6 v5 content-conditioned assay context conditions on Modal.
# GPU: H100! (default), 50 epochs, batch_size 256, MHCflurry additive, MAX=50k.
set -euo pipefail

for cond_id in 1 2 3 4 5 6; do
    echo "Launching v5 condition ${cond_id}..."
    modal run --detach scripts/train_modal.py::distributional_ba_v5_run \
        --cond-id "${cond_id}" \
        --epochs 50 \
        --batch-size 256
done

echo "All 6 conditions launched."
