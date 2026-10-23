#!/bin/sh
set -eu
cd /private/tmp/presto-modular-model
PYTHONPATH=/private/tmp/hitlist-mapp-recipe /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_2330_codex_mapp-biological-comparison/code/admit.py
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_2330_codex_mapp-biological-comparison/code/verify_admission.py data/availability_recipes/2026-10-22-mapp-captured-composition-v1 experiments/2026-10-22_2330_codex_mapp-biological-comparison/results/admission-verification-v1.json
PYTHONPATH=/private/tmp/hitlist-mapp-recipe /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_2330_codex_mapp-biological-comparison/code/admit_min2.py
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_2330_codex_mapp-biological-comparison/code/verify_admission.py data/availability_recipes/2026-10-22-mapp-captured-composition-min2-v1 experiments/2026-10-22_2330_codex_mapp-biological-comparison/results/admission-verification-min2.json
PYTHONPATH=/private/tmp/presto-modular-imports:/private/tmp/hitlist-mapp-recipe OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_2330_codex_mapp-biological-comparison/code/train.py
