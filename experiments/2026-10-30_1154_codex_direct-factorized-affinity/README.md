# Direct factorized affinity decoder

Date: 2026-10-30. Agent/model: Codex / GPT-6. Status: launched on Modal; predictive results pending.

User-selected shared-v7 architecture: one 16-dimensional learned vector per
categorical value of quantity, preparation, mechanism and detector. Concatenate
these four slots with the common candidate latent and use one width-192 nonlinear
head (Linear/GELU/Linear) to predict the standardized log-concentration. The five
concentration endpoints share this head. There are no separate affinity heads or
small additive residuals. Unknown/conflicted method slots are fixed zeros;
source IDs are not decoder inputs. The same biological encoder and normalized
register pooling run for both MHC classes. Kinetics, stability, qualitative binding
and ligand outputs retain their own measurement definitions in the same model.

## Frozen data and training contract

Identical source bytes, observations, values, inequalities and split membership to
`../2026-10-29_2303_codex_assay-measurement-layer`. This is the curated Hitlist/IEDB
cohort, with existing study holdouts plus four retrospective fluorescence holdout
components and a cross-output peptide embargo. There are 3,518,242 training rows.
All source SHA256 values and original projection counts are frozen in
`reproduce/parent-prepared.json`. No new synthetic labels or new data curation:
inherited generated ligand comparisons retain their original contracts. Detector
metadata is source-tagged, not comprehensive verified protocol metadata.

Two fresh fits: direct-s42 and direct-s43, 4,000 updates each, batch128, width192,
3 layers, 6 heads, FF768, dropout0.1, constant AdamW3e-4, weight decay1e-4.
Same endpoint/class sampling as the pooled controls: 60% ligand, 20% literal IC50,
15% other physical measurements, 5% qualitative binding. No pretraining or warm
start. Selection: minimum equal-endpoint validation loss, evaluated each1,000
updates, checkpoints each500. Exact squared standardized residuals; one-sided
squared inequality violations; BCE for binary outcomes, unchanged weights.
The observation's quantity selects its established target scale and loss; its
method selects one output of the fixed shared decoder panel. Registry expansion
covers every usable concentration method in TRAIN, not just IC50. Known factors
in an unexposed combination mean indirect training, not direct combination evidence.

This family changes head sharing and decoder form relative to the pooled controls;
it does not isolate either change alone. All holdouts have previously been
inspected; no blind or biochemical qualification claim. The shared-v6 comparison
remains an immutable historical experiment, with any cancellation recorded there.

## Reproduction and artifacts

Exact invocation/environment and git/source fingerprints: `reproduce/launch.json`
and `reproduce/launch.sh`; model snapshot path is recorded there. New Modal source
upload reuses the already authorized data in volume `presto-checkpoints`.
Requested hardware: H100!, at most two concurrent GPUs. Actual hardware and memory
will be recorded in each report. Raw outputs: volume directory named after this
experiment; local verified downloads under repository `artifacts/<experiment>/`.
`analysis/watch.py` collects receipts, prediction dumps and metrics automatically.
Dashboard: http://127.0.0.1:54505/report.html .

## Verification and decision

Software fixtures test all categorical gradients/updates, both MHC classes,
shared quantity decoding, opposite assay-dependent ranking, exact/censored loss,
checkpoint/version behavior and the canonical manifest trainer. Artificial numbers
in those fixtures are only implementation/expressivity checks.

Held-out validation/test results, runtime, actual hardware, exact counts, bound
satisfaction, per-quantity/method/class regression and binding-threshold metrics,
ligand AUROC/AUPRC and full per-example dumps remain pending. No predictive gain
or winner is claimed. Architectural default follows the user's explicit decision.

## Launch receipt

Source revision: `7083bd899a29e8b251924079ca2941238e396784`. Frozen snapshot
contains 219 files (3,571,688 bytes). Modal app:
https://modal.com/apps/iskandr/main/ap-3BRMaO8FDAZyvk1gaz3JYs .
Coordinator: `fc-01M66S2W6JXF5JTYW580TJAG2D`. Two H100! fits; the old app was
stopped first and no Presto GPU containers remained. The CLI message “App
completed” means the detached local submission exited; remote fit status comes
from the coordinator/collector. `report.html` is an additive presentation update
showing frozen pooled controls beside new results; frozen training sources and
the original dashboard bytes are unchanged.

## Completed software verification

Full suite: **2,896 passed, four skipped**, 402.26 seconds. All 82 focused
source/decoder/support regressions pass; repository and experiment-source lint
and strict docs pass. A real completed v6 checkpoint produces identical CPU
predictions and fixed panel values under the frozen runtime and the new historical
reader for all four IC50 class/method cells. See `results/software-verification.json`.
The initial broad pass had two fixtures missing the now-required known assay
registry; those fixtures now explicitly declare their source-reported method.

One fit's startup inspection confirms NVIDIA H100 80GB HBM3, 85,009,367,040 device
bytes (81,559 MiB reported by nvidia-smi), Torch 2.14.1+cu130 and CUDA 13.0. The
11 source files total 30,252,937,352 bytes; staging/indexing occurs before the first
optimizer update. Final per-run hardware receipts will be collected automatically.
