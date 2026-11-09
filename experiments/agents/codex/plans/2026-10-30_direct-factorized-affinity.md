# One direct factorized affinity decoder

Date: 2026-10-30. Agent/model: Codex / GPT-6. Status: implementation authorized.

## User decision
Replace the small assay residual on separate quantity heads with a shared head
combining the biological latent and a separate embedding for every categorical
assay aspect. Make this the canonical new-training architecture, not another
optional experimental path. Do not claim a predictive gain before evaluation.

## Executable design
- Version the model as shared-v7 and freeze the complete v6 model/readout for exact
  historical inference. New training uses factorized mode only.
- A single nonlinear decoder consumes concatenated candidate latent h and four
  distinct embedding slots: quantity (IC50/KD/EC50/KD-via-IC50/KD-via-EC50),
  preparation, mechanism and detector. Use 16-dimensional categorical vectors and
  Linear -> GELU -> Linear to one scalar in the quantity's existing standardized
  log-concentration target space. Remove the five independent affinity heads.
- Retain the same biological encoder, groove interaction and register pooling for
  both MHC classes. Score all registered assay configurations before selection;
  observation metadata selects supervision, never changes the biological latent.
- Allocate categorical rows only for configured values. Unknown method slots are
  fixed zero vectors rather than randomly initialized untrained parameters;
  the same decoder predicts an explicitly unknown-method reference. Conflicted
  source tags use that reference. This is not an intrinsic thermodynamic KD.
- Keep exact/inequality losses, endpoint-specific train-only scales, all other
  biological outputs and source curation intact. Known complete descriptors must
  be in the frozen registry; do not silently discard them. Restrict this shared
  affinity decoder to concentration endpoints; kinetic and cellular processes
  retain their explicitly defined endpoints.
- Ensure support, checkpoint round trips, warm-start rejection across versions,
  freeze dependency closure and per-coordinate update evidence cover the decoder.

Named categorical vectors allow quantity-only transfer without changing old rows.
Supported but unexposed combinations are reported as indirectly trained only if
all their factors and the shared decoder have actual update evidence. They are
not described as directly measured or scientifically validated.

## Verification
- Prove the five quantities share one head and both classes traverse it.
- Prove every selected categorical factor and the biological latent receive
  gradients and actual updates; unselected rows have no data gradient.
- Fit an artificial nonlinear interaction fixture with the real decoder to show
  it can learn assay-dependent ordering, beyond scalar calibration. This checks
  expressivity only and is never reported as biological accuracy.
- Verify fixed-panel invariance, unknown/conflict handling, both censor directions,
  untrained-track rejection, historical inference, save/load and common-manifest
  training. Run relevant regression suites and repository lint/docs checks.

## Real-data run
Register a new immutable experiment family. Reuse the prepared cohort and splits
from 2026-10-29_2303_codex_assay-measurement-layer, with unchanged values, censors,
source holdouts and sampling. Expand the declared assay registry to all admitted
training concentration observations. Run the direct factorized model for seeds
42/43, 4,000 updates, same width192/3layers/6heads, batch128 and AdamW3e-4 as
pooled controls. Preserve validation/test dumps, exact n, bounds, per-assay/class
regression, threshold metrics and ligand AUROC/AUPRC. These are inspected
retrospective development holdouts, not a blind test. Frozen pooled runs provide
context; this change combines head sharing with nonlinear decoding.

Use at most two H100! GPUs, including older active fits. Preserve existing finished
results. If replacing the older remaining control queue, record cancellation and
collect available artifacts before launching the requested direct decoder.
Record frozen source, invocation, dataset hashes, hardware, runtime and results in
README, reproduce/launch.*, results and experiments/experiment_log.md. Never edit
an already launched source snapshot. Report actual observations, not promises of
improvement. Complete closure when fits finish.
