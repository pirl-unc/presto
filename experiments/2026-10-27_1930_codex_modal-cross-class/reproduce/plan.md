# Modal cross-class comparison — prospective hardware revision

Codex / GPT-6. User explicitly authorized Modal and asked to remove the arbitrary
local2GiB cap/startup delays. This supersedes the earlier local-only preference.
The in-flight CPU shared-I seed42 run remains a separately labeled reference;
remaining CPU conditions are superseded. Primary comparison runs all ten original
conditions on requested PRESTO_MODAL_GPU=H100!, at most two concurrent GPU workers,
4CPU cores,16GiB host memory, float32, no AMP/compile/TF32, max2hours per job and
no automatic numerical retries. Record actual GPU name/memory, CUDA/library versions,
wall time, peak allocated/reserved GPU memory and hostRSS. Do not compare CPU and
GPU runtime as the same contract. All10GPU fits use the SAME manifest trainer and
optimizer step as their frozen model version; no alternative training loop.

Seeds42/43 × shared-v4 I-only/II-only/joint, full-context-v3 joint, three-aa-v2 joint.
Exactly1000updates, batch64, AdamW3e-4 wd1e-4, dropout.1, max/min epochs100,
patience100, identity weight0, no pretraining/synthetic biological labels.
Model source hashes match the local benchmark. Source-only validation selects
checkpoints. Opposite-class validation/test only after selection, through the same
model with routing expanded and ALL weight tensors exactly preserved. Target-class
labels never train, calibrate or select the source-only model. Serving support remains
unchanged. This is a diagnostic cross-class evaluation, not qualified deployment.

Reuse precisely the admitted original full-IC50 split and lossless class filters:
train21636 exact(I14375/II7261), val2925(I2051/II874), test2803(I1845/II958).
Source manifest SHA256914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be.
Direct purified humanIC50→log10(nM), train-only standardization, squared standardized
error plus existing one-sided bound loss on heldouts. Missing export qualifiers
follow original exact-policy with missingness/provenance retained. No KD/MS/TCR
proxies. No row caps or resplitting; all original hashes and global peptide/measurement
separation are verified by the prior admission. Observed9mer overlap is a prespecified
secondary stratum, not an assigned binding-core label. Holdouts are historically
inspected and share studies; these are development results, not new independent
scientific qualification. Equal updates imply unequal epochs/class exposures.

Freeze source-only package snapshots (no credentials or unrelated workspace data),
exact manifests/input hashes, pinned image requirements and launch invocation.
Upload only required source/data to Modal. Per-job immutable output directories,
checkpoint files and prediction dumps persist in a dedicated family path on the
presto-checkpoints volume. Publish only generated progress/metrics to a private
Modal Dict for local dashboard polling. Download every completed raw run as a
checksummed archive; independent metric verification is local, without test rescoring.

Metrics: validation/test all-label loss; exact Spearman/Pearson/RMSE; <=500nM
accuracy, balanced accuracy, precision, recall,F1,AUROC,AUPRC; source-mean baselines,
censor satisfaction, length/9mer familiarity/per-MHC strata and paired-seed deltas.
Retain selected per-example val/test predictions and all conditions, including failures.
The live view shows training/source-validation curves and selected-checkpoint test
scores, with the CPU reference clearly separate. No test-driven tuning or extension.
Close README, plots/tables and canonical experiment_log.md after all terminal runs.
PR131 scientific milestones remain separate. This is not a causal flank-only ablation:
model representation, fusion width and register pooling differ among versions.
