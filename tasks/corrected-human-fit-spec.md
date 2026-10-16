# Conditional next PR: bounded corrected-human IC50 fitting diagnostic

Agent/model: Codex (GPT-6). This plan precedes the corrected-human preflight's
counts and any new predictions. Proceed only if its unchanged prospective
candidate-support minima pass: direct binding:IC50 from IEDB/CEDAR, at least 50
observations, 50 peptides and 50 exact values in every split. Otherwise record the
support failure without relaxing thresholds. This is a narrow diagnostic under
#48/#53, not a claim that all outputs or biological independence are qualified.

Freeze a new experiment directory and source commit before launch. Use released
Presto 0.1.13 and Hitlist 1.78.0, the corrected native manifest, the exact verified
MHC catalog and the new preflight's canonical source/model/membership fingerprints.
The canonical prospective supported-output manifest must pass unchanged before
model construction. Confirm identical root assignments and model membership.
If replay changes a fingerprint, stop and investigate; never rebind silently.

Use the same d32/l1/h4 expanded sequence-only model, full profile, all observed
families and both MHC classes. Claim only the directly measured IC50 assay output.
Other endpoints may contribute canonical losses but remain unvalidated. Keep all
synthetic ratios, augmentation and explicit regularization weights zero, no flanks,
strict MHC admission, data/split/model seeds 42, no pretraining or old weights.
Fix shuffled sampling, batch 64, learning rate 0.001, weight decay 0.01, canonical
uncertainty weighting, two epochs with at most 100 training batches per epoch.
Use complete validation for canonical aggregate-loss checkpoint selection, then
reload the selected checkpoint in a fresh model and evaluate every validation/test
row and complete bag. Record that selection is aggregate loss, not IC50-only loss.

Before optimization, save the initialized model and obtain IC50 predictions on a
fixed, explicitly identified training-only diagnostic set. Preserve training RNG
states around diagnostics and do not inspect test predictions during selection.
After fitting, evaluate the same training observations with the selected checkpoint.
Enable canonical per-output update tracking, preserve finite/nonzero gradient and
update counts, effective losses and all checkpoints/curves. Diagnostic fitting
passes only with finite optimization and lower exact training IC50 RMSE than
initialization; this does not establish held-out quality.

Fit a training-exact-value mean baseline and a fixed ridge baseline (alpha 1) using
peptide length, amino-acid composition and MHC class. Standardization is fit on
training only. Publish all train/validation/test predictions with source and
observation IDs, original/resolved MHC, cell/context/assay fields, raw value/unit,
log10(nM) target, qualifier/bounds, prediction and residual/censor satisfaction.
Preserve possible repeated peptide observations and bootstrap by peptide rather
than treating repeated observations as independent.

Primary held-out comparison: exact IC50 RMSE in log10(nM). Also report Pearson,
Spearman, qualifier-aware loss/censor satisfaction and classification at <=500 nM:
accuracy, balanced accuracy, precision, recall, F1, AUROC and tied-score-correct AP.
Only exact values and bounds that determine the threshold class enter binary
metrics; report excluded ambiguous bounds and undefined metrics explicitly. Use
1,000 paired peptide bootstrap resamples, seed 4242, for RMSE differences.

The bounded model clears the prospective quality screen only if validation IC50
Spearman is positive and the upper 95% bootstrap bound on model-minus-training-mean
RMSE is below zero. Report the ridge comparison regardless; prefer the model only
if it improves on the ridge baseline under the same validation rule. Test is a
single final descriptive evaluation, not a tuning criterion. A failure remains a
published failure; additional optimization or architectures require a separate
prospective family and retain test-reuse limitations. No broad biological
independence claim is available from unknown donor/specimen/acquisition metadata.

CPU-only small-model diagnostic: eight CPUs, 64 GiB RAM, 512 GiB scratch, one
Torch/BLAS thread, zero loader workers, no GPU. Set a 90-minute work alarm and a
95-minute function timeout covering source replay, fitting and complete evaluation.
Save interrupted/failed receipts unchanged. A larger quality-training budget is
separate; do not infer that this bound is adequate for a production model.

Close the family with a downloadable datacache archive, all hashes, verified
native membership, label policies, coverage, leakage, original logs, dependency
versions, source/config snapshots, checkpoint identities, initial/trained training
examples, full validation/test dumps, baselines and uncertainty. Independently
verify public downloads and recompute headline metrics. Separate versioned PR,
format/lint/full checks, review, merge and verified clean-main distribution.

Launch implementation details fixed before predictions: MIL instance cap is zero
(complete bags for training and validation), max_batches=100, max_val_batches=0,
AMP/compile/PCGrad off, warmup-cosine canonical schedule, no probe or motif scans.
Use all 2,074 direct-IC50 training observations for the initialization/trained
diagnostic. Implement alpha-1 ridge with NumPy's centered standardized closed
form (intercept unpenalized), avoiding a new runtime dependency. All 23 feature
columns (length, 20 amino-acid fractions, I/II indicators) are fixed. The exact
public candidate manifest SHA-256 is
13fe8bd4dab158bd06ae31fc7996fba98f73b3944fa68c3f57463beec34e7cb0.
The frozen preflight minima passed; no predictions have been generated.
