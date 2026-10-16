# Scoped sampling and probability-quality investigation (#94)

Registered prospectively by Codex (GPT-6), before either new fit. Compare exactly
two conditions on the previously frozen PXD013455 10,000-peptide cohort: balanced
batches enabled versus disabled. All effective regularization weights are zero
in both conditions, including binding orthogonality. Keep existing seed42,
data_seed42, d32/layer1/head4 expanded topology, CPU one Torch/BLAS thread,
batch64, lr0.001, weight decay0.01, uncertainty weighting enabled, two epochs,
100 training batches per epoch, full 1,000-row validation/test. No generated
negatives, MHC, auxiliary measurements or live assay inputs. The source pin,
8,000/1,000/1,000 membership and support fingerprint must match the first study.

Question: how much of the original probability-quality failure is associated
with sampling-induced exposure changes? The original run also had an active
orthogonality prior; it is historical context, not the sole balanced control.
Measure actual positive/negative proxy exposure and finite output gradients.
Compute unweighted validation log loss/Brier, AUROC/AP and threshold0.5 metrics,
with train-prevalence and train-only length/composition logistic baselines.
Select the preferred condition only by validation log loss; do not choose new
hyperparameters or calibration transformations after examining results. Retain
both conditions regardless of outcome. The old test has already been inspected;
all repeated test results are exploratory, not an untouched confirmatory test.
No independent-search/donor/specimen generalization claim. Shared proteins and
one search remain explicit limitations. No other output is supported.

Run sequentially in one bounded Modal CPU function: eight CPUs, 32GiB RAM,
512GiB ephemeral disk, no GPU, 60-minute function/55-minute work alarm. Canonical
preflight runs again before fitting and must pass the exact fingerprint and
traceability gates. Both conditions use identical source/partition pins. Freeze
the wheel, Git commit, source/config/manifest hashes and complete dependency
versions/install reports. Use datacache for shared inputs, membership and all
result archives. Preserve failure logs, checkpoints, complete val/test predictions,
lineage, per-step/epoch update evidence, regularization receipts, baseline state,
comparison tables and paired bootstrap intervals. Archive every original file,
publish all hashes and independently download them. No silent historical edits.

Pin pandas2.3.3/pyarrow25.0.1/numpy2.5.3/scipy1.18.1/scikit-learn1.9.1 and released Hitlist1.75.0/datacache1.27.1 to match the original cohort serialization and baseline contract. Torch2.14.1 CPU and mhcseqs2.6.13 are explicit. This separate experiment environment does not change the shared environment or the frozen human-cohort producer.
