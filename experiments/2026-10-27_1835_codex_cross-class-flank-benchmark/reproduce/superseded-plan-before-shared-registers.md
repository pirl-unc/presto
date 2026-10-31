# Cross-class transfer and flank comparison

Codex / GPT-6. User requested actual performance benchmarks, live side-by-side
training/evaluation, and especially I-only to II and II-only to I transfer.
Base production implementation: f24f7dfd6404c6eb450545daf72769269f55a5bf.
Historical three-residue implementation: 9dd71348124faa49a4c6a60872385c86895d50b0.

## Questions and frozen first-pass scope

1. How much IC50 prediction transfers between MHC classes without any target-class
   binding training labels?
2. How does the new full-context model compare with its previous three-residue
   composition implementation under the same update budget?
3. What are their CPU time, throughput and memory costs?

This first pass is a fixed-budget development benchmark, not a convergence or
independent-study qualification. Run seeds 42 and 43 for four conditions:
full-context I-only, full-context II-only, full-context joint, and historical
three-residue joint. Each has exactly 1,000 optimizer steps, batch64, AdamW3e-4,
weight decay1e-4, dropout.1, identity weight0, no pretraining/synthetic labels.
Use max_epochs100/min_epochs100/patience100 so the step budget controls stopping.
Validation-selected checkpoint is primary, selected on the TRAINED classes only.
The target-class validation labels never determine selection or training settings.
Do not extend, choose seeds, or tune architecture based on test outcomes.

Eight serial fits total. Fit all admitted training rows from the selected classes,
without a row cap; different numbers of epochs/exposures follow from equal updates.
Report counts/steps/exposures and unequal class prevalence explicitly. The two
historical/new model versions have different fusion widths; the comparison measures
the complete flank-representation change, not a parameter-count-controlled causal
ablation. The production architecture and training entry points remain unchanged.

## Data and leakage contract

Reuse the hash-pinned 2026-10-21-full-ic50 cohort and original assignments from the
2026-10-26_1610 convergence family. Source manifest SHA256:
914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be.
Train:21,636 exact observations (I14,375; II7,261). Validation:2,925 (I2,051;
II874). Test:2,803 (I1,845; II958). Preserve exact/bounded holdout targets and units,
resolved explicit MHC pairs, canonical peptides, provenance and all identifiers.
Only direct purified IC50 maps to log10(nM), train-only standardization and squared
standardized error; inequalities use the existing one-sided loss. No proxy KD,
MS, TCR or auxiliary labels. No synthetic biological data. Existing holdouts were
historically inspected and share studies; call them development holdouts.

Before fitting, verify source bytes, full row conservation, exact peptide and
measurement-root separation across all classes/splits; stop on leakage or source
mismatch. Derived class manifests are lossless row filters with source hashes and
per-row IDs preserved. Never convert a source-class training row into a target
holdout. Record length distributions and shared-study limitations.

## Zero-shot protocol and architecture limitations

Current serving correctly refuses untrained class branches/features. I-only
training cannot learn class-II register attention; class-one-hot and flank columns
can also be untrained. Do not suppress these guards or claim qualified serving.

After a source-only fit, construct an explicitly diagnostic view using the SAME
Presto class, expanded class routing, copied source checkpoint weights and no
optimizer updates. The primary transfer protocol neutralizes never-exposed fusion
columns to zero and initializes an absent register scorer to zero (uniform pooling).
Every trained tensor/column must match the source checkpoint exactly. Record the
new/untrained/neutralized parameters. This tests transfer of learned sequence/core
weights without letting arbitrary untouched weights drive the unseen class.
Also retain a raw cold-start control with deterministic initialization of missing
components; report it separately. Neither view changes the saved trained model,
its support metadata or the normal prediction API. Target-class scores are marked
out-of-support research diagnostics, never ordinary qualified predictions.

Use ordinary supported evaluation for in-class outputs. Save cross-class
validation/test observations and raw/transformed predictions at the selected source
checkpoint. No target-class fine-tuning in this first pass. If diagnostic transfer
cannot preserve all trained parameters, stop and revise before observing scores.

## Metrics and live view

Live local dashboard: columns evaluation MHC-I / MHC-II; rows train I-only,
II-only, joint plus three-residue joint comparator. Display seed, steps/epochs,
training loss, source validation Spearman/RMSE, completion status, throughput,
elapsed time and peakRSS. Plot learning curves against optimizer steps and wall
clock. Show historical fully converged results in a separate reference section;
do not rank them as matched-budget runs. Refresh from atomic JSON/history receipts.
Target-class validation and test cells populate only after source-only selection.
Test evaluation occurs once per prespecified selected model/control; do not stream
test metrics each epoch. Preserve displayed history and downloadable tables/plots.

For each model/class/split: all-label loss, exact n/Spearman/Pearson/log10(nM)RMSE;
500nM accuracy/balanced accuracy/precision/recall/F1/AUROC/AUPRC, censor satisfaction,
prediction spread and length strata. Compare paired seed results; 2,000 fixed-seed
peptide-cluster bootstrap draws for RMSE/Spearman differences. Do not claim
population uncertainty from two seeds. A transfer advantage must outperform the
source training-mean baseline; compare with matched-budget target-trained models.
A full-flank win needs consistent paired improvement; mixed results stay mixed.

## Local resources and execution

Honor the user's existing local-compute preference: CPU float32, four numerical
threads and one interop, no autocast/compile, no cloud/GPU or package changes.
Profile the same frozen training-only minibatches before fitting (3 warmups,
5 timed updates per model version); no validation/test scoring during profiling.
Record parameter counts, requested/actual hardware, wall time and peakRSS.

This new bounded benchmark has its own prospective limits: startup availableRAM
6GiB/free disk15GiB; running availableRAM3GiB/free disk10GiB; processRSS4GiB;
1hour per fit. One fitting process at a time. The dashboard is read-only. A resource
failure preserves artifacts and stops the serial controller; no automatic retries
or lowered floors. These limits do not change any older experiment contract.

All fits use their frozen version's existing manifest fit_index/train_step. The
historical version is a frozen comparison only, not a production-selectable backend.
Family-local code handles data filtering, orchestration, diagnostic evaluation and
reporting; do not introduce another optimizer/training loop or task-specific model.

## Required closure

Register experiments/2026-10-27_1835_codex_cross-class-flank-benchmark before launch.
Freeze launcher, invocation, environment, runtime source hashes, manifests, split
census and limitations. Verify meaningful fixtures for cross-class parameter
preservation, no-target-label selection, metric recomputation, dashboard updates
and failure stopping. Retain per-example validation/test dumps, all conditions,
checkpoints/paths, metrics, plots and resource receipts. Independently recompute
metrics from dumps, replay saved training examples in a fresh process and verify
all source hashes again. Update family README and experiments/experiment_log.md.
The full biological qualification and PR131 scientific gates remain separate.
