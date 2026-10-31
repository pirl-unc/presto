# MHC reference species evaluation — prospective contract

Registered 2026-10-26 before fitting or inspecting predictive results, Codex / GPT-6.
This extends the admitted `2026-10-21_0618_codex_mhc-reference-admission` cohort.
It tests source-annotated single-domain species classification. It cannot establish
class-II pairing, peptide affinity, homology independence, or useful transfer.

## Data and output contract

Use only manifest SHA256
`4e4d4a5d2b3d47fc16ddf63ddeeaf6111836138661356ab6805945f13c7ef06c`,
with observations SHA256
`5ba2ef6f40b047ddc256c61fb06e802c82bc2a400830a178d44a0a82455ef510`,
under the original modular worktree's `data/modular_cohorts/2026-10-21-mhc-reference/`.
Retain all 13,218 actual annotations, original aliases, fixed reference-chain/domain
components, and 6,092/694/6,432 train/validation/test assignments. No resplitting,
rebalancing, alias expansion, synthetic labels, assays, censoring or peptide data.
Species order is human, mouse, rat, rhesus_macaque. An annotation has exactly one
nonempty groove and one corresponding mhc1/mhc2 label; both slots train the shared
mhc_species readout. Class and pairing labels remain absent.

The known 5,664-row human test component dominates observation-weighted results.
Report those results alongside equal-component metrics and per-species/slot
results. Exact-source grouping does not establish homology-independent generalization.
No new holdout, independent clinical test or calibrated biological identification
claim is made. Source producer version remains unknown.

## Fixed conditions and selection

1. Training-prior control: add one pseudocount to each of the four training class
   counts and normalize. No test/validation fitting.
2. Composition control: length plus fractions of the 20 canonical amino acids,
   and a binary groove2 indicator. Standardize using training mean/population SD
   only (zero SD becomes one). Multinomial softmax with summed training negative
   log-likelihood plus 0.5 times squared coefficient norm; intercept unpenalized.
   Fixed L-BFGS, maximum 2,000 iterations, gradient tolerance 1e-7; verify an
   independent analytic gradient and require max absolute gradient <=1e-3.
3. Canonical modular identity-only model, fresh seeds 42, 43 and 44. Fixed shared
   sequence encoder: d128, two layers, four heads, dropout0.1, max length512;
   species vocabulary above, identity task mhc_species, no assay endpoints.
   AdamW learning rate3e-4, weight decay1e-4, batch64, identity weight1, max50
   complete epochs, min10, patience10, no step cap. Select minimum canonical
   observation-weighted validation CE. Preserve every epoch and all training
   exposures; evaluate each selected checkpoint on test exactly once. No best-seed
   selection, architecture search, holdout tuning or transfer claim.

The baseline control is bounded CPU work and may run while the independent IC50
fit uses its frozen resource allocation. Neural conditions must wait until all
IC50 conditions finish or stop, and then run sequentially. Do not alter that
family's source, manifests or controller. A failed preceding execution requires
review before another neural launch; a failed scientific screen is a recorded
result, not a reason to repeat or tune a condition.

## Evaluation and engineering screens

Retain every train/validation/test prediction with normalized source observation,
label, class order, slot, group and probabilities; retain native neural dumps.
Report n, components, cross-entropy, accuracy, balanced accuracy, per-class
precision/recall/F1/AUROC/AUPRC, macro F1, confusion matrix, multiclass Brier,
10-bin top-label ECE, probability SD, and equal-component mean CE/accuracy.
Report per-slot metrics with absent classes explicit; no regression/500 nM
metrics apply because the task has no affinity labels.

Each neural seed must pass software/source/coverage verification, beat the prior
training CE, achieve validation balanced accuracy >=0.8 and recall >=0.5 for each
species, and have an upper95% paired-bootstrap component-mean CE difference <0
against both fixed controls. Use 2,000 bootstrap resamples, seed20261026, sampling
reference components uniformly and preserving each component's mean loss.
Minimum validation support is 10 components and20 annotations per species;
this is an engineering admission minimum, not a statistical power claim.
All three seeds must pass for this narrow engineering screen to pass. Report
between-seed variation and test metrics without using test to change the recipe.
The bootstrap describes the frozen reference-component sample, with unverified
homology independence and small nonhuman groups; it is not a biological
population confidence statement. Do not call this general model qualification.

## Execution, verification and closure

Family `experiments/2026-10-26_1720_codex_mhc-species-evaluation/`.
Raw artifacts in the original root's matching `artifacts/` directory. Local CPU,
existing Python3.12.6 environment and read-only pinned Hitlist overlay; no install
or cloud/GPU. BLAS/Torch four threads for neural, one for controls, interop one.
At launch require >=10GiB available RAM and15GiB free disk; live floors5GiB RAM
and10GiB disk, process RSS ceiling6GiB. Controls/preparation/verification each
limit10minutes; each neural condition limit3hours, polled every0.25seconds.
Stop and preserve failures; no overwrite or automatic retry. Freeze code, full
runtime hashes, manifests, invocation/environment, git commit/dirty state and
package/hardware versions before execution. Record wall time and peak RSS.

Independent verification must join all predictions to the original immutable
source, reproduce features/scaling/control coefficients and metrics without
importing the reporting functions, verify component metrics, and replay the
selected checkpoint on the first64 training examples in a separate process.
Never rescore test for verification. Check all epochs are complete, all expected
training labels are exposed and the selected coverage gate passes. Preserve
hashes of checkpoints, prediction files and reports. Close README, summary JSON,
comparison CSV and canonical experiment log after terminal results. Keep partial
results and an explicit uncompleted status while queued/running. No merge follows
from this isolated stage.
