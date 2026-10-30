# Shared assay measurement layer: matched comparison

Agent/model: Codex / GPT-6. Registered 2026-10-29, before preparation or fits.
Status: implementation verification and preparation; no predictive results yet.

The canonical v6 measurement layer restores joint output descriptors on the
shared v5 biological representation. This first focal comparison changes IC50
method readouts only: purified competitive radioactivity versus fluorescence.
The same generic registry/decoder supports other compatible direct endpoints;
their expansion is not conflated with this contrast. Other nine endpoint heads
and every encoder/register path remain shared and identical across conditions.

## Frozen design

Six fresh fits: pooled, scalar calibration, factorized latent-dependent residual,
each seeds 42 and 43. Width192/layers3/heads6/FF768/dropout0.1; rank8 residual,
scale0.1, zero projection initialization; identical baseline parameters and RNG.
Preparation and mechanism have only one level in this focal registry. Their
individual effects are not identified by this comparison, regardless of gradient
updates; the empirical contrast is detector/method within purified competition.
No pretraining. AdamW3e-4 constant, decay1e-4, batch128, 4,000 updates, validation
every1,000, checkpoint500. Minimum equal-endpoint validation loss selects the
checkpoint. Same parent endpoint/class sampling weights in every condition:
60% ligands,20% literal IC50,15% other physical,5% qualitative. This is a focal
budget, not a complete pass through all source observations. No added ranking
or agreement loss. Exact scalar squared error and one-sided inequality errors
remain in train-only standardized target space; binary observations use BCE.

Parent: `2026-10-29_1956_codex_curated-data-retrain`, source-review-v2. Its 3,519,577
TRAIN observations are the starting pool, with source-reviewed explicit/inferred
bounds and 618 unresolved exact values withheld. Ten distinct outputs: IC50,
KD,EC50,KD(~IC50),KD(~EC50),half-life,koff,td50,qualitative binding,elution.
Source reported MS positives and parent-generated unobserved ligand comparisons
retain their original provenance and generation contract; no new synthetic data.
Cellular binding/stabilization/functional IC50 is not admitted as purified binding.
Known method citation conflicts use the quantity reference, not a claimed protocol.

The original inspected validation/test observations remain. Four additional
fluorescence study components are selected by metadata-only deterministic rule
in `reproduce/extra-holdouts.json`, before examining predictions. They were in
previous v5 TRAIN: these are retrospective development holdouts, not blind data.
Whole selected components move to validation/test; their peptides are embargoed
from all remaining TRAIN outputs. Original heldouts take precedence; additional
test takes precedence over additional validation for peptide overlaps. All
removed rows and counts must reconcile in `results/prepared.json` before launch.
No warm start can expose these added holdouts to the new fits.

## Evaluation and completion contract

Every run saves full validation/test predictions, native-unit metrics, exact
Spearman/Pearson/RMSE, bound satisfaction, threshold binding metrics and ligand
AUROC/AUPRC. Assay × class × study metrics and original versus added-holdout
populations are reported separately. An untrained assay track fails the coverage
gate. Qualification, source-tag accuracy and causal biology remain separate.
No winner is declared until matched validation/test results and uncertainty are
available. No model promotion from software tests alone.

Hardware: explicit H100!, at most two concurrent GPUs including the two active
parent jobs. A durable coordinator waits for both saved parent calls to terminate
before submitting pairs. Source/runtime hashes and git state are frozen before
upload. Actual GPU/memory, runtime and artifact receipts are recorded per run.
Large raw artifacts live under the named Modal `presto-checkpoints` volume in
this experiment's directory; exact paths/hashes accompany collection receipts.
The local dashboard and final closeout collector retain summaries here. Source
snapshot location and launch command are in `reproduce/launch.json`.

Post-launch reader audit: the main-tree historical v5 loader now retains saved
coordinate-update masks (or marks their absence unknown). The frozen comparison
starts fresh and loads only v6 checkpoints, so this v5-only reader correction
does not change its execution. A separate local queue monitor exposes terminal
coordinator errors on the dashboard; it does not alter source data or fitting.

Launch admission correction: Modal rejected the initial 131,072 MiB ephemeral-disk request before starting functions. The rejected bundle is preserved under attempts/disk-request-rejected. The corrected hardware request is 524,288 MiB ephemeral disk; model, data, GPU count and runtime are unchanged.
