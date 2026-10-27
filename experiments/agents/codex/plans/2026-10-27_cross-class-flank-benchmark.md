# Shared candidate scoring and prospective cross-class benchmark

Codex / GPT-6. Revised before any fit/profile/held-out scoring in response to the
user's request for a register mechanism that can be trained on MHC-I alone.
The superseded plan is retained in reproduce/. This spec replaces its diagnostic
neutralization design; no such models have been fitted.

## Production architecture (one current path)

Promote presto.shared.v4 in the current Presto class. The common sequence encoder
processes complete peptides, candidate cores, and both MHC grooves. For each
candidate, average its independently encoded core with the mean contextual token
states at its core positions in the full peptide (fixed 1/2 weights). This retains
core-relative position information and permits all flanks to influence the core.
There is no separate flank encoder, flank projection, class one-hot, or learned
context gate. MHC-I's single whole-peptide candidate reduces to the ordinary full
peptide encoding. MHC-II enumerates all contiguous nine-residue candidates. These
are geometry rules, not separate learned models. This does not yet model MHC-I
bulges or establish that flank effects can be learned without MHC-II observations.

Fuse this 128-wide candidate with the same two 128-wide MHC representations and
existing optional species/pairing features. Apply the existing endpoint head (or
cellular component head) to EVERY candidate. Pool those scalar scores by normalized
log-sum-exp with fixed temperature 1 in the model's target space: soft-min for
IC50/KD/EC50/proxy affinity/koff, soft-max for half-life/Tm and cellular/receptor
readouts. Corresponding softmax derivatives are reported as register weights.
One candidate returns its score exactly. Normalization avoids an automatic benefit
from repeating identical candidates. This is a differentiable statistical pooling
rule, NOT thermodynamic occupancy or a mechanistic equation for IC50/kinetics.
There is no independent register scorer. Each readout is trained by single-
candidate observations as well as multi-candidate observations. Class uncertainty
continues to mix class-specific scalar results with the resolved probabilities.
Existing multi-allele aggregation, endpoint masking, downstream MS/digest sums,
observation offsets and losses retain their contracts.

Keep mhc_classes as a supported-routing declaration and preserve admission's
source-class support guard. Always use the shared register rule; reject disabling
it in new config. Single-core II training no longer prunes a learned module or
rejects longer held-out II solely for register count. Research zero-shot evaluation
expands routing in a copy of a selected source-only model; strict-load identical
state_dict bytes, with no new/zeroed parameters and no target optimizer update.
The saved source model remains unchanged. Expanded-route results remain explicitly
out-of-class research diagnostics, not class-qualified serving claims.

Preserve v3 full-context checkpoints via an inference-only legacy module and its
frozen feature-layout helper, alongside existing v2 compatibility. Exact replay
and refusal of v3 warm-start/current training are required. All current entrypoints
continue to instantiate the same Presto and manifest trainer. No alternative
production backend or task-specific training loop.

## Verification before predictive fits

Test singleton identity, variable-length normalization, duplicate-score invariance,
correct pooling direction, monotone gradients to every candidate score, and exact
score/readout coupling. Test I-only updates reach every new core/readout parameter,
strict identical-weight route expansion, distal flank sensitivity through shared
encoder, padding/batch invariance, both class/endpoint gradient paths, context
isolation, checkpoints/freezing/resume, and v2/v3 inference replay. Run the relevant
modular/unified tests and repository lint. No scientific claims from fixtures.

## Fixed-budget experiment matrix and data

Two seeds (42,43), five conditions each: v4 shared-candidate I-only, II-only, joint;
v3 full-context joint; v2 three-residue joint. Ten serial fits, each exactly 1,000
updates, batch64, AdamW lr3e-4 wd1e-4, dropout .1, identity weight0, max/min epochs100
and patience100. No pretraining or synthetic biological data. Use all admitted
training rows, no row caps. Equal updates imply unequal epochs/class exposures.
The two older versions are frozen controls only. This compares entire architecture
changes, not a parameter-count-controlled flank-only ablation. No architecture or
budget changes after test inspection. Earlier converged runs are separate reference.

Reuse original, historically inspected development splits from full-ic50, source
manifest SHA256 914e0930283a9e5f7a41024576976f3e92e280f0719feda261ffcf48bf01d7be.
Training 21,636 exact labels (I14,375/II7,261); validation 2,925 (I2,051/II874);
test 2,803 (I1,845/II958). Direct purified human IC50 only, log10(nM), train-only
standardization, standardized squared error and existing one-sided bound loss.
No proxy KD/MS/TCR labels. Preserve source qualifier policy, provenance, explicit
MHC grooves, split and IDs; filtered manifests are lossless. Verify original hashes,
counts and peptide/measurement-root split disjointness before fitting. Shared studies
and previously inspected holdouts limit interpretation; this is not independent
scientific qualification. Record lengths, actual training examples and exposures.

Select checkpoints using only the source-trained classes' validation RMSE. Target-
class validation/test are evaluated only after source selection, without fitting or
calibration. Preserve source-training-mean controls on each evaluation class.
Report in-class and zero-shot rows separately, with exact-value n/Spearman/Pearson/
RMSE, all-label loss, <=500nM accuracy/balanced accuracy/precision/recall/F1/AUROC/
AUPRC and censor satisfaction. Save per-example validation/test predictions; report
length strata and prediction spread. Compare paired seeds and 2,000 fixed-seed
peptide-cluster bootstrap draws for differences. Two seeds do not establish
population uncertainty. Preserve negative or mixed transfer results.

## Live display, local resources and reproducibility

Open a local read-only dashboard before fitting. Columns MHC-I/II, rows condition
and seed, live training/source-validation curves against steps/wall time, runtime
and RSS. Target validation and final test cells fill only after selection. Historical
fully converged baselines appear separately. Serve only generated state/UI on
127.0.0.1; retain snapshots and final plots/tables.

Local CPU float32, four numerical threads/one interop, no AMP/compile/cloud/package
changes. Profile all three versions on identical training-only batches: 3 warmups,
5 timed steps. Record parameter counts and hardware. Startup available RAM6GiB,
disk15GiB; running RAM3GiB/disk10GiB; process RSS4GiB; one-hour limit per fit. One
fitting process, stop on resource/admission/fit failure without automatic retries or
relaxed floors. Freeze code/manifest hashes, exact invocation/env, source git commit
and dirty status before launch. Use each runtime's canonical fit_index/train_step;
family code only orchestrates, observes, evaluates and reports.

Independently recompute saved metrics, verify source hashes and replay predictions
in a fresh process. Preserve launch receipts, failures, checkpoints/raw paths,
val/test dumps and plots. Close README and experiments/experiment_log.md. This
benchmark does not replace PR131's outstanding scientific qualification gates.
