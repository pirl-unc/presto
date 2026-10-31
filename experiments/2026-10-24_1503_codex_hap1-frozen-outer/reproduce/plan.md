# Once-only HAP1 outer evaluation after the length comparison closes

October24,2026, Codex / GPT-6. Prospective evaluation plan; no outer predictions
or scores have been produced. Metadata-only inspection confirms complete-precursor
counts87,260train/19,744validation/22,724test, including ERAP1
8,281/1,970/2,297. The purpose is to distinguish genuine held-out prediction from
repeated inner-CV development. This is not APC concentration validation.

## Preconditions and freeze

1. Finish and independently close the registered1007length comparison, including
   its preserved numerical failure/recovery. No additional fitting or tuning on
   the HAP1 inner folds after that comparison, regardless of its outcome.
2. Create a separate timestamped experiment and pin canonical source88cad10d… /
   recipea1eec392…, baseline0730 and physical0810/1007 artifact hashes. Freeze
   full-training checkpoints, transforms, offsets, parameters, code and this plan
   BEFORE loading held-out response values for modeling.
3. Require source/recipe fidelity, train-only transform replay, scalar recurrence,
   and predictor tests showing that evaluation targets cannot change predictions.
   Numeric/source failures are not grounds to select another model. Preserve them.

## Fixed model matrix; no optimizer

For all ten KOs, evaluate the seven archived0730full-training direct/control
models: zero, condition-only, fixed precursor, calibrated precursor, and sequence
penalties.001/.01/.1. The already chosen sequence_0.1 is the primary model; the
already chosen calibrated-precursor control is its primary comparator. Show all
variants without choosing a new winner from outer outcomes.

On ERAP1 only, additionally evaluate every archived0810 and qualified1007
full-training physical model, exact and sampled-trained. Inference is deterministic
exact forward computation for both parameter sets. Keep all source origins and
parent-input alternatives; regenerate only the query paths needed for held-out
peptides. Never fit a new offset on held-out rows. New1007augmented_exact versus
sequence_0.1 is the primary physical comparison; all other pairs are descriptive.

No fresh pretraining, calibration, optimizer updates, new seeds, model selection,
early stopping, threshold/feature tuning, new biological labels or source curation.
No new RL sampler, clipping/std/reuse/KL experiment is implied by this evaluation.

## Data and metrics

Use unchanged positive-response/B2M/complete-precursor eligibility, exact molecular
component splits and original qualifiers/missingness. Retain lineage and every
replicate cell. Assay-to-output is log10 KO/WT MHC-elution intensity response;
all other assay families and nM/binder metrics are inapplicable. Species and
genotype remain constant context. HAP1 is not a professional APC. Shared controls,
study/batch and unresolved clone/specimen/homology independence remain limitations.

Compute hierarchical weights independently within each split: equal condition,
component within condition, peptide within component, replicate within peptide.
For the ERAP1 comparison renormalize its weights. Save every prediction and enough
source identifiers to recompute pooled and condition/component/length-stratified
MSE/RMSE/MAE/R2, Pearson/Spearman and prediction/target spread. Constant correlations
are undefined. Preserve validation and test results separately; evaluate once
after all model choices are sealed, without acting on validation first.

Primary practical gate for sequence_0.1: >=2% lower weighted testRMSE than the
frozen calibrated precursor control, with within-condition prediction SD>.001 for
all ten conditions. Primary ERAP1physical gate: >=2% lower testRMSE than the frozen
sequence baseline and correction SD>.001. Report validation gates too, but neither
changes settings. Passing is evidence for this observed-positive elution response
contract only, not unseen KOs, independent cohorts or intracellular amounts.

## Execution and closure

Local CPU/float64/one numerical thread, no cloud or environment changes. Caps:
preparation900s, total inference/metrics1800s, independent closure900s, peak4GiB.
Store all input/checkpoint/source hashes, frozen invocation, per-example validation
and test dumps, scalar verification and comparison tables/plots. Independently
recompute predictors and source grouping/weights/metrics without fitting.
Once scored, label these outer sets inspected and prohibit tuning against them.
Update canonical log and model_to_beat only with scoped evidence. The full APC
availability goal still requires native/independent processing, selection/detection
constraints, precursor-relative calibration and validation in APC material.
