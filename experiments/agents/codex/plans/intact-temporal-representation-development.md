# Separate sequence capacity from shared kinetics before another biological test

Status: prospective next development experiment; not launched. This follows0046
and0116. The full precursor-relative APC availability/concentration objective is
unchanged. Existing100/120-minute results stay inspected, and no revised model
will be scored there to generate a replacement blind-test claim.

## Why this comparison

0116 finds only2.3%training-MSE improvement from a stronger optimizer. Increasing
concave curve constraints account for about8%of original physicalMSE, while
per-peptide constrained curves fit much better. The next question is whether
sequence-supported temporal differences exist that the shared bond model misses,
as distinct from fitting peptide identities or freeing unsupported kinetic rates.

## Frozen development contract

Use only the unchanged0046training626cells/96peptides at0–80minutes, canonical
recipe and source hashes. No fresh source labels, model-selected exclusions or
new Hitlist curator. For each peptide use its first eligible time's replicate
mean as an observed baseline. Predict later log product/intact-ratio changes from
that baseline. Baseline measurements are allowed conditioning data, not predicted
future outcomes; retain their measurement IDs and reuse dependencies explicitly.
Keep every later original replicate observation, with equal peptide/time/replicate
weights renormalized within each training/validation fold. Missing stays missing.

Fix five peptide folds by SHA256(`intact-temporal-representation-v1|` + sequence)
modulo5. All repeated observations of a peptide stay together. Parent is shared
and overlapping subsequences exist, so this is diagnostic same-parent sequence
CV, not independent-protein/homology or study validation. Require>=10peptides per
validation fold; stop without changing salt/threshold if support fails. This is
already-inspected training data; label it development regardless of CV partition.

For each fold compare:
1. Zero change/persistence.
2. Shared linear elapsed-time slope, fitted only on training curves.
3. Length-conditioned slope (log length plus intercept).
4. Sequence-conditioned slope (length,20fractions,20N/20C indicators plus intercept).
5. Original uniform independent-bond ratio changes.
6. Original residue-before-bond model with penalty.01.
7. The same residue model with penalty0, explicitly testing regularization.

Linear predictions are slope(features) times elapsed time/80. Transform features
with fold-training means/scales, remove unsupported directions by a fixed SVD
threshold1e-10relative to max(largest,1), and preserve transforms/omitted directions.
No peptide identity, previous slope, target-derived composition or held-out curve
offset is a feature. Ridge alpha.001/.01/.1/1 is selected in three deterministic
inner peptide folds within each outer training fold, assigned by
SHA256(`intact-temporal-inner-v1|` + sequence) modulo3; require>=5inner-validation
peptides, otherwise stop rather than adapt selection. Keep the intercept unpenalized.

Physical change equals f(sequence/span,t)-f(sequence/span,baseline_time), so fixed
MS offsets cancel. Optimize only shared rates/pre-exposure on training changes,
with the original bounded parameterization and the fixed0116L-BFGSsettings.
Use three data-free initializations: (allrates.001/min,tau10),
(allrates1e-5/min,tau60), and(allrates1e-4/min,tau30). Do not initialize from the
0046Adamfit, which saw every original-training peptide and would leak information
across these development folds. Choose the lowest training objective, never
validation; report all starts. This correction predates registration/execution.
No sampled arm:
0046established matched exact/conditional-score agreement without predictive gain,
and this phase tests representation, not gradient estimation.

## Qualification, evidence and stop rules

Before fitting, verify baseline provenance, no target/curve leakage across folds,
training-only feature support, zero-change cancellation, exact/raw feature mapping,
independent ridge predictions and physical finite differences. Preserve every
fold condition, inner selection, coefficient/parameter/optimizer history and every
per-example development prediction. Metrics are weighted MSE/RMSE/MAE/R² and pooled
Pearson/Spearman, plus peptide/fold strata and prediction spread. No IC50 metrics.
Use paired peptide bootstrap1000resamples/seed31415 only as descriptive development
uncertainty, not independent-subject confidence. No retuning after outcomes.

Sequence-conditioned slopes need>=5%pooled CVRMSE gain over the selected length
model, positive pooled change correlation, and nonconstant output. An unpenalized
physical model needs>=5%gain over its penalized counterpart to justify changing
regularization. Report every result even if neither gate passes. Training-fit
gains alone do not qualify a model. If sequence fails, stop expanding this single-
substrate representation and return to broader measured substrates/contexts.

## Fresh biological evaluation prerequisite

The pre-execution historical audit finds explicit disclosure in1123README:37 and
1138README:33 that original PXD044463 validation/test descriptive repeatability
was inspected by upstream review. The2358model's exclusion of test parents does
not erase that exposure. Do not describe this partition as a completely untouched
biological test. Qualify another prospective source through Hitlist for that claim;
any reuse of the old partition must be explicitly exploratory. Freeze compatible target,
initial-state/observer treatment and full candidate support before outcome access.
Neither same-peptide CV nor relative MS intensities establish unobserved peptide
mass or APC concentrations. A generative processing model must eventually constrain
those quantities with appropriate precursor, initial-state and observation evidence.
Any qualifying generative candidate must then receive a matched biological
exact-versus-discrete comparison with the same data, observation model, objective
and parameter support. Preserve the existing exact and REINFORCE references;
separate simulator-forward peptide-set learning from exact-forward score-gradient
hybrids explicitly. This diagnostic's omission of sampling does not remove that
comparison or the full cellular-quantity requirement from the active goal.

## Execution and closure

One registered timestamped family with exact git/env/input hashes, source snapshots,
all validation-prediction dumps and report/registry closure. No outer biological
test is used in this development phase; say so explicitly. Localfloat64CPU/one
thread,3GiBRSS,20minutes/stage,32MiBnew artifacts,1GiBfree disk+32MiBstartup margin.
Recheck shared free space before launch; no relaxation, cloud or environment changes.

## Numerical specification before code/execution

Baseline is the unweighted mean of original replicate targets at the first
eligible time (replicate cells have equal weights within peptide/time). Save every
baseline assignment ID, count and source value; later observations are never
averaged as targets. Normalize equal peptide/time/replicate weights separately
for each fit/evaluation split. Inner-CV selection aggregates squared error equally
over held-out peptides, not equally over folds of unequal sizes; fixed ascending
alpha order breaks exact ties. Preserve all inner candidates and predictions.

Direct raw features are log length (both representations); sequence additionally
uses20amino-acid fractions and20one-hot indicators per terminus. Fit centers/scales
from fold-training peptides with equal peptide weights, project out constant and
dependent directions using covariance eigenvalues>max(largest,1)*1e-10. No target
enters the transform. Keep an unpenalized shared-slope intercept; minimize
weightedchangeMSE +alpha*mean(nonintercept_coefficients²). If no supported feature
direction remains, use the shared slope with an explicit zero-feature record.

Use L-BFGS-Bmaxiter1500/maxfun3000/maxls40,gtol1e-9/ftol1e-13,rawbounds[-30,30]
for every physical initialization. Only original-fold training curves enter its
loss or convergence decisions. Store all evaluations/accepted iterates, final
predictions, offsets (baseline-only, no profiled future offset), and gradients.
Positive correlation/nonconstant gates use pooled outer-development predictions;
nonconstant means predictionSD>=0.1targetSD. The same weighting definitions apply
to controls and physical arms. No stopping threshold changes after observations.
