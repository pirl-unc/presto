# Macrophage presentation response with an independent proteome surrogate

Prospective after0410, before admitting this cohort or viewing predictive outcomes.
The previous turn is progress: ten fixed native-response fits, once-only held-out
comparison, independent closure and publication atf7b76a3. Preserve the full goal
of APC peptide availability modulo precursor input. Native ratios are not that
endpoint; this experiment adds actual APC measurements and independent digest data.

## Question and fixed data contract

In PXD057588 Table9, do measured parent-proteome changes and an explicit processing
model predict macrophage MHC-I peptide response better than condition alone?
Table11 is independent global digest-peptide MS under the corresponding coculture
conditions. It supplies a condition-level precursor surrogate, not matched aliquots
or measured free-peptide concentrations. The study includes mouse macrophages and
tumor partners; Table10 MHC loss belongs to the partner and must not be transferred
to the macrophage. Tables7/8/10/12 remain source context, not extra training labels.

Use the canonical Hitlist EvidenceStore/row-recipe interface against the existing
verified mouse-reference generation. Pin manifest and exact Hitlist code. Materialize
the source-reported log2 fold-change representation, including all source values,
derivation/control identities, source locators, measured/partner populations,
normalization, sequence/component/reference and specimen-link precision. Preserve
the entire all-table mapping/component graph. Do not parse workbooks into a parallel
curator or re-run reference searches. Retain all candidates/dispositions through
the common query/materialization path. Missing query fields/joins become concrete
Hitlist issues; add canonical support rather than guess or silently omit them.

Primary cohort: Table9 MHC-bound macrophage fold changes with canonical sequence,
full-parent occurrence support, and Table11 precursor support for every distinct
candidate parent sequence. Keep every exact overlapping origin; collapse identical
parent sequences, not measurement identities. A parent estimate requires at least
two distinct digest sequences. For each parent/condition, first average each digest
sequence's reported log2 fold changes over its treatment channels (no channel-index
pairing across assays), then take the median across distinct digest sequences.
Convert source log2 ratios to log10 by multiplying log10(2); retain original values.
Shared digest peptides remain explicitly ambiguous contributors to candidate-parent
estimates; they are not independent measurements of each protein. Record contributor
IDs/counts and alternatives. These fixed input estimates do not use MHC outcomes.

Condition matching uses source-backed GL261 versus CT2A coculture group labels,
never same-named TMT channels. Confirm exact source labels before encoding the two
categories; do not infer specimen/replicate correspondence from suffix numbers.
No selected/significant panels, target clipping, imputation, unreported negatives,
affinity filter, length cutoff, biological KO inference or phenotype relabeling.
Complete-case source selection and the precursor-supported population remain explicit.

Split existing all-source parent components by sha256("pxd057588_apc_response_v1:"+
component_id) modulo20:0..13train,14..16validation,17..19test. Preserve all MHC channels
and digest dependencies of a component together. Require at least50/20/20 admitted
peptide rows in train/validation/test; fail the fixed screen without reseeding or
relaxing precursor support. Weights average condition, component, peptide row and
source treatment channel within each split. Counts/support may be inspected before
fitting; held-out target summaries/predictions remain unopened until all fits finish.

## Parsimonious model and controls

Use0410independent-cut continuous-input/common-removal model, with removal1 and all
contiguous1..Lproducts mathematically supported. Inputs are complete parent sequences,
query peptide spans, condition and optional fixed parent precursor log-ratio. Control
amount sums all origin F terms with equal unit input per distinct candidate parent.
Treated amount sums 10^precursor_log_ratio * F_treated per parent. Output is a shared
condition offset plus log10(treated/control amount). No per-peptide free offset.

The observed target combines processing, MHC selection and measurement. Same-peptide
affinity/stability and detector cancellation are hypotheses under paired conditions,
not demonstrated invariants: changed MHC allele mixture, competition and sample
handling can violate them. Do not call this a separately identified cleavage or
free-concentration fit. The model retains all short/nonbinding products in its latent
support even though observed supervision is MHC-bound.

Fixed comparisons on precisely the same cohort/weights:
1. No change.
2. Condition-only linear offset.
3. Precursor-only: average equal-parent fold ratios, with no fitted parameters.
4. Condition offset plus precursor-only.
5. Uniform-bond processing plus precursor, exact gradients.
6. Free adjacent-residue processing plus precursor, exact gradients.
7. Same free adjacent-residue processing plus precursor, sampled kinetic gradients.
8. Free adjacent-residue processing without precursor, exact gradients (input ablation).
9. Frozen0410human native sequence-rate prior plus precursor, fit only the two
   condition rate scales and offsets; no refitting imported rho/sequence rates.

Controls use training-only condition contrasts/global intercept, penalty.01 except
offset globalintercept, deterministic L-BFGS-B with maxiter1000/gtol1e-9/ftol1e-12.
Physical arms use200full-batch float64Adamupdates,lr.02,betas.9/.999,eps1e-8, seed42,
no clipping/schedule/restarts/validation selection. Initialize free rates uniformly
at1 and condition coefficients at0. Adjacent rates use the same40left/right indicator
features, train-only equal-parent reachable-bond weighting, centering/scaling and
SVDrelative1e-10. Allocate only supported coordinates; report held-out directions
outside support. Penalty.01 except offset globalintercept. Frozen-prior raw feature
coefficients are algebraically recovered from0410sequence_exact and never selected
using APC labels. Source chemical conditions are not transferred.

Sampled arm: K512conditional origin/age samples per peptide/condition group using
the exact origin mixture including fixed precursor input, shared treated/control
uniforms, exact forward amounts/residuals/offset gradients. It is the same conditional
score hybrid as0410, not simulator-only RL or GRPO. Before fitting, check four fixed
states (uniform/free, zero/predetermined NumPy703normal*.03 state) with16 independent
groups. Full unregularized gradient meanrelative<=.10 and RMS<=.25; block errors
also recorded. When input is absent/identical, shared rate cancellation must hold;
with differing parent input mixtures it need not. Finite differences at the same
states must agree within1e-7. No gate tuning; exact arms proceed if sampling fails.

## Execution, validation and stop rules

Register one experiment before cohort arithmetic. Freeze exact query requests,
source hashes, feature/precursor recipes, code/config and software/environment.
Reuse qualified physical primitives and metric helpers; do not build another broad
source audit. Independently verify source-to-export quantities and original contributor
membership for the chosen endpoint, then fit. Local CPU only, one requested numerical
thread, no shared environment changes. Caps300s/exactarm,1800s/samplearm,4GiB.
Project sampled200stepcost after its first20real steps; preserve partial run on failure.

All arms/settings are frozen before held-out evaluation. Fit all requested controls
and admitted physical arms, seal final checkpoints, then evaluate validation and
test once for every completed arm. No test-driven retry. Preserve all source-linked
train/validation/test predictions, source targets/weights, input precursor estimates,
checkpoint trajectories, parameter support, gradient probes, sample counts and time.
Primary metric: weighted MSE in log10 ratio. Also RMSE/MAE/R2/Pearson/Spearman, response
sign, prediction spread, parent/component/condition/origin-ambiguity strata. No nM
or500nMbinding metrics apply. No concentration promotion from this endpoint alone.

Independent closure reconstructs final physical predictions with separate scalar
amount arithmetic and all source values/weights/metrics, without optimizer replay.
Compare precursor/context controls before attributing gain to processing. A useful
result must improve both condition-only and condition+precursor held-out MSE.
If no processing arm does, record the negative result and stop tuning this test.
The next step must address independently measured machinery/selection/observation
or an external biological population. Keep the full APC availability objective open.
