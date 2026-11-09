# Bounded ERAP1 remaining-length comparison

Registered October 24, 2026, Codex / GPT-6, before implementation or new fitting.
Parent result: `2026-10-24_0810_codex_erap1-trimming`. The objective remains
precursor-relative peptide availability in APCs; this experiment is a development
test of one missing model dependence, not achievement of that objective.

## Question and explicit revision of the previous plan

Does remaining-substrate length improve the residue-only physical model and add
predictive information beyond the frozen direct sequence baseline? The prior
independent-source plan required new numeric anchors before any additional rate
dependence. The completed Maben audit failed that gate. This plan explicitly
relaxes that requirement ONLY for a bounded structural falsification on existing
development data. It does not admit new biochemical labels, derive numeric priors
from prose, or claim independently calibrated rates. Independent measurements
remain necessary for physical interpretation and the final availability goal.

Primary qualitative motivation: Chang2005, PMID16286653,
https://pmc.ncbi.nlm.nih.gov/articles/PMC1287962/, reports substrate-length effects.
Three coarse bands are a modeling hypothesis, not a universal enzyme law:
1–8, 9–16, and >=17 residues. Rates in 9–16 are the reference. Two free log-rate
modifiers represent the other bands; neither sign is forced by this source.
Training-only observed peptide counts by length are 8:153, 9:2005, 10:701,
11:331, 12:78, 13:29, 14:7. Long states are latent precursors, not observed labels.

## Unchanged data and evaluation contract

- Canonical HAP1 Shapiro2025 PXD056426 source88cad10d… and recipea1eec392…;
  original source/recipe and baseline artifacts must pass their pinned hashes.
- Exactly the previous ERAP1 outer-training cohort: 8,281 observation cells,
  3,304 peptides, 1,296 molecular components, 1,514 parents, 3,409 origins.
  Keep every alternative origin and the 1,741,916 birth-to-target paths.
- Positive KO replicate minus mean log10 of >=2 positive WT intensities;
  existing B2M exclusions and complete-precursor policy unchanged. No imputation,
  inferred bounds, generated negatives, target averaging or synthetic biology.
- Same five component folds, hierarchical weights, source uncertainties and
  fold-specific frozen sequence_0.1 baseline/gamma. No outer validation/test
  predictions or scores; no retuning of earlier MAPP/native/macrophage tests.
- This measures relative MHC-elution response among observed positives. HAP1 is
  not a professional APC; shared studies/controls and homology limits persist.
  Binding/nM thresholds do not apply. No external species/KO transfer claim.

## Model and fixed comparisons

Keep the two-compartment unit-input/unit-clearance construction and source influx
from the preceding model. N-terminal transitions remove one residue. At a state
with remaining length l and removed residue a, log rate is theta[a]+eta[band(l)].
eta[9–16]=0; monomers have zero further trimming and clear at rate1. Preserve all
1..L products, without an MHC-length filter. KO multiplies trimming by sigmoid(z),
as before; this is an effective response parameter, not measured residual ERAP1.
Parent KO input is 10**(gamma*measured_parent_log10_response); origins are mixed
before taking log10(KO amount/WT amount). All other machinery is unchanged.

Four new arms, each full training plus five fold fits:
1. Length-only physical exact: shared theta, two eta, z (4 parameters).
2. Residue+length physical exact: 20 theta, two eta, z (23 parameters).
3. Frozen sequence baseline + residue/length correction, exact (23 new parameters).
4. Same augmented model, conditional sampled score gradients, after qualification.

The augmentation subtracts gamma times the direct precursor surrogate before
adding the physical ratio, as previously. Profile one offset on fitting rows at
every objective. Keep all seven original controls and all four old physical arms
as frozen references, never refit or reselect them. At eta=0 the new model must
reproduce old scores, predictions and unpenalized objectives within absolute2e-9.

Weighted MSE + .01*mean(theta**2) + .01*mean(eta**2) + .01*z**2.
Zero initialization, seed42, float64 CPU/one numerical thread, Adam150 steps,
lr.03, betas(.9,.999), eps1e-8, final checkpoint only. No pretraining, restarts,
post-outcome extensions, feature selection, clipping, group standardization,
sample reuse, or reference KL. This family does not train other Presto branches.

## Computation and qualification before biological fitting

Represent each origin by ordered start states0..a for fixed end b. Prefix sums of
log(rate/(1+rate)) compute every birth-path contribution; final hazard is1+rate
(or1 for a monomer). Segment boundaries cancel in prefix differences. Avoid a
large residue-by-length path-count tensor. All states and paths remain enumerated;
the sampled arm retains exact forward amounts/posteriors and does not save exact
scores. Report actual state/path score work and runtime without claiming otherwise.

Required tests: independent scalar recurrence/linear balance, monomer/no-trimming
boundary, multiple-origin sums, finite differences, exact conditional-score
expectation, and eta=0 equality with old residue-count model. Full-cohort compare
at three fixed probes, and three finite-difference directions per probe (1e-7
absolute derivative tolerance). Failure stops fitting until a documented numerical
fix; never relax scientific gates or replace failed outcomes.

Sampled augmented data-gradient qualification: three parameter probes (zero and
plus/minus .2 times seed42 normal direction), eight draws each, seeds5100+100*probe
+repeat, paired WT/KO uniforms. Try K512, then only if failed K2048. Relative L2
error of mean <=.10 and RMS per-draw relative L2 error <=.25 at every probe,
denominator max(exact norm,1e-8), using the unpenalized profiled data loss. Preserve
both attempts. If both fail, skip sampled fitting and finish exact comparisons.

Resource caps: build/qualification900s; each exact fit600s, sampled fit1200s;
all fits7200s; peak4GiB; independent verification900s/4GiB. Stop on cap violation,
retain partial artifacts and revise prospectively before further work.

## Decision and closure

Primary gate: augmented exact inner-CV RMSE >=2% better than frozen sequence_0.1
and weighted physical-correction SD >.001. Report all arms regardless of outcome,
including training/fold/CV MSE, RMSE, MAE, R2, pooled Pearson/Spearman, spread,
component and length strata, parameters/support/gradients and runtime. No blind
test or concentration claim even if the gate passes. A failed gate stops this
length-only extension; do not keep changing features against these same outcomes.

Preserve prospective config/source/environment, git state, checkpoints0/50/100/150,
optimizer/RNG/history, every full/fold prediction, training and inner-CV dumps,
cohort/origin/path references, comparisons and independent verification. Independently
reconstruct recurrence predictions, source grouping, offsets, penalties, metrics and
gate without importing producer model code. Update README, canonical log, task
review and PR. Update model_to_beat only if this contract's baseline actually changes.

## Checklist

- [x] Inspect previous model, source gate failure and training-only length support.
- [x] Freeze this revision and comparison before implementation.
- [ ] Implement and qualify the state-length extension.
- [ ] Run the fixed biological comparison locally.
- [ ] Independently verify and publish the outcomes, including failures.

## Numerical stop and prospective recovery

The first implementation passed response/objective equality and finite differences,
but the subsequently added full-path check (which should have preceded fitting)
failed: global prefix accumulation differs from the residue-count reference by up
to2.375e-7, above the unchanged2e-9 tolerance. Independent diagnosis confirms this
in both the executed Torch producer and NumPy reconstruction. The much smaller
response error does not override the registered path gate. Training was terminated
with exit143 after11 completed fits and an incomplete twelfth; preserve every
checkpoint, prediction, log, original source and diagnosis under an explicit
attempt-global-prefix directory. These are unqualified partial runs, not winners.

Before restarting the same fixed family, replace the global prefix with per-origin
prefix sums. Bucket origin lengths by powers of two, pad only within each bucket,
sum along rows and scatter valid positions back into original path order. This
keeps exact support with <2x padded states and avoids accumulation across origins.
No data, model, seed, optimizer, loss, bands, budget or scientific gate changes.
Require full-path, response/objective, scalar recurrence and gradient qualification
before ALL new fits. Add a long multi-origin regression test that fails with the
old global prefix. Keep the original2e-9 path and1e-7 finite-difference tolerances.

The inherited gradient gate code used a1e-12 denominator floor instead of the
registered1e-8. Independent recomputation found norms.00094–.00128, so both floors
were inactive and all actual results identical. Correct future code to1e-8 and
preserve the original receipt. Rerun qualification prospectively; no changed
thresholds, sample sizes or seeds. The new numerical attempt starts all fixed
fits from zero only after admission; this is not an outcome-driven fit extension.
