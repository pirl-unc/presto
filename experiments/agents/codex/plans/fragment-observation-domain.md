# Full-observation numerical admission before a bounded biological fragment fit

## Decision and goal

The adaptive boundary proposal passes all48 small-system checks and independent
replay, but only7/21 full-parent cases pass atK512. For eta=+1 it reduces worst
RMS gradient error1.3585→0.2290 and worst mean absolute log error0.5294→0.0950;
none meets the0.05 log-error gate. For eta=-1 all cases still fail. Stop this
proposal extension as registered. Do not choose more samples/seeds or another
bridge variant to rescue these exposed cases.

Next determine whether the one-parameter fragment-length mechanism can be tested
on the actual biological observation domain with bounded differentiable training.
The scientific question is whether it improves the existing shared-observation
fit, not whether a numerical fixture can be made to pass. The sampled arm remains
an explicitly failed reference until separately qualified. The full APC quantity
goal remains open, including machinery, transport/clearance, MS calibration and
cellular/species validation; purified digestion is a precursor model component.

## Unchanged biological source and observation contract

Use the pinned canonical Hitlist digestion recipe and the existing source-linked
shared-observation family2026-10-23_1501. Freeze its source/recipe/feature hashes
and row identities at registration. Retain90,630 training and14,822 development
measurements, their complete-case selection, source contributors, molecular-form
ambiguities, instrument scope and equal-parent/condition/curve/time weights.
Recheck counts and exact targets rather than assuming those numbers persist.
All previously used test parents remain excluded from fitting/evaluation. Do not
call development a new blind test. Use canonical exports, not a new curator.

Before numerical work, create a projection of only training parent sequences,
protocol covariates, required source spans and times. It may use known observation
support but not measured quantities to select easy targets. Cover every required
training span/time, including repeated-origin aggregation if present. No new
exclusions, peptide-length filters, imputed negatives or response calibration.
Quantities remain closed during this numerical admission stage; document whether
the source serialization contains excluded records even when they are never used.

## Physical model and practical precision criterion

Retain all fragments1..L and the same full-support branching generator:
`hazard_k(fragment)=exp(x_k @ theta)*(fragment_length/20)**eta`.
Theta has40 supported sequence/protocol coordinates; eta is one global effect.
The physical process always conserves residues. Expected fragment counts per
initial parent are the molecular output. The model definition has no probability
floor, observed-set normalization or MHC selection.

The previous blanket numerical criterion required every CRAM output, including
vanishingly tiny unobserved tails, to be nonnegative; it failed. This prospective
stage uses a **different, explicitly limited admission criterion**: every actual
required log amount and the training-loss derivatives must be numerically
resolved, while all full-output negativity remains recorded. This does not
retroactively pass the old gate or qualify arbitrary unobserved tail queries.
The generator still includes the complete state space. Do not clip/discard
negative entries, omit biological rows, or relabel numerical artifacts as biology.
Any later inference API must expose the numerical scope and use physical sampling
or a qualified positive calculation for unsupported queries.

At the original40-coordinate initializer, evaluate eta=-1,0,+1 at every unique
training condition/time and all its required spans. Compare fixed CRAM48 two,
four and eight substeps; the eight-step reference is predetermined, not a rescue
chosen after results. Require positive finite values for every required span,
maximum two/four/eight log disagreement<=1e-5 natural-log units, and eta0 parity
with independent analytical log counts. Save all required per-span predictions,
full arrays or exact archive paths, minima/negative counts and timings. Treat
underflow as unresolved; do not replace it with epsilon.

For each condition/time/eta, verify three fixed normalized directions (sine and
cosine over40theta coordinates, plus eta) against independent NumPy central
finite differences of a fixed signed log readout covering all required spans.
Use h=1e-5, tolerance1e-5+1e-6*absolute derivative. Also record the full autograd
41-coordinate gradient. This is a directional empirical precision check, not a
universal true-error proof. Match the independently implemented triangular solver
and exact small-system reference; reuse previous proofs only for unchanged code
and states. Bound each fresh worker at120s/4GiB, CPU float64 one thread. Preserve
all failures. Do not launch biological fitting if any required input is unresolved.
No development quantity or loss is used to make this numerical decision.

## Coherent memory-bounded biological trainer, only after admission

Keep the shared response layer:38 training-supported instrument/sequence
coordinates and one constant nuisance offset per training condition. Predict
`log10(signal)=log10(expected count)+response+condition offset`.
The offset leaves absolute between-condition scale unidentified. No per-peptide,
per-time or curve intercepts are added. The response remains fitted nuisance
structure, not independently calibrated MS detectability.

The earlier trainer profiles offsets analytically. For scalable minibatches,
optimize those same offsets jointly, initialized to their weighted least-squares
solution at the common starting parameters. This is the same underlying weighted
least-squares objective; its minimum over offsets is exactly the old profiled
objective. Qualify that equality and envelope gradients on a fixture and actual
initial values before launch. Treat any optimization difference as a controlled
trainer change, not biological evidence.

Sample one condition/time block with probability equal to its total training
weight. Within that block, divide observation weights by the block probability.
The expected minibatch data loss and gradients then equal the full weighted
objective. Add the unchanged theta/response regularizer once per update. Prove
this expectation by enumerating all blocks on the independent-cut reference.
This avoids retaining autograd graphs for all protein/time matrices at once.

Two arms use identical block draws, initialization, observation features,
regularization and fixed optimizer schedule:
1. eta fixed0, analytical independent-cut amounts;
2. one learned eta with the full-support deterministic fragment solver.

Freeze the eta parameterization/range, regularization, optimizer/learning rate,
step count, seed, runtime cap and checkpoint policy after runtime qualification
but before any new biological update. Do not import a validation-selected
initializer or select schedules from observed fit outcomes. Every allocated rate,
response, offset and eta coordinate must have support and update diagnostics.
A common final checkpoint is evaluated; no exposed-test rescue or outcome-driven
schedule extension. Keep original constant/response/time and published hybrid
controls visible with their distinct optimizer contracts.

During fitting check positive, resolved log amounts for every selected block;
compare two/four-step outputs using the same frozen threshold. An invalid update
stops the run and preserves it; no silent eta reset, count floor or dropped row.
Final training/development prediction dumps must check every required output,
not only sampled blocks. A checkpoint failing that check is not qualified.
Report train/development weighted loss, RMSE, Pearson/Spearman, predictive spread,
original temporal-ratio metrics, expected amount/rate/response ranges, coefficient
coverage, active numerical constraints and actual memory/runtime. Development
condition offsets define a scale-invariant metric and must not be presented as
absolute out-of-sample signal predictions. No binding-threshold metrics apply.

## Stop and interpretation rules

This plan first admits the actual observation domain and measures runtime; a
separate frozen execution receipt determines the bounded fit. It authorizes no
fit on a conveniently smaller cohort if the full registered cohort fails.
If the exact fit is numerically feasible, its biological result determines
whether further work on this added mechanism and its sampled estimator is useful.
If it fails numerical admission, publish the exact offending spans/parameters and
replan from that evidence rather than dropping them. In either case, preserve
all original exact/sampled failures and the full APC availability objective.
Existing source/calibration gaps remain under their Hitlist issues; file a new
issue only for a concrete new missing source or metadata requirement.
