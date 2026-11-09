# Training-only diagnosis of the intact kinetic failure

Registered October 25, 2026, after the completed 0046 temporal evaluation. That
family's 100/120-minute outcomes are inspected and must not be used for model
selection or described as blind in any extension. The previous goal turn made
progress: full biological/independent replay, report and canonical registry closure.
The full precursor-relative APC availability goal remains active and unachieved.

## Hypotheses and scope

H1: the independent-bond model's increasing, concave log product/substrate curves
force substantial lack of fit. H2: the fixed 800 Adam updates left avoidable
training-objective error. Diagnose both with unchanged 626 original training
measurements from 96 peptides, not new curation or additional source quantities.
This is a training diagnostic, not a new held-out biological success claim.

Input allowlist: original train covariates/outcomes, exact uniform/residue
checkpoint/summary, and production model source. Never call its data() function
(which reads held-out covariates). Read no validation/test prediction or outcome
file. The previously published scores remain unchanged; existing test seals stay
intact. No new Hitlist recipe is needed because the canonical cohort is unchanged.

## Shape bound

For constant independent bond hazards, natural-log product/parent ratio equals
u times the sum of rates outside the product's interior plus log(1-exp(-lambda*u))
for each existing boundary. Its first derivative is positive and second derivative
nonpositive; fixed peptide offsets do not alter either. Derive and test this on
the actual production component using 0/20/40/60/80 minutes and the original
initial/final parameters, plus a fixed mixed-rate parameter vector.

For each training peptide, aggregate replicate means using original weights.
Fit three descriptive curve classes: free timepoint means, nondecreasing means,
and nondecreasing concave means. Use an exact small active-set quadratic solver
enumerating constraint subsets (at most 5 times, 7 inequalities). Preserve every
per-example fitted value, active constraints, gradients and KKT residuals. Verify
monotone results with independent weighted PAVA and concave results using SLSQP;
fixed prediction tolerance1e-7 and KKT/primal feasibility1e-9. Constant/increasing/
decreasing/peak fixtures must qualify the actual solvers before observed-data use.

Report original-weight MSE for each shape class and archived physical fits. The
free-curve residual is within-time replicate variation, not pure measurement noise
or a biological noise floor. Report counts of negative adjacent mean changes and
negative unpenalized peptide slopes. No interpretation as causal destruction or
assay artifact. The difference between shape floor and free floor measures forced
descriptive error, not independent predictive gain.

## Optimizer check

Use the exact same production objective, bounds, regularizer, profiled training
offsets, and 626 measurements. For each uniform/residue model run L-BFGS-B from:
(1) its archived final Adam parameters; (2) original initializer; (3) all rates
1e-5/min, tau60min. Numerical raw-coordinate bounds[-30,30] approximate the same
sigmoid physical bounds and avoid overflow; state this small domain restriction.
Fixed maxiter1500, maxfun3000, maxls40, gtol1e-9, ftol1e-13. No restart based on
outcomes, extra seeds or test evaluation. Six conditions, all reported.

Save every function evaluation and accepted optimizer state, final training
predictions, params/rates/tau/offsets, convergence/status/gradient, objective and
MSE, runtime and counts. Compare objective/MSE against original Adam. Independent
NumPy forward/offset replay must agree1e-10; finite-difference final-objective
gradient absolute tolerance1e-6. Do not infer global convergence from local solvers.
All allocated parameter gradients and changes remain reported, including coordinates
at numerical boundaries or effectively flat directions.

## Decision rules

If optimized training MSE improves >=5% over Adam, optimization remains a material
candidate limitation; otherwise stop extending optimizer machinery on this cohort.
Quantify the shape penalty regardless. If shape-constrained descriptive MSE exceeds
the free-curve residual by >=10% of the original physical MSE, classify shape as a
material incompatibility with these training trajectories, without claiming its
biological cause. Both can be true or false. No promotion of these train-only fits.
Use the result to write a prospective next model/evaluation plan; do not score a
revised model on the inspected temporal test to obtain a replacement headline.

## Reproduction, resources and closure

One timestamped experiment with frozen spec, argv/env/git/hash receipts, train
predictions, all conditions, independent checks, summaries, report and canonical
log. No validation/test dump because the sole purpose is diagnosis on previously
fitted training data. No synthetic biological labels, affinity or binding metrics.
Local CPU float64/one thread, no Modal or package changes. Caps3GiBRSS,
20minutes/stage,32MiBnew logical artifacts,1GiBfree disk+32MiBstartup margin.
Keep failed runs/checks; no overwrites. Source/recipe publication proceeds separately.
