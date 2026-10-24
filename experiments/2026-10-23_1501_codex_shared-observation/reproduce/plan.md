# Preserve cross-peptide signal before expanding the kinetic model

## Evidence and hypothesis

The direct-rate diagnostic (2026-10-23_1438) changes training ratio RMSE only from 0.797339 to 0.797292. It moves 1,288/2,121 rates below 1e-10/hour and shifts median predicted reference abundance downward by 14.28 log10 units. Thus more successful optimization of the ratio loss does not establish peptide concentrations. No global optimum is proved, but another optimizer-budget expansion is not justified by the observed gain.

Next hypothesis: calibrating each peptide from its own measured reference discards information needed to constrain relative peptide abundance. Replace those independent reference calibrations with a small shared observation model while keeping the independent-cut kinetics fixed. Do not simultaneously add fragment-state kinetics. This tests an observation/objective change, not a claim that MS response has been independently identified.

## Source and metadata preflight

Use the same canonical Hitlist digestion source and admitted curves; both previous test parents stay closed. Training and development labels keep their original membership. Preserve every original source value, positive-selection rule, background/contributor link, missingness and modification ambiguity. Include the reference time itself as a measured signal target instead of dividing every peptide by it. The existing source has 23,054 training curves / 90,630 common-positive measurements; verified from the pinned recipe at specification time and must be rechecked at registration. The earlier observation audit used 91,226 all-positive points; it is a different time-support contract.

The publication used two MS instruments and substrate-dependent reduction/carbamidomethylation. The pinned recipe exports contexts/raw quantities/filtered/final vectors/constructs but not the existing digestion_runs table with instrument_raw. Before assuming shared response, use the canonical source's run links and homogeneous protocol metadata to audit actual contributors to each retained observation. An aggregate with unresolved mixed instruments/forms is a documented limitation, not permission to assign an arbitrary run or common gain. Extend the shared recipe/projection if needed; do not build a Presto-only source curator. Hitlist770 owns digestion observation lineage;762 owns broader known-load response calibration. Do not reopen closed775.

Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC10850103/ (Methods, MS measurements and data processing). Protocol-wide facts are not independently observed per-run assignments.

## Minimal model and output

Keep the 40-coefficient positive sequence/protocol bond-rate model, exact expected amount m(peptide,time) per initial parent, and complete-product sampling. All fragments remain in the process; no MHC-length or affinity selection. Do not use condition-specific free bond identities as the biological model.

Predict log10(signal) = log10(m) + shared_response(peptide, known acquisition scope) + one condition-wide nuisance offset. The offset is constant across peptides and times, profiled analytically with the frozen weights; it is not one offset per peptide, time or curve. Its inclusion explicitly leaves absolute between-condition scale unidentified. Known nominal precursor concentration remains recorded rather than being relabeled as measured fragment abundance.

Fixed observation variants:
1. Constant peptide response: no sequence response coefficients.
2. Linear shared response: length and amino-acid fractions (19 contrasts), centered/scaled and rank-reduced on training contrasts only. No identity embeddings, parent IDs, per-peptide intercepts or unconstrained nonlinear response net.

Do not import the poorly transferring 22-peptide SWATH calibration coefficients into this DDA assay. A shared response fitted here remains a nuisance hypothesis until independent known-load data establishes transfer. If acquisition scopes cannot be reconciled, freeze scope-specific deviations only where supported, report their exact rank/parameter count, and do not infer absent effects.

## Objective, controls and fitting

Use original log10 positive source signals and equal-parent/condition/curve/time weights. The proposed profile removes only a condition-wide constant and preserves across-peptide composition and temporal yield changes. Compare signal-shape loss, original temporal-ratio metrics and predicted mass/rate distributions separately; improvement under a changed objective is not improvement on the old endpoint by definition.

Retain the old ratio-trained fixed checkpoint, a shared response-only control and a simple time-only control. Freeze actual optimizer limits, initialization, regularization and score-estimation settings after numerical/runtime qualification and before fitting; no validation search or exposed-test rescue. Validate source values and every effective target transformation first. Use one declared initializer and fixed final checkpoint per arm.

Exact and sampled variants must share the same source, observation layer, objective and initialization. Qualify fresh sampled complete-partition score gradients at initial and realistic saved states with fixed K=32/128/512 and 16 groups. Report error, rare-product counts, score evaluations and time. If using exact residuals/denominators, label it a hybrid explicitly. Do not launch a biological sampled fit if the K512 mean-gradient error exceeds 20% of exact norm or rare-event support causes undefined estimates; preserve that negative qualification instead of increasing K opportunistically. Exact fitting can proceed while the sampled limitation guides a separately specified conditional/importance-sampling estimator.

## Numerical and scientific checks

Before fitting, qualify analytical profiling against direct weighted least squares, its gradients against finite differences, original positive-rate fragment amounts against the independent calculation, residue conservation and predictor invariance to the nuisance offset. Do not add epsilon amounts or silent rate floors that change the model. Numerical bounds, if needed, are explicit experimental constraints with active-bound diagnostics.

For every fit preserve all effective labels, predictions, parameters and gradient-support masks. Report train and fixed-development signal-shape/ratio RMSE, Pearson/Spearman, prediction spread, within-parent peptide rankings, expected fragment masses and implied response range. Development offsets use aggregate labels only to define a scale-invariant shape metric; do not present that as calibrated absolute held-out signal prediction. No new test estimate exists.

Check for remaining scale degeneracy by fixed post-fit perturbations of shared rate scale and compensating allowed nuisance offsets, and report objective/mass sensitivity. A fit that only improves by extreme or unsupported response gains is not a working concentration model. Independent MS calibration, APC machinery/transport/clearance, species/cell transfer and cellular validation remain required.

## Decision after this experiment

If shared observation constraints improve the fit and stabilize relative amounts, retain that observation contract and next compare a minimal fragment-state-dependent cutting process by differentiable expected-fragment evolution versus sampled discrete trajectories. If not, use the precise residual/identifiability evidence to choose a state-dependent extension and stronger calibration data; do not add arbitrary capacity or call these purified-digestion amounts APC concentrations. Register each family before execution and publish all negative outcomes.
