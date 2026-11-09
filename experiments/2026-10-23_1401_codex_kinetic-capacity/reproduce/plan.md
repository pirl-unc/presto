# Training-only decomposition of kinetic model capacity

## Motivation and scope

The40-coefficient independent-cut fit is stationary but leaves substantial training error. The registered one-entry-event extension converges toward essentially immediate entry and gives no development improvement; independent forward/gradient replay verifies that result. This does not distinguish feature/regularization limits from a process/measurement-family mismatch. Before adding more kinetic states, quantify these separate limitations.

Use only the six existing training parents and seven conditions from the same pinned shared Hitlist2.0.9 recipe. Both development-validation and previously exposed test parents are excluded before target construction. No source relabeling, altered curation, invented negatives, biological split reuse or new concentration claim. The deterministic capacity experiments below are diagnostics; their extra degrees of freedom are not candidates for promotion or transfer to APCs.

## Fixed comparisons

1. Retain the saved regularized40-coefficient sequence/protocol prediction as a fixed reference. Use its exact source targets/weights and final coefficients. Do not retrain it.
2. Remove the0.01/2 coefficient penalty while retaining the same40features. Start from the saved fit. This isolates whether the penalty prevents fitting the observed training trajectories.
3. Replace the sequence/context mapping with positive free bond rates per condition, initialized to that same saved fit's rates. This relaxes all sequence/context sharing while retaining the identical independent-cut process, observed outputs and objective. Allocate only rate groups that influence admitted training outputs. Bonds with identical internal/boundary incidence across unique admitted spans may share an effective rate; never claim independently identified physical rates from observationally equivalent bonds. Unsupported bonds remain fixed at their original rates and are not optimizer parameters. The two Ffh conditions can have distinct rates in this diagnostic; this explicitly removes context-transfer constraints as well as sequence-transfer constraints.
4. Compute a feature-independent lower bound on achievable error under necessary kinetic shape constraints. Within each condition and literal peptide span, fit one common log temporal curve to all retained source biological labels, with the first common positive time fixed to0 in log-ratio space. The unconstrained weighted per-time mean is a repeat-label disagreement floor, not a biological noise ceiling. Project those means onto (a) log-concavity plus the immediate-cut growth envelope and (b) log-concavity plus the entry-model growth envelope. These are necessary conditions, not sufficient characterizations of either full process, so their optimum errors are lower bounds. Do not call the projected curves mechanistic predictions.

## Shape projection

For increasing times t0...tm and z0=0, minimize sum_r,j w_rj*(zj-y_rj)^2 with the original hierarchical weights. Let b be the number of required fragment boundary cuts (1 at a protein terminus,2 internally). Adjacent constraints are:

- zj+1-zj <= b*log10(tj+1/tj), or b+1 for the entry-envelope version.
- (zj-zj-1)/(tj-tj-1) >= (zj+1-zj)/(tj+1-tj).

Adjacent constraints imply the corresponding all-pair growth bound and discrete log-concavity. Zero-reference time is a positive measured time; never take log(0). The prior recipe only admits proper5–40-residue products from longer proteins; verify this rather than infer it. Work in physical time for concavity and log time only for the growth envelope.

Use a tiny convex quadratic program per curve. Independently verify feasibility, KKT stationarity/complementarity and the weighted objective, and cross-check representative active sets with an independently formulated solver. Preserve active constraints, unconstrained means, projected per-observation predictions and residuals. Do not silently project inadmissible source rows or change measured values.

## Optimization and numerical checks

For comparisons2/3, use the same exact expectation primitive. No stochastic estimator is needed for this capacity bound; existing exact and sampled fits stay fixed references. Start at the saved regularized rate field, use Adam lr0.01 for2,000updates, then a training-only L-BFGS polish limited to500function evaluations and gradient tolerance1e-8. No validation criterion or restart search. Preserve all optimizer settings, checkpoints, gradients/coverage, final objective and termination reason. If the optimizer does not become stationary, report that as an incomplete capacity bound; failure of that fit cannot establish process impossibility. Record both the post-Adam and polished states, including any worse/failed polish.

Verify independent NumPy predictions/analytic gradients and objective against the actual final states. Compare zero-update predictions to the saved reference, including all condition-specific allocations. Mathematical gradient/support checks precede fitting. Free-bond rates need not be uniquely identifiable; report incidence groups, unsupported bonds, parameter counts and remaining equivalences.

## Interpretation and next action

Report equal-parent training RMSE/MSE and per-parent results with pooled Pearson/Spearman, explicit analytic rank ties and output variation. No held-out metrics exist in this family because it is a training-capacity diagnostic, not a predictive evaluation; preserve every training prediction and make this absence explicit in README/registry.

- A large unregularized40-feature improvement implicates the fixed penalty as one limitation.
- Additional free-rate improvement implicates sequence/context sharing; it does not prove a transferable richer encoder will reproduce that gain.
- The shape-projection excess over the replicate-mean floor quantifies error that even freely chosen curves cannot remove while retaining those necessary constraints and a common time-independent observation response.
- Remaining gaps are not automatically biological kinetics: assay response, precursor contaminants, contributor changes and source error remain possible.

Freeze a subsequent transferable representation or state-dependent processing hypothesis only after this evidence. Do not insert per-condition identity parameters into the general APC model. Keep the full goal—relative peptide concentrations under APC machinery, combined with independently scoped measurement/presentation models—open.

## Execution boundaries

Register a new timestamped family before running. Pin original recipe/source and the saved reference checkpoint. CPU one thread, float64, unchanged shared environment, no Modal. This family may overlap the already running sampled-entry fit; record concurrency and do not present wall times as an isolated hardware comparison. Independent scripts, complete source snapshots, launch JSON/shell, all targets/predictions/metrics and failed attempts belong in the canonical registry.
