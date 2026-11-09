# Resolve kinetic capacity optimization before extending the process

## Evidence and question

Family2026-10-23_1401 independently verifies training RMSE0.88315 for unregularized40-feature rates and0.79734 for2,121 free condition-specific rate groups, versus0.90257 for the fixed regularized reference. Both fits stop at500 L-BFGS function evaluations with gradient maxima above1e-8. Necessary-shape projections give lower errors but are not realizable-process guarantees. Do not claim the remaining gap proves state-dependent kinetics are necessary.

Test whether poorly scaled log-rate coordinates and rate-zero/infinite limits account for incomplete optimization. This is a numerical/training-capacity follow-up on the same model, not a new transferable architecture or biological evaluation. Use only the same six training substrates/seven conditions; exclude both prior held-out strata before any target construction. Preserve the same source manifest, ratios, weights and incidence-group allocation.

## Stage A: training-only geometry and boundary audit

Before fitting, independently inspect saved rate ranges, reference fragment masses, objective/gradient evolution and the smallest/largest-gradient groups. Separate unsupported fixed bonds from supported grouped parameters. Identify near-zero or saturated boundary rates and characterize whether nonzero observed-reference calibration would require an arbitrarily large peptide response. Do not silently treat a ratio-limit solution with zero absolute fragment mass as a physically identified concentration.

Reconstruct residual Jacobians and inspect conditioning by condition using sparse operators; avoid materializing a giant dense observation-by-parameter matrix. Check whether flat directions reflect equivalent internal hazards, saturated boundaries or shared features. A nonzero gradient/update is not identification. Preserve per-group parameter diagnostics and representative training curves; no outcome-based source exclusions.

## Stage B: bounded change of optimizer coordinates

The free-rate objective separates across conditions. Refit those blocks in direct nonnegative rate coordinates with sparse analytic Jacobians and trust-region nonlinear least squares, initialized from the saved rate field. Freeze the exact solver limits/tolerances after StageA mathematical/runtime qualification, before fitting. Report actual function/Jacobian evaluations, projected first-order optimality, bound-active parameters and every final prediction. No restart search or newly inspected validation/test outcomes.

The boundary log-ratio has a stable closure at rate0:

f(r)=log(t/ref)+log(exprel(-r*t))-log(exprel(-r*ref)).

At r0 it equals log(t/ref), and its rate derivative is-(t-ref)/2. Use stable small-rate series/independent numerical checks for derivatives and the original formula away from cancellation. Internal survival contributes-r*(t-ref). Qualify direct-rate and log-rate formulations, Jacobian-vector products, positive-rate prediction equivalence and tiny constrained cases before any large fit.

Allowing a zero boundary rate extends the relative-ratio objective to its mathematical closure; it does not create a nonzero fragment amount. Explicitly distinguish interior physical rates from boundary infima/undefined absolute calibration. Do not insert zero-rate ratio limits into the reusable concentration predictor or reinterpret them as positive yields.

Retain the previous log-rate fits unchanged. This is a diagnostic coordinate comparison, with solver limits recorded separately from biological model changes. If sparse fitting remains nonstationary, preserve that uncertainty instead of repeatedly expanding budgets without evidence. A better fit still needs transfer evaluation; condition-specific identities are never promoted into the general APC model.

## Next scientific decision

Only once numerical limitations are characterized, select between a richer but parsimonious transferable sequence-to-rate mapping and a process whose cutting rates depend on the current fragment/processing state. The latter has a genuine discrete trajectory and can be compared with matrix-free expected-fragment evolution, but requires its own prospective mathematics/runtime/data contract. Independent MS response, APC machinery/transport/clearance and concentration validation remain part of the full goal.

Register this work as a new family before execution; unchanged shared environment, local CPU, canonical Hitlist recipe, all predictions and independent verification. Current two experiment families must retain their original artifacts, failed attempts and explicit absence of new held-out evaluations.
