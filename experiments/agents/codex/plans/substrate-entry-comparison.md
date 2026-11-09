# Prospective substrate-entry comparison

## Evidence and scope
The biological independent-cut fits converge but fail the combined held-out gate. Training-only observation family2026-10-23_1257 shows that restoring the applied baseline leaves92.73% of original growth-envelope violations and98.76% of concavity violations. Most training weight has a small subtracted reference baseline. This does not prove processing biology is the cause; observation/model misspecification remains unresolved.

Test one process hypothesis: substrate molecules enter processing at different times rather than all bonds becoming available at time zero. Keep current exact and sampled references unchanged. One extra global positive entry rate is the first extension, not a broad neural redesign. No parent/peptide-ID embedding or fitted MS gain. This is not expected to remove every remaining shape restriction.

## Model
Let entry time E~Exponential(lambda), shared entry-rate parameter across constructs. Before entry the original parent is intact. After entry, use the existing sequence-conditioned independent bond-cut process with age t-E. For a proper fragment f:

m_f(t)=integral from0 to t of lambda*exp(-lambda*u)*m0_f(t-u) du.

For the intact parent, add exp(-lambda*t). Retain every contiguous span, including singleton and nonbinder products. Complete sampled trajectories contain the intact parent when E>t, otherwise a full independent-cut partition at age t-E. Expected residue mass must remain equal to the initial parent length. Physical time, supplied protocol proxy and measured per-peptide reference MS signal remain explicit; no molecular-concentration calibration is implied.

Use the existing40 sequence/protocol coefficients plus log(lambda), initialized at lambda=1/hour, without asserting this is a known biochemical rate. Matched one-rate+entry control has two coefficients. Immediate-entry limit must recover the existing model. Inspect gradients, fitted limits and profile/rank support; a nonzero update is not identification of a meaningful entry timescale.

## Mathematics and shape qualification before biological fitting
Derive entry convolution and score gradients independently. Test tiny complete cut-state enumerations plus independent one-dimensional numerical integration at known rates/times. Verify expected residue mass including the unentered parent and gradient correctness. Establish which growth bounds change (early proper-fragment birth has one extra entry event) and whether log-concavity remains. Do not advertise entry delay as an explanation of persistent nonconcavity if its mathematics forbids that.

Prefer a stable positive differentiable integration over the conditional exponential entry-time CDF, with the activation probability handled analytically. Fixed Gauss nodes approximate the convolution; call it differentiable quadrature, not exact arithmetic. Qualify converged128/256-node or independent adaptive-integration reference against the cheaper candidate grid over the actual time/rate domain and mathematical fixtures. Exact raw independent-cut expectation remains a separate fixed reference. If analytic closed forms are used, validate near-equal rates and cancellation explicitly rather than tolerating negative/tiny corrupted masses.

The sampled arm draws real entry times and complete partitions, with fresh draws each update. First qualify full score and irrelevant-bond-integrated score gradients on training data. Keep the log-ratio residual/denominator contribution explicitly analytic/quadrature-hybrid. No pure-RL, unbiased-exact or compute-saving claim if the objective uses quadrature or exact auxiliary terms. The log-entry score is1-lambda*E for entered paths and-lambda*t for unentered paths. Retain rare-event/zero-count failures. Freeze sample count/optimizer budget after training-only mathematical/runtime qualification, before new validation results.

## Biological comparison and evaluation boundary
Use the unchanged shared2.0.9 recipe, six training and three development-validation parents. Same common times, references, corrected targets, equal-parent weights and original observation provenance; no source exclusions or alternate raw-target relabeling. The two previous test parents remain closed to further development evaluation in this family. The earlier test result and source selection are known, so these development comparisons cannot certify generalization. An independent source/preparation cohort is still required for promotion.

Compare matched immediate-entry and delayed-entry sequence/protocol models and one-rate controls with the same declared optimizer/budget/regularization for shared coefficients. Frozen final checkpoint; no rescue grid on validation. Use original source-aware temporal repeatability as a context diagnostic, not a predictive benchmark. Preserve all training and development-validation predictions, loss terms, parameter/feature support, integration errors, sampled-gradient variance, work/runtime and exact initialization. No test dump is expected: explain the closed-test development policy in both README and registry. No MHC affinity/binder metrics for this assay.

## Stop/replan and full objective
Do not run expensive biological fitting until numerical qualification and a measured local runtime budget are frozen. If entry adds an unsupported parameter or fails matched development controls, record that outcome and inspect training residuals; do not expand a model until it passes. No promotion from this family alone. APC machinery effects, transport/clearance, independent MS response and relative concentrations across peptide identities remain outstanding. A future fragment/processing-state-dependent trajectory may be needed; do not equate this one-event extension with the completed APC simulator.
