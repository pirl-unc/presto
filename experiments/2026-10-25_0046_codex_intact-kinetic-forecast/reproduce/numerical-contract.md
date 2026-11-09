# Frozen numerical details before fitting

The shared recipe admits626training,181validation and186test observations,
96/75/74peptides. Full recipe support and independent target replay must pass.
All admitted peptide sequences have exactly one occurrence in the140-aa substrate.

Run both uniform and residue-conditioned physical models with exact gradients,
and each with conditional score gradients at16/64draws and seeds42/43/44:14fits.
All use800Adam updates,lr.03,betas(.9,.999),eps1e-8,float64,oneCPUthread and fixed
final checkpoints. Initial rate.001/min,tau10min; sigmoid bounds from the model
plan. Residue penalty.01mean squared centered natural-log rates. Forward losses
and held-out inference remain exact for every condition. Profile peptide nuisance
offsets using training weights only. No validation/test value enters fitting.

Controls usex=t/80. Mean and last-time-mean are per-peptide fitting anchors.
Global slope minimizes profiled weighted MSE. Shrunk slopes minimize profiled
weighted MSE +alpha*mean((slope_s-global_slope)^2), optimizing the shared slope
jointly. Solve its symmetric normal equations foralpha.001/.01/.1/1. All slopes
are identifiable from two distinct fitting times. No intercept penalty. Select
the direct comparator on validation RMSE, with deterministic declared model order
breaking exact ties; select preferred exact physical variant the same way.

Store initial and every post-update parameter vector, Adam first/second moments,
step, profiled intercept and dedicated Torch generator state in one compressed
NPZ per fit. This preserves every checkpoint without thousands of small files;
global RNG is unused by fitting. Save loss/gradient/runtime trajectories and final
covariate-only predictions before predictive outcome access. Report raw ratio
metrics and change relative to last fitting-time mean so sequence offsets do not
masquerade as learned temporal dynamics.

Qualification uses actual production functions: independent NumPy formulas;
all32cut states for a6-residue chain; full9,870-span residue-mass/fragment-count
identities at u=.001/10/100/240 and rates1e-7/.001/.1/1; finite differences of the
actual training objective for both models at initial and fixed sin-coordinate
perturbation(.2amplitude) parameter points. Absolute gradient error must be<=1e-6.
At the initial point, conditional gradient means from128independent repeats for
16and64draws must agree within6estimated standard errors per coordinate (1e-10
numerical floor); report relative error without adapting a pass tolerance.

Unconditional sampling uses fixed seed90210, budgets256/2048/16384, exposure
10/90/130minutes and initial uniform rates. Record all-span mean error, zero-hit
fraction for fitting peptide events and runtime. These are synthetic stochastic
process checks, never synthetic biological training labels. Every sampled cut
configuration conserves140residues. Conditional sampling forces query boundaries
and unbroken interior; tests explicitly verify query inclusion.

Evaluation uses recipe weights; Pearson/Spearman are unweighted. Bootstrap
1,000paired peptide resamples with seed31415 for physical-vs-comparator RMSE
differences, explicitly not specimen uncertainty. Preserve all predictions and
metric reductions. No model changes follow evaluation. Independent replay must
verify source/targets, every final prediction and all reported overall metrics.

Before qualification/fitting, implement the conditional batch through independent
Binomial(N,p) bond counts, forcing boundary counts toN and internal counts to0.
The complete-cut log probability is additive over bonds, so these counts are
exact sufficient statistics for the average score ofNindependent conditional
cut configurations: the estimator distribution and objective are unchanged.
A full Bernoulli-draw path is retained and explicitly tests forced event inclusion.
Report this compression and do not claim to have simulated continuous event times.
Each fit is a separately monitored stage with the unchanged20minute cap; total
physical artifacts must remain under128MiB. This execution detail predates all
numerical qualification and fitting and avoids thousands of small checkpoints.
