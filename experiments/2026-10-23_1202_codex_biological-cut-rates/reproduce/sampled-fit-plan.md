# Frozen single-seed biological hybrid fit

After training-only gradient probes, partial analytic integration lowers measured
RMS error at both tested states. At K512 it is0.268 of the initial gradient norm
at initialization and1.131 at the fitted state; final estimates remain noisy and
32 probe groups are too few to characterize rare-event tails. This supports a
bounded feasibility fit, not a low-variance guarantee. No selection based on
held-out model metrics: architecture/objective/initialization/optimizer remain the
fixed sequence_context reference, with the estimator as the sole fitted change.

Run one seed42, K512 unconditional full-protein partitions at each condition/time,
2,000 full-batch Adam updates lr0.03, regularization0.01/2 nonintercept squared,
exact same initialized coefficients and final-checkpoint rule as exact reference.
Use observed span occurrence counts with analytically integrated irrelevant bond
scores. Exact residuals, probabilities and conditional gradients remain necessary:
this is a partial analytic hybrid, not pure RL or a score-evaluation saving.
No clipping, standardized advantages, sample reuse, pseudocounts or hidden annealing.
Verify optimized gradients against the frozen prior probe draws at initial/final
states before any update. Keep checkpoints0/100/500/1000/2000 and all final
predictions. Stop on nonfinite gradients or exact-probability underflow, retaining
failure state; do not add an outcome-driven rescue. Preserve the full exact fit.
Single-seed result measures feasibility; no statistical superiority or robustness
claim. Subsequent same-cohort predictions are exploratory; no independent source
validation or concentration calibration is supplied by this comparison.
