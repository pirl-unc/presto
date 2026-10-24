# Verification-only correction before replay

The first independent prediction verifier fails its ridge coefficient replay.
Its training weight variable `w` is rebound by the per-split metric loop; subsequent
models therefore reconstruct coefficients with the previous test weights. The
production fitter uses a separate `weights` binding and does not rebind it in
metrics. Preserve failed verifier/log, name its training weights explicitly, then
rerun all original gates. No source, fit, checkpoint, prediction, metric, split or
tolerance changes are permitted. A successful replay must numerically establish
that all archived coefficients equal the actual training-weight reconstruction.
