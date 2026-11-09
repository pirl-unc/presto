# Frozen biological execution contract

Frozen after all full-observation and objective checks, before implementation or
execution of fitting updates. Both arms get200 Adam updates, lr0.03 (the prior
biological optimizer scale), fixed seed42 and exactly the same probability-weighted
condition/time draws. Final checkpoint only. This is a bounded pilot, not a claim
that200 minibatches converge as well as the old2000 full-batch fit. All prior
controls remain visible with that distinction.

Common85 parameters:40 rate coefficients,38 shared response coordinates,seven
condition offsets. Both start with the original ratio-trained theta, zero response,
and identical weighted-optimal initial offsets. Fragment arm adds one raw scalar
with physical eta=tanh(raw_eta), initialized0 and constrained prospectively to
(-1,1). The existing0.01 L2 coefficient applies to nonintercept rate/response
parameters; the same coefficient penalizes physical eta. Offsets are unpenalized.
No clipping, floors, discarded rows, outcome-driven retry or schedule extension.

All selected required values must pass2/4-step log agreement before and after each
fragment update. A failure stops that arm and preserves the offending parameters,
optimizer and block. Final evaluation checks every training/development query.
CPUfloat64,one thread,4GiB and3600s per arm; resource failure is an outcome.

Primary comparison: matched final development shape RMSE/correlations and temporal
ratios, with full training loss, spread, coordinate update/support, amount/rate/
response ranges and numerical constraints. Development scale offsets use its
labels solely for the existing shape diagnostic. No original test evaluation or
concentration claim. Per-example predictions and checkpoints are mandatory.

Numerical gate and all explicit settings/schedule are in fitting.json.
