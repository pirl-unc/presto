# Training-only Rao-Blackwell comparison

Frozen after the original complete-partition gradient probes, before evaluating
this new estimator. Hypothesis: irrelevant-cut score noise contributes to variance.
Use precisely the same fixed initial/final training states, K32/128/512,32 groups
and RNG seeds. Preserve prior estimates; require bit-for-bit full-score gradient
replay. For each observed span, integrate scores of bonds outside its existence
event to zero. The remaining conditional score is deterministic given presence:
boundaries cut, internal bonds uncut. Multiply that score by sampled presence/mass
and the exact residual coefficient. Sum all contributions. This is an unbiased
partial analytic hybrid with actual unconditional complete-partition samples;
retain exact probabilities/residuals and no compute-saving or RL-only claim.
No LOO baseline is needed for this occurrence estimator. Report its full gradient
vectors, mean/SE/RMS error against the same analytic reference. No parameter fits,
label edits or held-out evaluations. If useful, freeze a separate iterative
sampled fit budget from training-only runtime/variance before launching it.
