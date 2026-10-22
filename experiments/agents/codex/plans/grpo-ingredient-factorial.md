# GRPO ingredients and partial enumeration: prospective experiment

October 22, 2026; Codex / GPT-6.
Family: `experiments/2026-10-22_1120_codex_grpo-ingredient-factorial/`.
This is the first step of [the active availability program](availability-learning-program.md),
not a biological concentration fit. Follow with partial-observation/multiset and
real-source families; do not substitute this control for that end state.

## Hypotheses

- H1: current leave-one-out REINFORCE matches the exact gradient in expectation;
  group standardization changes that estimate and may change recovery.
- H2: clipping is inactive on a fresh on-policy first update, but can affect
  reused samples; its interaction with reuse must be measured, not attributed
  to a bundled optimizer name.
- H3: a reference KL term changes the optimum. Evaluate both fidelity to the
  original target and the explicitly regularized target.
- H4: enumerating conditional lengths while sampling starts provides a useful
  variance/work tradeoff. Report target terms and dense model work separately.

## Fixed data, model and endpoint

Reuse the exact integer-symbol dataset, teacher, features and policy of
`2026-10-21_1700_codex_discrete-distribution-comparison`, frozen launcher SHA256
`e66c3a5cffc0d512989094ad653ba7ea4f3ff56778d3b298fecb8e809eee5945`.
144 length-16 parents, alphabet 0–7, all 70 spans length 1–5, four generic contexts;
96/24/24 parent split gives 384/96/96 rows. Data/teacher seeds 20261021/20261022.
Parent disjointness does not imply short-substring independence. No biological
sequence, concentration, assay, censoring or binding classification is present.
Compatibility and selection factors remain arbitrary evaluation-only quantities.

Two 83→32→1 Tanh MLPs predict start and conditional length. Scalar output biases
remain disabled. Freeze the initial policy for each seed as the reference;
matched conditions share initial/reference weights and input ordering.

For beta in {0, 0.1}, the declared objective is
`KL(q || p_target) + beta * KL(q || q_reference)`.
Its unrestricted optimum is proportional to
`p_target**(1/(1+beta)) * q_reference**(beta/(1+beta))`.
Report losses and distribution metrics against both this optimum and the original
target; an improvement in a changed objective is not improvement on the old one.

## Conditions

- Full 2×2×2×2 factorial: joint probability-ratio clipping off/on (epsilon 0.2),
  group standardization off/on (population SD + 1e-6), reuse 1/4, beta 0/0.1.
  All use eight actual start/length draws per row, leave-one-out cost centering,
  and seeds 42/43/44: 48 fits.
- Exact enumeration at each beta: six fits.
- Original frozen REINFORCE-8, beta zero, no reuse: three fits, retained unchanged.
- Hybrid: sample 2 or 8 starts per row and enumerate every valid conditional
  length, beta zero/no reuse, three seeds each: six fits.
- Separate query-matched reuse comparison: reuse=4, standardization off, beta=0,
  clipping off/on, 5,760 optimizer updates each, three seeds: six fits. Compare
  against the 1,440-update original REINFORCE baseline at the same number of
  sampled target lookups; retain its fourfold optimizer-work cost explicitly.

Total 69 fits. No extra conditions or seed selection based on test outcomes.

## Estimator details

Joint span probability ratios, not token-wise language-model ratios. Cache the
sampling-policy log probability and sampled target/reference terms for each
collected batch. On each reuse step recompute current density cost
`C = log q - log p_target + beta*(log q - log q_reference)` and leave-one-out
baseline from the other draws. Detach centered costs; optionally divide by
within-row population SD of current costs plus epsilon.

Unclipped surrogate is mean `ratio * detached_advantage`, with
`ratio = exp(current_log_q - sampling_log_q)`. The clipped cost surrogate is
the maximum of that term and `clamp(ratio, 0.8, 1.2)*detached_advantage`.
This is a GRPO-style ingredient comparison with recomputed density costs, not
canonical fixed-reward/frozen-advantage GRPO. Unclipped, unstandardized importance
gradients target the declared objective for a fixed evaluation policy; adaptive
updates on reused finite samples can still bias learning. Clipping and empirical
standardization need not preserve the exact gradient. Measure these differences.

Hybrid: sample starts, calculate conditional probabilities and costs for all
valid lengths, center with other starts' conditional expected costs. Sum the
score-function terms over lengths using detached conditional probabilities.
Mask before multiplying so zero times negative infinity never enters a loss.
Enumerate tiny distributions in tests to verify unbiased expectation.

## Budgets and selection

Adam 0.003, zero weight decay, batch 16, float32, no dropout/pretraining/early stop.
Primary budget is 1,440 optimizer steps (60 batch-size-adjusted passes). Reuse 4
has 360 collected batches, or 15 source passes; reuse 1 has 1,440 collected
batches, or 60 passes. These are matched optimizer budgets, not matched data
refresh or oracle budgets. Extra query-matched conditions use 1,440 collected
batches/5,760 updates. Record all counts and actual runtimes.

Model seeds 42/43/44; minibatches 10000+seed, sampled actions 20000+seed.
Validate every 120 optimizer steps and at completion, selecting minimum declared
validation objective. Evaluate test once after selection. Never change recipe
from held-out results. No GPU/Modal: local CPU, one Torch/BLAS thread,
deterministic algorithms, environment/versions and actual source hashes captured.

## Gradient probes, controls and outcomes

Before fitting use the first four training rows, seed-42 initial policy and a
fixed deterministic parameter perturbation (0.05 * sin(flat parameter index))
to define a second policy for off-policy probes. Both are independent of fit
outcomes. For beta 0/0.1 and the four clipping/standardization combinations,
compute 128 independent gradient estimates against the exact gradient, at the
sampling policy and at the perturbed policy. Also probe original REINFORCE and
hybrid budgets. Report bias/error of average, single-estimate error/cosine,
summed variance, importance-ratio range, clip activity and logical target terms.

Save per-example full distributions for all splits and conditions, input/target/
reference identities, every validation history, parameter update coverage,
checkpoints and first-four-training-row replay. Report original and regularized
KL, TV, probability RMSE/Pearson/Spearman, entropy, noncompatible mass, context
strata, effective sample size, clipping/normalization diagnostics and costs.
No log10(nM) or 500 nM metric applies. Recovery screen is all three seeds with
validation TV to the objective optimum <=0.10 and noncompatible-mass MAE <=0.05;
descriptive effect comparisons do not imply statistical significance.

Before launch verify unbiased categorical and hybrid expectations, first-step
clipping identity, changed-policy importance correction, frozen reference,
regularized optimum and true reuse accounting. Freeze all executed sources and
the prospective spec before fitting. Afterward independently recompute held-out
metrics from dumps, check receipts and reload selected checkpoints in a fresh
process. Preserve failed conditions and close the family before promotion.
