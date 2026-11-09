# Finite observations and multiset losses: next prospective comparison

October 22, 2026; Codex / GPT-6. Part of the active
[biological availability program](availability-learning-program.md).
This follows the completed 69-fit GRPO-ingredient family, which retained exact
enumeration and simple REINFORCE as references. No result is asserted here.

## Hypotheses and purpose

H1: a whole-multiset score-function estimator can optimize the same observable
distribution loss as exact differentiation, without an added entropy reward.
H2: naive sampled-histogram squared error includes sampling variance and is a
different objective; its apparent reward must not be mistaken for recovery.
H3: finite observed counts and detection bias weaken latent recovery even if
the optimizer is correct. A fully blind subset makes its mass unidentifiable
from conditional observed counts without additional assumptions.

This is an intermediate synthetic observation test, not the endpoint of the
active concentration goal. Continue to source-qualified endogenous biological
measurements after evaluating it. Total concentration/precursor normalization
is not identified by this conditional-count experiment.

## Fixed data and observation contract

Reuse the exact generic integer-symbol parents, context, features, teacher and
model from the prior two families: 96/24/24 parents, four contexts each,
384/96/96 rows, 70 spans per row, the same frozen parent splits. No real
biological sequences, molecular units, censor limits or assay outcomes.

Two known observation mechanisms, independently generated before fitting:

- Positive detection: `w(s)=0.05+0.95*(last_symbol/7)**3`, positive on all spans.
- Blind subset: the same weights times the earlier arithmetic compatibility
  mask. This is a deliberately incomplete observation scope, not a biological
  binding assertion.

Observed probabilities are `p_obs(s)=p_latent(s)*w(s)/sum(p_latent*w)`.
Generate 256 observed draws per row per mechanism using fixed NumPy seeds
20261023 (positive) and 20261024 (blind). The first 32 draws define the low-count
condition and all 256 the high-count condition. Preserve full counts, identities,
known weights and generation seeds. No observed-zero count is called a biological
negative. True latent distributions are available to the simulator and final
evaluation only; they must not be inputs to any training loss or checkpoint
selection. Training data objects expose features, weights and counts separately
from the evaluation oracle.

## Conditions: 60 fits

Cross two observation mechanisms × two count budgets × five methods × seeds
42/43/44. No condition selection after looking at held-out oracle values.

1. Exact conditional multinomial negative log likelihood (per-observation CE).
2. Exact squared error `sum_s (q_obs(s)-f(s))**2`, where `f` is the observed
   empirical distribution.
3. Whole-multiset REINFORCE with an unbiased estimator of that squared error;
   16 draws per generated multiset and four independent multisets per input.
4. The same unbiased multiset estimator with 64 draws and four multisets.
5. Naive sampled-histogram squared error, 16 draws/four multisets; diagnostic
   with a deliberately different expectation, not a matched optimizer.

All observation predictions use the same known weights:
`q_obs = normalize(q_latent*w)`. Retain uniform-latent and simulator-oracle
diagnostics without fitting. The oracle is a diagnostic, never a selection
candidate. Keep unobserved candidate identities in latent predictions.

## Whole-multiset objective

For K independent draws from q_obs, let n_s be counts in one generated multiset.
The unbiased cost is

`sum_s n_s*(n_s-1)/(K*(K-1)) - 2*sum_s n_s*f_s/K + sum_s f_s**2`.

Its expectation equals `sum_s(q_obs-f)**2`. It can be negative on a sampled
multiset, although the expectation is nonnegative. Four independent multisets
give a leave-one-out cost baseline per input. Multiply detached centered cost
by the sum of log probabilities of all K draws. Preserve multiplicity; the
ordered-draw and multiset log probabilities have the same policy derivative.
There is no extra entropy term: the distribution-matching objective already
defines the desired probability distribution.

The naive cost `sum_s(n_s/K-f_s)**2` instead has expectation
`sum_s(q_obs-f)**2 + (1-sum_s q_obs**2)/K`.
Verify both expected costs and gradients by exhaustive tiny categorical examples
before launch. Do not interpret the naive control against the wrong objective.

## Training, validation and metrics

Same two 83→32→1 Tanh MLPs and seed-matched initialization, Adam .003, zero weight
decay, batch 16, 60 complete epochs/1,440 updates. No pretraining, dropout,
clipping, group-SD standardization, reuse or reference KL. Those optional
ingredients have already been isolated; keep this observation comparison simple.
Minibatch seed 10000+seed, action seed 20000+seed; float32, deterministic CPU,
one Torch/BLAS thread, no GPU/Modal. Record logical draw/count accesses, dense
policy work, memory and actual time rather than infer efficiency from K alone.

Every 120 steps evaluate common held-out observed-count CE and select its minimum
for all methods; oracle latent probabilities never participate in selection.
Evaluate selected test once. Report the declared training loss as well as common
observed-count CE and TV/KL/RMSE/correlations to true observed and latent
distributions, entropy, invisible-subset mass error and context strata. Preserve
all per-example latent/observation distributions and complete count records.
No nM/binder cutoff metrics apply. Primary comparison is held-out observable
recovery and sampled-vs-exact squared-error fidelity; latent recovery is separate.

No hard success claim from a loose screen alone. For each mechanism/count budget,
report every seed and comparison with uniform, exact likelihood and exact squared
error. A lower set reward does not establish latent recovery. Three seeds do not
justify broad significance claims or posthoc hyperparameter tuning.

## Required verification

- Validate source identities, nested count budgets and count conservation.
- Prove the corrected and naive estimator expectations and policy gradients by
  enumeration; check multiplicity, independent group baselines and zero-weight
  masking. No zero-times-negative-infinity arithmetic.
- Demonstrate at least two distinct latent distributions with identical blind
  observation distributions. This tests the identifiability limit explicitly.
- Verify training/selection do not read latent oracle labels.
- Register the dated family and freeze all source/configuration before fitting.
- Independently reconstruct observation generation, metrics and checkpoints;
  preserve failures, complete validation/test dumps and source/environment
  receipts before closing the family and advancing biological-data work.
