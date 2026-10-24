# Exact and sampled learning of synthetic span distributions

Prospective contract, October 21, 2026. Agent: Codex / GPT-6.
Family: `experiments/2026-10-21_1700_codex_discrete-distribution-comparison/`.

## Question and scope

Can both exact differentiable marginalization and an actual discrete
policy-gradient learner recover the same nonuniform distribution over spans?
Compare inference fidelity, generalization, gradient variance, reward queries,
runtime and sampled multisets. This is a generic algorithm experiment using
integer-symbol strings, not amino acids, biological processing, genes or real
antigens. It does not establish either method for Presto's biological task.

The toy target is deliberately available as an oracle. This separates whether
the sampling method works from whether the biological target can be identified
from available observations. There is no pretrained model, real assay, molecular
unit, censoring, IC50 threshold, MS calibration or peptide-binding inference.

## Frozen data and model

- Alphabet: integers 0–7; parent strings length 16; all spans length 1–5.
  There are 70 supported spans per parent. No generated biological sequences.
- 96/24/24 independent parent strings in train/validation/test, each under the
  four generic contexts `(−1,−1), (−1,+1), (+1,−1), (+1,+1)`: 384/96/96 rows.
  NumPy data seed 20261021; unique parent strings; all contexts of a parent stay
  together. Short substrings intentionally overlap: this evaluates new parents,
  not unseen symbols or novel short-span generalization.
- Candidate features: first/last symbol one-hot (8+8), symbol frequencies (8),
  length/5, squared normalized length and normalized start (3): 27 features.
  Concatenate features, features times each context coordinate, and both context
  coordinates: 83. Start features are the valid-length mean for that start.
- Teacher: fixed independent normal coefficients (SD 1.25) for start and length
  logits, NumPy seed 20261022. Normalize starts then valid lengths conditional on
  start. Publish coefficients, inputs and exact target probabilities.
- Student: independent start/conditional-length MLPs, each Linear(83,32), Tanh,
  Linear(32,1); masked softmax; no dropout. The exact and sampled methods have
  identical model structure and matched starting weights for seeds 42,43,44.
- An arbitrary compatibility tag is true for length >=3 and
  `(first+last) mod 3 != 0`. It does not affect training reward/support.
  Noncompatible products can have positive target mass. Report their retained
  probability as a test of distribution learning, not binding prediction.
- A fixed toy observation bias `0.2 + 0.8*(last+1)/8` defines a secondary selected
  distribution proportional to product mass × compatibility × bias. This is
  evaluation-only arithmetic with known factors, not a learned MS/pMHC module.

## Conditions and objective

1. Uniform over the 70 valid spans; no fitted parameters.
2. Exact reverse KL `sum q*(log q - log p_target)`.
3. Discrete REINFORCE with 8 sampled spans per row.
4. Discrete REINFORCE with 64 sampled spans per row.
5. Seed-42 diagnostic: REINFORCE with 64 samples and no entropy term.

Conditions 2–4 use all three matched seeds and the same target objective.
Sample an actual start and then an actual conditional length; return the exact
substring and preserve multiplicities. Removing duplicate samples changes the
frequency distribution and is not allowed in the training estimator.

For reverse KL, the score-function surrogate is the mean of detached
`(log q(sample)-log p_target(sample)-baseline)` times `log q(sample)`.
The baseline is the mean cost of the other draws for that same row (leave one
out), independent of the selected action conditional on the row. The expected
extra derivative of log q is zero. Verify this against enumeration before any
fit. The no-entropy diagnostic removes log q from the cost and intentionally
changes the objective; it is not a matched contender.

Both learners compute policy logits for all spans in this prototype. Sampling
reduces logical reward lookups, not the full policy computation. The teacher
dataset is precomputed for both methods; counts describe target terms consumed
by fitting, not calls to an expensive external oracle. Count fitting lookups
separately from evaluation/probe lookups. No speedup claim based on counts alone.

## Training and selection

CPU, one Torch/BLAS thread; float32; no GPU/Modal. Adam with learning rate 0.003,
no weight decay; batch 16; 60 complete epochs (1,440 updates), no early stopping.
Matched minibatch permutations across methods for each seed. Independent sampling
RNG derived from model seed. Evaluate full validation every five epochs and pick
minimum exact reverse KL (including the no-entropy control). Test once after
selection. No hyperparameter changes or test-driven extensions within this family.
Record actual versions, source hashes, commit and dirty diff, hardware and clocks.

Before fitting, measure training-only gradient estimates on the same fixed
four-row batch and seed-42 initialization: exact gradient, 128 independently
sampled estimates at each sample budget, gradient cosine/error/variance and
the error of their average. This is part of the registered family.

## Metrics and prospective screen

Save per-example train/validation/test target and predicted distributions for
every selected checkpoint and the uniform baseline. Metrics: mean reverse KL,
forward KL, cross-entropy, total variation, Jensen–Shannon, entropy, probability
RMSE/Pearson/Spearman, noncompatible-mass error and toy selected-distribution TV.
Report target/predicted nonconstant spread, per-context results, parameter-update
coverage, runtime and fitting oracle queries. Save validation history and sampled
64-element test multisets for each selected model, without reselecting models.

A method passes this bounded recovery screen if all three seeds have validation
reverse KL <=25% of the uniform baseline, mean TV <=0.10 and noncompatible-mass
MAE <=0.05. Test is descriptive. A successful sampled method establishes basic
discrete distribution learning; this experiment does not establish biological
processing, real-data performance, exact resume, or a compute advantage.

## Verification and closure

Before launch: meaningful tests for masks/normalization, source split identity,
sampled support, complete multiset reconstruction, noncompatible mass, exact
finite-difference gradients and the estimator's exact expectation on a tiny
categorical problem. Explicit Ruff checks despite experiment exclusions.
Freeze executable sources before fitting and preserve launcher/env receipts.

Afterward independently recompute metrics from prediction dumps; reconstruct
teacher probabilities from saved input/coefficient data; replay selected
checkpoints in a fresh process on fixed training rows; check target/initial-state
equality across matched methods. Preserve every failed attempt and negative
result. Close README, prediction/artifact hashes, plots/tables and canonical log.
