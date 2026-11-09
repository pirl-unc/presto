# Variable-length availability policy: exact reference and lazy sampling

October22,2026; Codex / GPT-6. Previous goal turn was progress:24f1d2b verified
shared biological source sequences. Hitlist739 recipe and MAPP752 import remain
open; no biological labels may bypass those contracts. Work now closes a measured
computational gap in the candidate-policy architecture rather than repeating a
synthetic predictive benchmark.

## Hypothesis and model specification

The existing sixteen-token prototype evaluates all span logits before sampling.
Test whether a hierarchical start/length policy can sample and compute exact
selected log probabilities while scoring length candidates only at sampled starts.
It must preserve the same distribution and REINFORCE gradient as full enumeration.
This is necessary for full protein sequences; reduced reward calls alone do not
establish reduced network work.

Implement one small production module, models/peptide_availability.py. Input is a
literal amino-acid parent sequence and a declared numeric context vector. It emits
normalized probabilities over (start,length) intervals, or draws from that same
distribution. Length support is configurable, default1–40, and always respects
protein boundaries. No MHC-affinity gate: short and nonbinding products remain
candidates. Observability/search masks belong to downstream measurement models.
This is a product-distribution policy, not a validated physical protease simulator.

Use fixed amino-acid one-hot features (20standard+U/O, plus an explicit boundary
symbol for flanks), prefix sums for span mean composition, first/last/adjacent
residues, normalized length/position and the supplied context. Two small tanh
MLPs score starts and conditional lengths. Final scalar layers have no bias,
because a shared bias would cancel under softmax and be structurally untrainable.
Start base weights include log(number of valid lengths), giving a uniform span
measure when learned scores are zero. No species/cell identities are inferred
inside this computational primitive; callers must supply the audited context.

Exact enumeration is O(LW) and exposes every valid span. Lazy sampling computes
all O(L) start scores, samples starts, deduplicates them, and evaluates at most
O(KW) conditional length scores. Retain all repeated draws in the returned multiset;
only repeated network computations are deduplicated. Log probabilities stay
connected to current parameters; sampling choices are detached. A distribution
state is ephemeral and must be rebuilt after optimizer updates. No clipping,
standardization, reuse or KL-to-reference is added implicitly.

Document that chunked enumeration limits inference working tensors; merely
concatenating differentiable chunks does not eliminate training graph storage.
Report actual start-network rows, conditional-network rows including invalid
masked lengths, and valid span counts separately. Do not claim O(K) sampling or
full network savings while the start distribution still requires O(L) work.

## Verification before benchmark

Meaningful tests: exhaustive short-sequence span support and total probability;
zero-score uniform measure; exact composition/flank features at boundaries;
chunk-size invariance; invalid sequences/context/support rejection; selected lazy
log probabilities and gradients equal dense gathered counterparts; repeated draws
preserved; sampled frequencies agree within a fixed tolerance on a small fixture.
Test autograd for every parameter tensor under a nonconstant scalar objective.
Do not interpret that software check as biological feature exposure or training.

A separately registered local runtime family will use pinned real parent sequences
with declared bounded length strata and synthetic mathematical cost functions,
not assay labels. Compare dense sampling and lazy sampling on the same policy and
sample budget, with the exact distribution as reference. Freeze selection, seeds,
thread counts, warmup/measurement counts, code, runtimes and score accounting
before launch. It is a computational/gradient benchmark, not biological evaluation;
no predictive train/test metrics or concentration claims apply. Integrate the
policy into a biological recipe only after shared data admission is frozen.
