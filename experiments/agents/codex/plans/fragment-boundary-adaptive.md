# State-adaptive boundary proposal for fragment availability

Registered as the next focused experiment after the completed uniform and
truncated-initial-boundary families. Agent: Codex / GPT-6. No implementation or
sampling under this plan has run yet. The full APC concentration objective remains
open; this is a necessary numerical comparison within it, not a replacement goal.

## Evidence and hypothesis

Uniform boundary times pass 45/45 ordinary and 2/3 rare primary checks. Initial-
fragment truncated-exponential times plus a proposal-score control pass 48/48,
but only 8/21 full-parent cases under the original mean-absolute-log criterion.
All seven eta=0 cases pass; only one of fourteen eta != 0 cases passes. Some
positive-eta groups have effective sample size below 3/512. All saved path density
and score calculations independently replay. These results support testing
proposal mismatch as fragments change, without establishing it as the only cause.

**Single change:** update each required boundary's proposal hazard after every
cut using its current containing fragment and remaining observation time. Keep
the physical process, support, score-control formula, exact references, targets,
counts, seeds, rate initializer and acceptance criteria fixed. No model-capacity,
curation, clipping, sample-reuse or reward changes accompany this comparison.

## Physical model and target

Retain all complete partitions of the parent and all fragment lengths 1..L.
Original hazard at uncut bond k in live fragment [a,b) is
`h_k = exp(log_rate[k]) * ((b-a)/20)**eta`. Both children remain. The requested
quantity is the expected count/probability of one source span [s,e) at time T.
Internal target cuts are forbidden only in the importance proposal, with their
original survival factors restored in the weight. Every other product remains
explicitly present; no MHC affinity or observation filter is applied to support.

## Proposal and exact path density

Let R=T-t. At each event state, outside non-boundary cuts have their original
constant hazards until the next event. Each required uncut target boundary has
proposal hazard `b_k(t)=h_k/(1-exp(-h_k*R))`. Draw its waiting time from the
exponential of rate h_k truncated to (0,R), draw the outside-event exponential,
and take the earliest event. If an outside or another boundary cut occurs first,
recompute all affected hazards and redraw pending boundary waiting times from the
new state. This defines a normalized time-inhomogeneous jump process and ensures
every required boundary occurs before T. There is no ordered-boundary factorial.

Between consecutive events separated by d, one pending boundary contributes

`log S_q = -h*d + log(1-exp(-h*(R-d))) - log(1-exp(-h*R))`.

Its derivative with respect to log h is

`-h*d + B(h*(R-d)) - B(h*R)`, where `B(x)=x/(exp(x)-1)`.

A boundary event at the end contributes
`log(h) - log(1-exp(-h*(R-d)))`, with log-h derivative
`1-B(h*(R-d))`. For fixed paths, multiply this derivative by
`log(fragment_length/20)` for the eta score. Outside event/waiting terms use the
original hazard and usual score. Include terminal survival for all remaining
hazards. Compute stable log normalizers and B with explicit small/large argument
limits; no probability clipping, hidden tail truncation or altered physical rates.
Boundary events exactly at T are zero-probability limits and must not silently
produce an invalid log density.

Retain original/proposal complete path log densities and scores. Importance
log-weight is `log_p-log_q`. Count mean and unnormalized score numerator remain
unbiased; log mean and self-normalized score ratio have finite-K bias. Primary
control stays `mean_i[w_i*score_p_i - mean_{j!=i}(w_j)*score_q_i]`, divided by
mean weight; retain plain scores as a paired diagnostic. Do not backpropagate
through random draws or mistake the proposal for the biological law.

## Registration and numerical checks

Before execution create a new timestamped directory with this plan, launch.json,
exact source/environment hashes, launchers and reference artifact hashes.
Use the unchanged 45 ordinary / 3 rare states from the prior family's launch.json:
K256/2048/16384, 16 groups, same seeds and same primary gates. Counts, seeds,
targets and gates must not be revised after results. Retain original exact and
both previous importance results as fixed references. Same seeds across different
random-draw algorithms do not imply paired-path coupling.

Test the single-bond truncated-exponential limit, intact parent, endpoint/interior
spans and eta=0 constant-weight limit. Prove the segment/event density algebra
against independent scalar expressions and directional finite differences at
fixed event paths. The small reference uses an independent partition generator
and Frechet gradients; the verifier must not import producer proposal code.
Reconstruct every saved path's proposal waiting/event contributions independently.
Keep 48 complete first primary groups and eight full histories per state; all
other groups retain the complete sufficient statistics. Preserve failed runs.

## Full-parent comparison, conditional on small-system qualification

Use the exact frozen 21 cases in
`2026-10-23_1920_codex_boundary-exponential/reproduce/actual-inputs.json`, including
original source/input hashes. Seven training conditions/six constructs, eta=-1/0/+1,
T24h, centered lengths1/9/40, intact-parent control, K64/512 and eight groups.
Use unchanged seeds and the same signed readout [1,-.5,.25]/1.75 in all 41 learned
parameter directions. No source quantities or development/test examples are read.
The fixed CRAM48 arrays/readout gradients already independently checked in that
family may be reused by hash; they are not observations or universal references
away from these states. Retain the full-output positivity limitation.

Primary K512 gates, each case:
- each target's absolute mean relative probability error <=0.10;
- each target's **mean absolute** natural-log error <=0.05;
- aggregate 41-coordinate mean gradient relative error <=0.20;
- aggregate gradient RMS relative error <=0.50.

Compute signed bias and mean absolute error separately. Record the mathematical
reduction in JSON and test a cancellation counterexample so one cannot substitute
for the other. Require all 21 primary cases to pass before proposing biological
use; preserve reference-unresolved and runtime-failed cases separately.

One CPU/float64 worker, one thread, 120s/4GiB per case, no cloud or environment
changes. Since old worker totals include reference construction, record sampling
and reference times separately and do not compare unequal totals as speedups.
Save all target/group statistics, 63 complete first K512 projected raw groups,
eight full event histories per target, exact source/hash/seed lineage and all
failures. Independently replay density/scores and all gate decisions.

## Biological next stage and stop rule

If this single proposal change passes both gates, freeze a separate bounded
biological comparison on the existing canonical shared-observation digestion
cohort. Keep the previous independent-cut model and simple controls; change only
the fragment-length interaction, and compare deterministic gradients with the
qualified sampled estimator on the same likelihood, initialization and schedule.
Every required observed span/time must have positive, adequately resolved log
amounts/gradients throughout that run; check all required outputs without dropping
rows, clipping counts or silently reverting eta. Current numerical targets alone
do not establish that requirement. Use existing training/development assignments;
the already observed test parents remain closed. Report all coefficients' update
and exposure coverage, predictive spread, held-out development error, cost and
MS-response/condition-scale ambiguity. Do not claim concentration calibration from
relative intensity agreement.

If the proposal fails, publish that failure and stop this proposal extension.
Do not grow a sample-count/seed/hyperparameter search on the exposed cases. Use
the evidence to decide whether a bounded deterministic fit is computationally
admissible while the sampled arm remains unresolved, or whether another explicit
numerical hypothesis is necessary. This does not narrow or complete the original
APC availability goal. APC machinery, species/context contrasts, transport,
clearance, independent non-MHC MS response and cellular validation still require
their canonical sources and separate evidence.
