# Positive observed-fragment means through boundary-conditioned full paths

Prospective plan, October23,2026; Codex / GPT-6. The complete-fragment model
qualifies mathematically; uniformization is too slow on long positive-eta
parents. CRAM16/48 fixes runtime and meets absolute count/gradient gates but
fails the separately frozen strict positivity gate at tiny amounts. Preserve
those failures; neither clip negatives nor use the exposed tests to select data.
This plan qualifies a positive stochastic estimator and its log-domain behavior
before a biological objective is admitted. It does not change the kinetic model.

## Exact proposal and likelihood identity

For a target fragment [a,b) at horizon T, final presence requires its internal
bonds to remain uncut and each nonterminal boundary bond to have been cut.
There are zero, one or two required boundary events. Draw each labeled boundary
event time independently uniformly on (0,T), then sort events by their sampled
times. Physical parent termini require no sampled event. Simulate all other
allowed cuts by the original continuous-time hazards between scheduled events;
disable target internal bonds and unscheduled target boundary bonds in the
proposal only. Apply each forced boundary cut at its scheduled time. The entire
parent partition, including all outside products, is still represented.

For a complete target-containing trajectory tau, the proposal shares all allowed
event densities with the original process. If m is the required boundary count,
its importance log weight is

`log w = m log T + sum_boundary log hazard(event) - integral forbidden_hazard dt`.

The forbidden set contains target internal bonds and boundary bonds before their
scheduled event. Include the integrated forbidden hazard over the final interval.
Never omit eta dependence or integrated waiting terms. The proposal is normalized
and covers every target-containing original path except measure-zero ties;
`E_proposal[w] = expected copies of that span per initial parent`.
The gradient numerator is `E_proposal[w * score_original_path]`. Original-path
scores include all events and all integrated hazards. Proposal parameters can
depend on theta, but samples/weights are detached for this likelihood-score
estimator; naively differentiating just the reparameterized forced times is not
the registered estimator. Keep log weights and logsumexp for rare products.

Uniform boundary times are the fixed initial proposal. A truncated-exponential
time proposal is a separate future amendment, not an adaptive rescue in this run.
Zero-boundary intact-parent probability/gradient should reduce analytically to
no-cut survival. The current unconditional sampler remains the fixed reference.

## Numerical qualification before biological fitting

1. Register a timestamped family and independently derive/replay complete path
   proposal density, original density, ratio and both score vectors. Include
   left/right terminal, interior and intact-parent targets, all eta=-1/0/+1,
   zero-time limits, boundary-order alternatives and terminal survival.
2. Use the existing five-bond fixture, times .5/2/8, fixed groups/seeds/K matrix.
   Compare positive means, log means and all parameter gradients against dense
   generator/Frechet references, preserving ESS, zero-underflow counts, bias,
   group variance and actual runtime. Distinguish the unbiased mean/gradient
   numerator from finite-K biased log-mean/self-normalized score estimates.
   Freeze exact counts, gates and seed construction before execution.
3. On eta0, use independent analytic log probabilities/scores to qualify rare
   target behavior well below the CRAM absolute-accuracy scale. Do not use an
   inaccurate near-zero rational estimate as ground truth for log errors.
4. Then qualify the same fixed actual training-parent inputs with target spans
   chosen independently of measurement values, including short, long and rare
   fragments. No biological validation/test access. Compare CRAM only where
   empirical relative precision is demonstrably adequate, and preserve unresolved
   rare positive-eta cases as such. A numerical absolute-error gate alone does
   not license a logarithm or claim relative concentration accuracy.

## Biological comparison to register only after qualification

Keep the previously canonical digestion data, source grouping, precursor/load
context and shared MS response. Compare unchanged eta0, exact/rational-assisted
fragment-state gradients and a positive sampled branch using the same observation
objective and support. All observed targets must remain: no filtering to exclude
unresolved predictions, pseudocounts, numerical floors or hidden probability
renormalization. If a candidate cannot score every required observation reliably,
retain its failed admission rather than narrow the biological claim.

The current positivity failure is not a conclusion that the underlying CTMC
has negative counts; it is a numerical approximation issue. Any proposed change
to deterministic admission must explicitly define error/interval semantics and
verify loss/gradient accuracy on every required observation without altering
targets. Counts per initial parent still require independent clearance,
machinery/context and observation calibration to support the full APC goal.
