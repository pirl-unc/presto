# Truncated-exponential boundary proposal and normalized-proposal score control

Prospective extension of2026-10-23_1909 boundary importance. Uniform forced times
pass45 ordinary primary cases and2 rare-survival cases but fail fast boundary
events: K16384 effective sample size about35, probability error6.6% and log bias
.053. Preserve the entire uniform family and failure. This changes proposal
efficiency, not biological hazards, support, observation model or targets.

## Proposal and score contract

For each required boundary k independently sample a time from the exponential
rate `lambda_k=exp(log_rate[k])*(L/20)^eta`, truncated to(0,T). Sort labeled
times without adding a factorial. All other allowed full-partition events follow
the unchanged process; internal target bonds remain forbidden in the proposal.
The path ratio is original density divided by proposal density, including these
truncated-time densities. Log proposal time density is
`log(lambda)-lambda*time-log(1-exp(-lambda*T))`.
Its log-lambda score is
`1-lambda*time-lambda*T*exp(-lambda*T)/(1-exp(-lambda*T))`.
Contract that score with the boundary log-rate and eta's `log(L/20)` derivative.
Use expm1/log1p forms; no zero weights, probability floors or observation filtering.

Keep plain likelihood-score gradients as a reference. Add the unbiased
unnormalized-gradient control
`mean_i[w_i score_p_i - mean_{j!=i}(w_j) score_q_i]`.
The correction has zero expectation because q is normalized and the baseline is
independent of draw i. This is the proposal score, not the original restricted
process score; subtracting a constant baseline from the latter would be wrong.
Dividing the controlled numerator by the sample mean weight still introduces
finite-K log-gradient bias; measure it explicitly. At eta0 the proposal is the
exact conditional independent-cut path law, making weights constant and removing
irrelevant score noise through this control.

## Frozen comparison and gates

Register a separate family before implementation. Reuse every ordinary/rare
state, target, time, K,16groups and initial seed rule of the uniform family.
The forced-time uniforms are shared by seed; subsequent event streams may differ.
Compare means/log means, ESS, original and controlled gradients, numerator
unbiasedness diagnostics, runtime and linear underflow. Candidate primary
gradient estimator is the proposal-score control; retain plain scores as a
reference. Keep prior5% mean probability/gradient, .02 absolute log-bias and25%
RMS gradient gates atK16384. Full path/density/scores must independently replay;
all observed/unobserved product support is unchanged.

Only after these gates pass, freeze a bounded actual-training-parent input
qualification with no measurement targets. Include rare and short products,
long parents and eta0 analytic references; relative log/count precision rather
than arbitrary absolute errors must qualify any later biological logarithm.
Do not silently promote CRAM's strict positivity failure or drop difficult data.

No training/pretraining, biological targets, test evaluation or calibrated APC
claim at registration. Preserve all source/code hashes, group sufficient
statistics, raw representative groups, full path traces and failed attempts.
Actual biological comparison still needs a separate frozen admission/objective
contract retaining the canonical digestion cohort and shared observation model.
