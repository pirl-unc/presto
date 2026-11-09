# Qualification of the bounded fragment-fit observation objective

Companion to `fragment-observation-domain.md`; written while the full numerical
admission is running, before any new fit or measured loss inspection. This does
not authorize fitting until every required numerical case is admitted.

## Preserve the endpoint and identify the trainer change

For training condition c, let w_i be the existing equal-parent/condition/curve/time
weight, y_i the unchanged log10 MS signal, a_i(theta,eta) the log10 expected fragment
count, and z_i the unchanged 38-dimensional response projection. Define

    L(theta, eta, gamma, beta) = sum_i w_i [a_i + z_i gamma + beta_c(i) - y_i]^2
                                  + 0.01/2 [||theta[1:]||² + ||gamma||²]
                                  + a separately declared penalty on eta, if used.

There is one beta per training condition, constant across peptides and times.
There are no per-peptide/curve intercepts, observed-set normalization, imputed
negatives, or absolute concentration labels. The 40 rate coefficients, 38 response
coordinates and seven condition offsets are allocated only after support checks.
The length-dependent arm adds exactly one global scalar. Future APC machinery,
transport and clearance remain outside this purified-digestion fit.

At fixed theta/eta/gamma, the unique weighted-least-squares nuisance solution is

    beta_c* = sum_{i in c} w_i [y_i - a_i - z_i gamma] / sum_{i in c} w_i.

Substitution gives precisely the prior within-condition centered objective.
At beta*, partial theta/gamma/eta derivatives of L equal derivatives of the
profiled objective; the beta derivative vanishes. This equality does not imply
that finite-step joint SGD matches full-batch profiled optimization. Both new arms
must use the same joint trainer; earlier profiled controls retain their distinct
optimizer contract in the report.

## Unbiased condition/time minibatches

For each of the 28 training condition/time blocks b, fix p_b = sum_{i in b} w_i.
Since sum_i w_i = 1, these are sampling probabilities. The stochastic data term is

    L_b = sum_{i in b} (w_i / p_b) [a_i + z_i gamma + beta_c(i) - y_i]^2.

Then sum_b p_b L_b equals the full data objective, and the same identity holds for
all parameter derivatives. Add the regularizer once per update, not once per
observation or once per condition. Do not divide again by row count. Duplicated
biological replicate observations retain their original distinct source identities
and weights; deduplicated physical span outputs are gathered back to every row.

## Required qualification before the fit execution receipt

1. A small numerical fixture proves profiled/joint loss and envelope-gradient
   equality at beta*, including nonuniform weights and unequal numbers of time
   points. A deliberately perturbed beta must raise loss by
   sum_c weight_c*(beta_c-beta_c*)².
2. On the unchanged actual training initializer, use the analytical eta0 model
   to verify the same equality for the existing 90,630 targets, all supported
   rate/response coordinates, and all seven offsets. This accesses measured
   training targets only after numerical admission; development quantities are
   not used for schedule or initialization selection.
3. Enumerate all 28 blocks and probability-weight their data losses and gradients;
   compare to full-batch analytical loss/gradient. Include the regularizer exactly
   once and explicitly test an unequal-block fixture. Record all block probabilities,
   row identities, source/feature hashes and numerical discrepancies.
4. Verify common eta0 predictions and theta/gamma/offset gradients between the
   analytical and fragment arms on a small fixture and a declared actual block.
   Do not claim that a zero eta gradient is expected: the differentiable model
   must remain sensitive to changing fragment length even at eta0.
5. Only after these checks and full numerical admission, freeze the fit execution
   receipt: eta parameterization/range/penalty, optimizer/groups/LR/steps, seed and
   exact shared block schedule, timeout/RSS limits, initialization, logging and
   final-checkpoint-only policy. No data-dependent schedule extension or exposed
   test rescue. The baseline and length-effect arm receive the same draw sequence.

## Reporting and stopping

Check all selected required values with the registered 2/4-step log tolerance on
every fragment update; an unresolved query stops and preserves the failure.
Never discard a biological row or restore eta to zero to continue. Save parameters,
optimizer state, gradients/coordinate support, actual timing/memory and the
failure-producing block. Final train/development evaluation checks every required
query, not only blocks visited during training. Save source-linked predictions
and all existing shape/temporal-ratio metrics, coefficient ranges and update counts.

Development offsets, if profiled for the existing scale-invariant metric, use
measured development signal; label that explicitly. They are not deployable
absolute signal predictions. Neither nuisance response coefficients nor condition
offsets establish MS detectability or APC concentration. Numerical feasibility,
optimization progress, biological generalization and calibration are separate
claims and require separate evidence.
