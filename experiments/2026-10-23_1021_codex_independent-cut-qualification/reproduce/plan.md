# Independent-cut exact versus sampled qualification

Registered before implementation/fitting, 2026-10-23. Agent/model: Codex / GPT-6.
This is a mathematical and synthetic kinetic positive control, not a fit to biological data.
It implements the controlled-fragment-kinetics plan's independent irreversible cuts.

## Fixed process and hypothesis
An L=6 substrate has five independent exponential bond lifetimes, rates
[0.08,0.18,0.45,0.9,1.4] per hour. At time t a bond is cut with probability
1-exp(-rate*t). Every realization retains the complete partition, including
intact substrate and singleton fragments; no observed-set normalization.
Exact span amounts use the boundary-survival formula from the parent plan.
All 21 possible spans are observable with response 1, no noise, capture,
background, missingness, clipping, or empirical biological labels.
The outputs are expected fragment copies per initial substrate molecule.
Known response and initial amount remove the biological identifiability issue
for this qualification only. No claim that this process describes proteasomes.

## Mathematics gates
Compare the exact formula to independent enumeration of all 32 cut states,
for times 0,0.2,0.7,1.5,4,12. Check residue mass=6, fragment count=
1+sum(cut probabilities), partition mass=6 for every sample, and t=0 intact.
Compare autograd against independently enumerated analytic derivative for the
squared-mean objective. Exhaustively check the two-group score estimator's
expectation on all state pairs. Tolerance 1e-11 in float64.
Sample gradient probe at initial rates exp(log(0.3)+Normal(0,0.1)), seed42,
time1.5, 512 repetitions for groups8/32/128. Preserve every gradient and exact
reference; report RMS relative error, mean error, cosine and standard error.
No retuning of the specified matrix based on probes.

## Matched fitting
Train time points [0.2,0.7,1.5,4,12]; validation [0.4,1,2.5,8]; test
[0.1,0.6,2,6,20]. Splits test temporal prediction within the same synthetic
substrate, not generalization to unseen substrates or biology.
Exact and REINFORCE with two independent groups of32/128 per time,
seeds42/43/44, same initialized five trainable log rates. Adam lr0.03,
1000 fixed updates, all five times each step, no pretraining/regularization,
clipping/standardization/reuse/reference KL. Loss is mean squared error of
expected span amounts, all times/spans equally weighted.
The sampled estimator uses independent group cross-products of residual means;
per-draw linear costs and leave-one-out baselines produce the score gradient.
Both group contributions are required. No log-of-Monte-Carlo plug-in objective.
Save checkpoints at0,100,500,1000; final checkpoint selected prospectively.
Compare a fitted one-rate exact control (same1000updates) to test representational
necessity of bond-specific rates. Record all per-span train/val/test predictions,
rate recovery, MSE/RMSE/Pearson/Spearman, residue conservation, workcounts,
wall time and process memory. Binding cutoff metrics do not apply.

## Decision and closure
Gate for implementation correctness is mathematical agreement. Synthetic fit
criterion: final exact and sampled median relative rate error<0.1, test RMSE
below one-rate control. Report failures unchanged; do not increase updates or
change seeds to rescue. Neither success nor failure resolves biological MS
response, source selection, machinery, clearance, APC concentrations or
sequence-to-rate generalization. Freeze code/hash/commands before launch;
independent NumPy replay of checkpoints/predictions/metrics before closing.
