# Independent-cut kinetics: exact and sampled recovery

**Both methods recover the known synthetic rates, and their mathematical gradients agree.** This qualifies a matched kinetic process for later biological tests; it does not demonstrate that independent cuts explain real proteasome processing or APC peptide concentrations.

Registered and run October 23, 2026. Agent/model: Codex / GPT-6. Local CPU only, float64, one Torch thread. No Modal or GPU. All ten fits and 10,000 updates completed; final checkpoints were selected prospectively, without validation/test tuning.

## Model and synthetic data contract

A six-residue substrate has five independent irreversible bond cuts with true rates `[0.08, 0.18, 0.45, 0.9, 1.4]` per hour. The model fits five log rates directly. It has **no sequence-to-rate encoder yet**. Each cut's occurrence probability at time t is `1-exp(-rate*t)`. The exact model returns the expected number of copies of each of all 21 contiguous fragments per initial molecule. The sampled model returns complete partitions. Singletons, short nonbinders and intact substrate remain present; every partition conserves all six residues.

The teacher is noise-free, observes every span, and has unit detector response and known initial amount. There are no biological labels, missing products, background, censoring, capture losses or nuisance amplitudes. No MHC, affinity, stability, elution or TCR losses; binding-cutoff metrics do not apply. This intentionally removes the biological identifiability problem for a mathematical positive control.

Training times `[0.2,0.7,1.5,4,12]`, validation `[0.4,1,2.5,8]`, test `[0.1,0.6,2,6,20]` hours. Validation/test measure temporal prediction for the **same synthetic substrate**, not unseen sequences or independent biology. Every per-span prediction and target is saved in each run JSON (2,940 total values across ten fits).

## Fixed training comparison

Five trainable log rates, initialized around log(0.3) with normal SD0.1; seeds42/43/44 matched across methods. Adam lr0.03, default betas0.9/0.999, epsilon1e-8, 1,000 updates, all five training times per step. No pretraining, regularization, clipping, group standardization, sample reuse or reference KL. Loss: unit-weight mean squared error of expected fragment copies over all times/spans.

The sampled estimator uses two independent groups of32 or128 partitions per time. Their mean-residual cross-product is an unbiased estimator of squared mean error. Per-draw linear costs, both group score contributions and leave-one-out baselines estimate its gradient. No finite-sample logarithm/ratio plug-in is used. Individual loss estimates may be negative and are not clipped. The one-rate exact control shares a single rate across all bonds and has the same update budget.

| Method | Median relative rate error | Validation RMSE | Test RMSE | Median fit seconds |
| --- | ---: | ---: | ---: | ---: |
| exact | 1.542e-16 | 4.7256e-17 | 5.0678e-17 | 0.1745 |
| sampled32 | 0.03431 | 0.0071473 | 0.0063282 | 0.2809 |
| sampled128 | 0.016633 | 0.0031173 | 0.0027702 | 0.4626 |
| constant | 0.72036 | 0.16516 | 0.14176 | 0.1685 |

RMSE units are fragment copies per initial molecule. Values are medians over three seeds, except the single control; they are not biological confidence intervals. All three methods pass the preregistered rate-error<10% and test-RMSE-below-control criteria. Exact differentiation is the preferred reference while these small states can be evaluated analytically. Sampling works but retains finite-gradient noise.

The first exact fit took1.068s versus0.174s for later exact seeds, including one-time optimizer/framework startup; no explicit runtime warmup was performed. Runtime medians are descriptive. Entire checks/fits took4.261s, peak process RSS338.2MiB. Exact fitting evaluates105,000 span expectations per run. Sampled32/128 draw320,000/1,280,000 complete partitions per run; all methods compute25,000 bond probabilities. This is no evidence of a sampling compute advantage on such a tiny analytic model.

## Mathematical and independent verification

Exact amounts agree with exhaustive enumeration of32 cut states to1.11e-16. Expected residue mass error is zero; expected fragment-number error4.44e-16. Independent analytic loss gradients agree with autograd to5.20e-18; the exhaustive two-group score-gradient expectation differs by1.82e-17. Sampled partitions conserve residue mass exactly.

At the fixed initial point, 512 gradient estimates per group size yield RMS relative gradient errors0.788/0.344/0.170 for8/32/128 draws per group; mean-gradient relative errors0.0271/0.0267/0.00733. These characterize one point, not universal variance guarantees. Every gradient and its standard error is preserved in `results/gradient-probes.json`.

A separate verifier uses NumPy cut-state enumeration, analytic score derivatives and independently implemented Adam. Torch supplies only the same uniform RNG stream. It reproduces **all10,000 updates,30 noninitial checkpoints and2,940 prediction values**, maximum checkpoint log-rate error3.46e-14 and metric error1.93e-15. It imports neither the process nor training implementation.

The first verifier exposed rank instability in the one-rate control: mathematically equal spans acquired floating-point ties inconsistently. Original arrays/metrics and the failed verifier are retained. A documented post-fit reporting amendment computes equal classes from time, length and boundary count. Corrected control test Spearman is0.857106, versus original0.859143. No fit, checkpoint, RMSE or gate changed. See `reproduce/metric-tie-amendment.md` and `results/independent-replay.json`.

## Reproduction, scope and next decision

Exact plan, source hashes, git commit/dirty state, environment and invocation are in `reproduce/plan.md`, `reproduce/launch.json`, `reproduce/launch.sh` and `reproduce/source/`. Training producer is Presto12dc30e plus the explicitly frozen new experiment code; unrelated checkout work was preserved. Code and all small raw artifacts are within this directory. Logs preserve both verification attempts. Run only in a fresh copied experiment workspace with an empty results directory; the original launcher writes fixed result names.

The prior MAPP comparison normalized over observed peptide identities and could not identify total product amount. This model instead conserves mass and predicts unnormalized expected fragment copies, but its rate-recovery result is synthetic. Before a real comparison, use the shared digestion source, independently replay admitted preprocessing, define temporal response/selection conditioning, test repeatability and identifiability, and supply a parsimonious sequence-to-bond-rate function. APC machinery, transport, clearance and calibrated MS observation remain required for the full goal.
