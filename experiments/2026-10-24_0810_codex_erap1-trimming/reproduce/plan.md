# ERAP1 trimming: physical contribution beyond a sequence-response baseline

Registered before implementing or fitting this new model. This is a mechanism-focused
step toward the full APC availability objective, not a redefinition of that objective.
The previous 0730 turn made progress: verified HAP1 sequence signal and published it.

## Hypothesis and data boundary
Does a sequential N-terminal trimming model predict ERAP1 KO responses beyond the
fixed direct sequence model? Compare exact and sampled gradients under the same
physical model and objective, with a uniform-rate physical control.
The original admitted paper identifies ERAP1 as this panel's direct N-terminal
trimming enzyme. Do not reinterpret other KOs as cleavage-rate changes.

Use only ERAP1 rows from the identical 0730 complete-precursor OUTER TRAINING cohort:
8281 observations,3304 peptides,1296 components,1514 candidate parents,3409 distinct
occurrences. Source/recipe manifests remain88cad10d…/a1eec392…. Source metadata-only
inventory before this registration finds1741916 possible N-extended birth paths,
maximum extension5461 residues; every reached parent uses the standard20AA alphabet.
No outcome was fit or scored during that inventory.
Original molecular components and five inner folds remain fixed. All controls and
penalty0.1 direct baseline use their already frozen full/fold0730 predictions;
subset evaluation renormalizes original weights to ERAP1. The original gamma fits
from all ten training KO categories are frozen for each matching fold, so there is
no held-component target leakage. Preserve every target replicate; no new labels,
imputation, clipping, source curator or treatment/control cross-product.
Outer validation/test remain unopened for fitting or predictive scoring. No old
MAPP/native/macrophage held-out is reused.

## Physical model and assumptions
Use two explicit compartments, with dimensionless time units:
1. Cytosolic source proteins enter at unit flux. Independent irreversible bond cuts
have fixed rate1, and every fragment leaves the source compartment at rate1. The
existing exact steady-fragment formula gives each contiguous span's expected amount
F(s,e). Interpret its unit-rate exit as input flux J(s,e)=F(s,e) to the trimming
compartment. This is an uncalibrated uniform physical reference, not a learned TAP
selectivity model. All spans1..L remain supported, including nonbinding/short products.
No external pretrained rate prior is imported; therefore no cross-dataset prior
exposure is introduced. Independent proteasome/TAP rate calibration remains open.
2. In the trimming compartment, every span clears at rate1. For length>=2 it can
lose its first residue and become the next shorter span with the same C terminus.
A monomer remains a supported product, with trimming rate0 and clearance1. The
removed free residue has its own mass-flow ledger in software checks. Trim rates
are either one uniform rate or20rates indexed by the removed N-terminal residue.
WT rates t_a=exp(theta_a). KO rates=kappa*t_a with kappa=sigmoid(z). This is a
fitted effective reduction, not an assertion of zero residual enzyme activity or
absence of compensation/other aminopeptidases. No measured catalytic rate is supplied.
No additional length-specific rate or inferred substrate cutoff is introduced.

For a queried span starting at s and ending at e:
A_s = [J_s + t_(s-1)*A_(s-1)]/(1+t_s), with no incoming trimming at s=0.
Equivalently A_s = sum_(j=0..s) J_j * product_(k=j..s-1)[t_k/(1+t_k)] /(1+t_s).
This enumerates every upstream birth span ending at e, without constructing the
quadratic all-span table. Counts of removed residues summarize each unique path;
they preserve exact path probabilities for the stated residue-only rate model.
Source occurrences and candidate parents are all retained. Equal parent input is
an assumption. KO input for a parent is10^(gamma*its source protein log10 response),
using the matching frozen0730gamma. Sum amounts over all origins before ratios.

Physical response r=log10(sum A_KO / sum A_WT). Physical-only prediction is b+r.
Augmented prediction is frozen sequence baseline + b + r - gamma*x, where x is
that baseline's equal-parent mean log10 protein response. This replaces rather
than double-counts the baseline's precursor term. The additive correction cannot
change the baseline coefficients. Profile b as the fitting-weighted mean residual
at every step; this adds no peptide-specific free offsets.
The elution interpretation assumes unchanged same-peptide observation/selection
factors for the physical-only arm. The frozen direct baseline captures an empirical
observation/selection response in the augmented arm. Neither factorization identifies
causal rates or molar intracellular concentration from elution alone.

## Exact versus sampled estimator
Compute exact log-sum-exp over all birth-span/origin paths. The sampled arm retains
that exact forward amount and loss residual, then substitutes a score-function
estimate for its gradient. Sample paths from their normalized exact contribution,
using paired WT/KO uniform quantiles; score=logJ+sum log survival-log clearance/trim
hazard. Choices are detached. This is a conditional-score hybrid, not simulator-only
RL or GRPO. No clipping, advantage standardization, sample reuse or reference KL.
Count scored paths and measure actual elapsed time. Numerical gates use the data
objective gradient without regularization, to avoid masking estimator error.

Before fitting: analytic small-parent recurrence/linear-system/mass-balance tests,
exact gradient finite differences and exact enumeration of the sampled-score
expectation. Three fixed real-data parameter probes (zero and +/-0.2*seed42normal)
with eight estimator repetitions each. Try K128, then K512 only if128fails. Pass
all probes with relative norm(mean sampled gradient-exact)<=.10 and RMS relative
error<=.25. Freeze the first passing K. If neither passes, retain the failure and
do not run sampled biological fits; complete the exact biological comparison.

## Fixed fit matrix and evaluation
Four arms, each on full training plus five fixed inner folds:
- uniform physical-only, exact (one log rate plus z);
- residue-dependent physical-only, exact (20 log rates plus z);
- frozen-baseline plus residue physical correction, exact;
- same augmented model, qualified sampled gradients.

Float64 localCPU, one numerical thread; no cloud/Modal/installs. All theta and z
start at0. Adam,150full-batchsteps,lr.03,betas(.9,.999),eps1e-8,seed42,final checkpoint.
Objective is ERAP1-renormalized source-weighted MSE plus.01*mean(theta^2)+.01*z^2;
profiled offset is unpenalized. No restarts/schedule/selection based on held-out data.
All allocated trimming coordinates must have fitting-path support; record parameter
updates, gradient norms, data/penalty terms, physical correction variation and
kappa. No claim that every broader architecture branch trains.

Report train and pooled inner-CV MSE/RMSE(log10ratio),MAE,R2,Pearson,Spearman,
per-component scores, physical correction SD, and all per-observation predictions.
No nM/binder metric applies. Keep the original seven0730controls on this matched
ERAP1subset. No external/test validation is performed in this development family.
Development gate: augmented exact improves innerCVRMSE>=2% over frozen direct
sequence baseline, with physical correctionSD>.001. Physical-only results diagnose
whether the mechanistic structure alone explains the response. Any failure remains
a biological result; do not replace it with numerical sampler qualification.

## Runtime, artifacts and closure
Build+qualification cap900s/4GiB. Per-fit cap600s exact/1200s sampled,4GiB; whole
24-fit comparison cap7200s. Preserve every attempt; a breached cap is reported,
never silently relaxed or used to restart under a different biological contract.
Freeze launch/config/code, checkpoints every50steps, objective/gradient traces,
source/model input tables, parent/path inventories and all prediction dumps.
Independent NumPy scalar recurrence replay on every saved prediction, coefficient
and metric closure, plus source/weight/split audits. Optimizer trajectories are
preserved but not claimed independently replayed. Add a comparison plot, narrative
README and canonical registry entry. Full availability goal remains active.
