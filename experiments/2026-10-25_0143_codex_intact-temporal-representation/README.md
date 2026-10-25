# Temporal representations: measured changes in development CV

Codex / GPT-6. Registered 2026-10-25T01:43:29.754602+00:00 before execution at Presto
`d67fb48488960882481627c6f3a8bdaeb6e58bd3` with recorded task dirt and experiment-local code.
[Prospective specification](reproduce/plan.md), [exact launch/environment/input hashes](reproduce/launch.json).

The sequence-versus-length comparison **passes** its prespecified
development screen: RMSE gain **8.772%**.
Removing the physical rate penalty **fails** the separate 5% improvement
gate: gain **1.362%**. There are
**0 unsuccessful selected physical optimizers** among 15;
failed convergence remains reported and is not promotion-qualified.

These results concern nested development CV on previously inspected training
measurements. They are **not a new blind biological test**, independent-protein
transfer, an RL comparison, or evidence of calibrated APC concentrations.

## Data, conditioning and split

Exactly the original 0046 training cohort: 626 positive measured product/intact
substrate log10 intensity ratios, 96 peptides, nominal times 0–80 minutes.
PXD058121 F2 measures human alpha-synuclein (140 residues) with purified mouse-liver
20S; one source animal, unresolved replicate independence and unknown MS scaling.
Canonical recipe `pxd058121_product_substrate_forecast_v1`, manifest
`fb04bd482f5902d776bcd853d5280d86df7fd8b74424574c1bf793e3d2332f0c`, unchanged
source and exclusions. Shared data remain in the pinned prior-family paths.
No new curator, imputation, negatives, censor bounds, synthetic biological labels
or label-dependent filtering. Other assay families and IC50/binder metrics do not apply.

The first eligible time's replicate mean is supplied as a measured baseline for
each peptide. Predict later **change** in log product/intact ratio. Baseline cells
are not predicted targets; their exact IDs/values and reuse dependencies are saved
in [baselines](results/baselines.json) and [all changes](results/changes.parquet).
Equal peptide/time/replicate weights are recomputed in each split. Baseline noise
is shared across a peptide's later changes. No peptide identity or earlier fitted
slope is an input feature.

Five peptide folds use SHA256(`intact-temporal-representation-v1|` + sequence)
modulo 5; three inner folds use `intact-temporal-inner-v1|` modulo 3. Repeated cells
of a sequence stay together. Outer validation requires >=10 peptides, inner >=5;
all support decisions are in [preparation](results/preparation.json). Peptides
overlap in one parent: this is not a homology-disjoint evaluation. Original 100/120
minute outcomes are not read. Earlier PXD044463 test descriptive outcomes were
historically inspected, so that partition is not treated as a fresh external test.

## Model and fitting contract

Seven arms: zero change, shared elapsed-time slope, length-conditioned slope,
sequence-conditioned slope, uniform physical rates, residue rates with penalty
0.01, and residue rates without penalty. Length uses log length; sequence adds
20 amino-acid fractions and 20 indicators per terminus. Training-only unique
peptide means/scales and supported covariance eigenvectors remove constant/null
directions at the fixed relative 1e-10 cutoff. Prediction is elapsed time/80 times
a shared intercept plus the projected features' coefficients. The intercept is
unpenalized; ridge penalizes alpha times mean squared other coefficients. Inner
peptide CV selects alpha .001/.01/.1/1, with ascending-alpha ties and equal peptide
weight across inner folds. All 120 candidate fits and inner-validation predictions
are retained. No pretraining or sequence encoder is used.

Physical changes subtract the frozen production log product/parent prediction
at baseline time from its prediction at the later time, cancelling fixed peptide
offsets. Uniform has two parameters; residue has 17 category rates plus effective
pre-exposure tau. Bounds remain rates 1e-7–1/min and tau .001–120 minutes. Each of
15 fold/arm combinations uses three data-free starts (.001/min,tau10),
(1e-5/min,tau60), (1e-4/min,tau30). No initializer sees validation-peptide labels.
L-BFGS-B uses raw bounds [-30,30], maxiter1500, maxfun3000, maxls40, gtol1e-9,
ftol1e-13; choose the lowest training objective. The objective is balanced change
MSE plus alpha times mean squared centered log category rates. All 45 starts,
objective/gradient histories, final predictions, parameters and convergence
failures are retained. [Selection and parameter coverage](results/selected-models.json).
Gradient coverage does not imply biological identifiability.

This isolates representation/regularization after 0046 found matched exact and
conditional-score predictions without sampled-gradient benefit. No new RL claim
is made here. Affinity, stability, transport, MHC, TCR, APC machinery and detector
heads are not trained. A promising candidate still needs a frozen prospective
biological exact-versus-discrete comparison before advancing the full goal.

## Outcomes

Pooled outer-development errors and SDs are equal-peptide/time/replicate weighted;
Pearson and Spearman are unweighted. These are held-out peptide folds within an
already inspected development cohort. [All metrics](results/evaluation.json),
[fold training/validation CSV](results/comparison.csv), [peptide strata](results/strata.parquet),
[outer predictions](results/predictions.parquet), [inner predictions](results/inner-predictions.parquet).

| Model | Development CV RMSE | R² | Pearson | Spearman | Prediction SD | Target SD |
|---|---:|---:|---:|---:|---:|---:|
| persistence | 0.470580 | -0.693432 | undefined | undefined | 0.000000 | 0.361618 |
| shared_slope | 0.379389 | -0.100703 | 0.014003 | -0.006312 | 0.127201 | 0.361618 |
| length | 0.379042 | -0.098690 | 0.008941 | -0.004936 | 0.128035 | 0.361618 |
| sequence | 0.345792 | 0.085610 | 0.599177 | 0.599769 | 0.304500 | 0.361618 |
| physical_uniform | 0.386543 | -0.142609 | -0.012099 | -0.030995 | 0.149976 | 0.361618 |
| physical_residue_penalized | 0.386013 | -0.139474 | -0.008486 | -0.030941 | 0.152184 | 0.361618 |
| physical_residue_unpenalized | 0.380755 | -0.108643 | 0.045984 | 0.007165 | 0.165161 | 0.361618 |

Sequence promotion requires >=5% RMSE gain over length, positive Pearson and
Spearman, and prediction SD >=0.1 target SD. Physical regularization change requires
>=5% gain. Paired 1000-peptide bootstrap (seed31415) intervals and exact draws are
saved; they describe uncertainty within this parent, not independent specimens.
No threshold, split, features or tuning budget changes after observing outcomes.
If sequence fails, stop expanding this single-parent representation and seek
broader measured substrates/contexts. Either result requires fresh evaluation;
the full precursor-relative availability goal remains unachieved.

## Verification and resources

Independent source reconstruction checks baseline IDs/values, fold assignments,
prediction coverage and weights. Independent SVD projectors/augmented least
squares check direct fits and selection; NumPy physical formulas check all starts,
stored objective evaluations, final finite-difference gradients and selected
predictions. All overall/stratified metrics, paired bootstrap and fixed gates are
replayed. [Exact verification scope and errors](results/independent-replay.json).
L-BFGS internal line-search/update implementation is not independently replayed.

| Stage | Wall seconds | Peak polled RSS bytes |
|---|---:|---:|
| prepare | 1.887 | 294797312 |
| qualify | 3.419 | 303874048 |
| fit | 13.692 | 475496448 |
| evaluate | 2.555 | 458096640 |
| replay | 19.492 | 407289856 |

Local float64 CPU, one numerical thread; no Modal/GPU/installations. Fixed caps:
3GiB RSS,20minutes/stage,32MiB logical artifacts,1GiB free disk plus32MiB startup
margin. Launch each stage listed in launch.json using `sh reproduce/launch.sh STAGE`;
stage receipts prevent overwriting. Source snapshots are under `reproduce/source/`.
All stages were initially deferred below the disk reserve; no guard was weakened.
The artifact manifest hashes this report and retained outputs after closure.
