# Intact-degradomics forecast: direct regression wins; fragmentation fails

Codex / GPT-6, October 25, 2026. Registered before materialization and fitting at
Presto `3ab625edce9ceb6aed9d94ee104306dcf84e248c`, with recorded preexisting task-file
dirt and new experiment-local code. Local float64 CPU, one numerical thread;
no cloud, GPU, pretraining or environment installation.

**The mechanistic model fails its prespecified biological quality screen.** Its
validation-selected residue-conditioned variant has test RMSE **0.359038**, versus
**0.228184** for the selected direct temporal regression: **57.35% worse**. Exact
and conditional-score training reach essentially the same unsuccessful predictions.
Sampling takes approximately **1.8–2.5 times** the core fitting time.

Direct regression improves on persistence, but high raw-ratio correlations mostly
reflect previously observed peptide levels. After subtracting the last fitting-time
mean, test Spearman is only **0.23071** for direct regression and **-0.13886** for
fragmentation. Neither demonstrates strong peptide-specific temporal prediction.
This is a same-peptide forecast, not new-substrate transfer or APC concentrations.

## Measured data and shared admission

PXD058121 F2 kinetics: human alpha-synuclein, 140 residues, digested with purified
mouse-liver 20S. The source study uses one animal; independence of reported R1–R3
replicates is unresolved. Nominal zero already contains products. Species/material
metadata are supplied context, not learned predictors here.

Canonical source: `data/intact_degradomics_canonical/2026-10-25_0021_codex_intact-canonical`,
manifest `fd91bb6ff3178a2246b760f6b0f99bc3824d83815a65719c34587816f3459974`.
Shared recipe: `pxd058121_product_substrate_forecast_v1`, package
`data/intact_kinetic_recipes/2026-10-25_0046_codex_intact-kinetic-forecast`, manifest
`fb04bd482f5902d776bcd853d5280d86df7fd8b74424574c1bf793e3d2332f0c`.
Paths are relative to the Presto worktree. Runtime is the frozen Hitlist 2.0.34
development checkout on `97d3311a239cd901aa11b8ff8b627a7ce910cbd4`, not a claim of
release. Changed files, diff and hashes are in [reproduce/](reproduce/).

Admission requires F2 kinetics, positive finite assignment intensity, valid span,
one reported and replayed assignment, absolute mass error <=10 ppm, explicit
unmodified status, and input-mass identity/intensity agreement. All ambiguous
duplicate sequence/acquisition masses are excluded. A unique eligible intact
parent supplies the same-acquisition denominator; the parent is not a target.
Products require two distinct fitting times, without selecting for future presence.
F1 duplicate views and publisher-selected S4 rows are not extra observations.

The sole target is `log10(product intensity) - log10(intact-substrate intensity)`.
No imputation, generated negatives, inferred censoring, molar conversion or synthetic
biological labels. No affinity, stability, TCR, MHC-class, species, transport, APC
knockout or separate MS head trains. IC50 and 500 nM binding metrics are inapplicable.

| Split | Minutes | Measurements | Peptides | Acquisitions |
|---|---|---:|---:|---:|
| Training | 0, 20, 40, 60, 80 | 626 | 96 | 15 |
| Validation | 100 | 181 | 75 | 3 |
| Test | 120 | 186 | 74 | 3 |

All 8,138 assignment decisions remain. Admitted products each have one sequence
origin in the parent. Both held-out support gates pass. Weights give equal mass to
peptides, then times, then replicate cells. This is neither sequence-disjoint nor
independent-specimen evaluation. Source audits check quantity fidelity separately
from predictive outcome access; fitting reads training outcomes only.

## Models, inputs and training

Parent sequence, product span and time enter an independent-bond fragmentation
process. Each of 139 bonds has a positive rate, either shared or determined by the
residue immediately before it. At exposure `u = time + tau`, expected fragment
amount is internal-bond survival times the cleavage probabilities of existing
boundaries. Parent termini require no cleavage. All 9,870 possible spans, including
short products and intact parent, belong to the process. Query spans are evaluated
directly during fitting; complete expected-span dumps are saved for exact fits.
Residue mass is conserved. Unobserved span amounts are not validated concentrations.

Forecast: `log10(expected product / expected parent)` plus one training-only
profiled offset per peptide. These 96 nuisance offsets absorb unknown fixed MS
response and anchor forecasts to observed fitting data. They do not calibrate new
peptides. There is no learned sequence encoder or full Presto model in these fits.

Uniform: two optimized coordinates (rate and `tau`). Residue: 18 (17 observed
residue-category rates and `tau`). Sigmoid bounds are 1e-7–1 per minute for rates
and 0.001–120 minutes for `tau`; initialization is 0.001/min and 10 minutes.
Effective pre-exposure is inferred, not a measured source initial condition.

All 14 fits use 800 full-batch float64 Adam updates, learning rate 0.03, betas
(0.9, 0.999), epsilon 1e-8, fixed final checkpoint. Loss is weighted MSE plus
0.01 mean squared centered log rates for the residue model; uniform has no penalty.
Every allocated parameter receives gradients and changes. Final residue rates are
8.37e-5–9.98e-5/min; `tau = 102.21` minutes. Its objective changes by -0.000113 over
the last 50 updates. Replay does not prove convergence or biological identifiability.

Two exact fits use seed 42. Twelve conditional-score fits cross both models with
16/64 draws and seeds 42/43/44. These retain **exact forward probabilities and loss
residuals**. Query boundaries are forced cleaved and interiors intact; outside
bonds are sampled. Binomial counts compress sufficient statistics of equivalent
independent Bernoulli cut sets; a full Bernoulli path remains for qualification.
This is an exact-forward/REINFORCE-gradient hybrid, not simulator-only RL, continuous
event-time trajectories or GRPO. No clipping, standardization, sample reuse or
reference-policy KL is introduced.

Seven direct controls use training-only peptide anchors: mean, last-time mean,
shared linear time slope, or peptide slopes shrunk toward a jointly fitted shared
slope at penalties 0.001/0.01/0.1/1. Time is scaled by 80 minutes. Weighted least
squares fits coefficients. Validation alone selects comparator and exact physical
variant. No refitting follows held-out evaluation.

## Held-out results

Errors, R² and standard deviations use recipe weights; Pearson/Spearman are
unweighted. All training/validation/test metrics and peptide/replicate/time strata
are retained in [evaluation artifacts](results/evaluation/).

| Model | Validation RMSE | Test RMSE | Test R² | Test Pearson | Test Spearman |
|---|---:|---:|---:|---:|---:|
| Peptide fitting mean | 0.445013 | 0.602658 | -0.359165 | 0.815842 | 0.812309 |
| Last fitting-time mean | 0.229625 | 0.364187 | 0.503660 | 0.922212 | 0.908959 |
| Shared linear slope | 0.309585 | 0.368062 | 0.493042 | 0.801030 | 0.805770 |
| Shrunk slopes, 0.001 | 0.255323 | 0.370871 | 0.485273 | 0.834832 | 0.799389 |
| Shrunk slopes, 0.01 | 0.223788 | 0.307479 | 0.646196 | 0.879406 | 0.846997 |
| **Shrunk slopes, 0.1; selected control** | **0.206085** | **0.228184** | **0.805149** | **0.923571** | **0.909977** |
| Shrunk slopes, 1 | 0.287225 | 0.335877 | 0.577826 | 0.830548 | 0.831842 |
| Uniform fragmentation, exact | 0.304794 | 0.359391 | 0.516646 | 0.812660 | 0.810736 |
| **Residue fragmentation, exact; selected physical** | **0.304593** | **0.359038** | **0.517596** | **0.812986** | **0.810438** |

Physical validation RMSE worsens 47.80% and test 57.35% against direct regression.
Both required 5% improvement gates fail; positive raw correlations and variation
gates pass. Paired peptide bootstrap percentage-gain intervals are [-62.29%, -34.70%]
on validation and [-86.69%, -34.34%] on test (1,000 resamples, seed 31415). These
summarize peptide uncertainty, not independent biological specimens.

| Test change from last fitting-time mean | Pearson | Spearman | R² | Prediction SD | Target SD |
|---|---:|---:|---:|---:|---:|
| Selected direct regression | 0.15340 | 0.23071 | -0.07353 | 0.05931 | 0.22023 |
| Selected fragmentation | -0.21845 | -0.13886 | -1.65781 | 0.23127 | 0.22023 |

Persistence predicts zero change and has undefined change correlations. Its raw
Spearman of 0.91 shows why raw ratios alone overstate temporal skill. Regression's
changes remain compressed; fragmentation varies but predicts changes poorly.

| Sampled-gradient condition | Test RMSE mean ± seed SD | Core fit seconds | Relative to matched exact |
|---|---:|---:|---:|
| Uniform, 16 draws | 0.359376 ± 0.000074 | 4.52 | 1.82× |
| Uniform, 64 draws | 0.359391 ± 0.000045 | 6.34 | 2.55× |
| Residue, 16 draws | 0.358994 ± 0.000025 | 4.54 | 1.82× |
| Residue, 64 draws | 0.359020 ± 0.000027 | 6.36 | 2.54× |

Seed spread is optimizer stochasticity, not biological uncertainty. Exact core fits
take 2.49/2.50 seconds. Neither sampling budget improves the biological conclusion.
The [figure](results/evaluation/comparison.png) shows errors, physical temporal
changes and training curves; PDF is retained. The rendered figure was inspected.

## Verification and reproduction

Independent admission reproduces all 993 targets (maximum error 8.88e-16), lineage,
eligibility and weights. Qualification checks all 32 cut states of a six-residue
chain, all-span mass/count identities, production finite differences and conditional
mean gradients at frozen tolerances. These fixtures are not biological training labels.

NumPy/SciPy replay independently reconstructs all 13,902 physical predictions and
seven controls, maximum error 1.38e-13. All 126 overall metric records agree to
1.34e-15. Every 800-step Adam recurrence is checked; both exact fits' gradients are
independently reconstructed at every step. Sampled gradient draws are **not
independently replayed**: checks are pre-fit gradient qualification, fresh RNG states
and optimizer recurrence verification. Closure separately replays selection and
paired peptide bootstrap. No outcomes or thresholds are changed.

The 20 pre-closure stages total 137.53 seconds; maximum polled RSS is 823,705,600
bytes. Admission/qualification/evaluation/replay take 9.37/6.39/17.42/3.93 seconds.
Hardware is local macOS arm64 CPU, not GPU. Caps remain 3 GiB RSS, 20 minutes/stage,
128 MiB new physical artifacts, 1 GiB free disk plus 32 MiB startup margin. The
artifact cap conservatively measures shared-volume free-space change. No cap was
relaxed; pre-closure logical artifacts total about 46 MB.

Run [launch.json](reproduce/launch.json) commands in order from this directory:
`sh reproduce/launch.sh <stage>`. Pinned source/recipe paths are required; existing
stage receipts prevent overwrites. Reproduction needs a fresh output directory.
Overrides: `PYTHONPATH=/private/tmp/hitlist-intact-kinetics-20261025` and
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. Shared interpreter,
Python/package versions and dirty state are recorded. Executed scripts and imported
model code are frozen per stage in [reproduce/source](reproduce/source/).
The registration's unopened-outcomes flag records registration time; later access
is explicitly recorded in [outcome-access.json](results/evaluation/outcome-access.json).

All 801 parameter/optimizer states per fit, nuisance offsets, RNG states,
loss/gradient/runtime trajectories, final predictions and summaries remain in
[results/fits](results/fits/). Complete source/recipe packages remain at the data
paths above. This directory retains split data, fingerprints, all predictions,
source snapshots, stage logs, source-test output, resource receipts and hashes.
No checkpoint is pruned.

Shared recipe tests: 32 pass. An initial fixture omitted the F2 40-minute R1 extra
column; corrected fixture and failed log remain at
`/private/tmp/intact-kinetics-focused-v2.log` and `/private/tmp/intact-kinetics-focused.log`.
Production recipe was unchanged. Full local suite is deferred below its 2 GiB
free-space prerequisite. Source/recipe publication and hosted checks remain separate
unfinished work; the scientific report does not imply those releases are complete.

## Decision

Keep direct shrunk regression as the provisional reference for this exact temporal
contract. Do not promote fragmentation or unobserved span counts to an availability
predictor. Numerical checks reduce the chance of gradient/bookkeeping failure but
do not establish why biological prediction fails. Independent-bond assumptions,
initial products, fixed MS response and single-substrate support remain limitations.

Unlike HAP1 response and measured-MS successes, this directly tests ordered
product/substrate trajectories and finds no mechanistic advantage. MAPP observed-set
gains also do not identify unobserved mass. New model selection requires a fresh
prospective evaluation; inspected 100/120-minute outcomes cannot become a new blind
test. The full APC availability/concentration goal remains active and unachieved.
