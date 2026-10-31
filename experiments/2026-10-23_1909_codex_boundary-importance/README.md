# Positive boundary-conditioned importance sampling

**All 45 ordinary primary cases pass, but one of three rare stress cases fails.**
Uniformly proposed boundary times give positive log-space estimates even for
vanishingly rare products, but badly mismatch fast-cleaving boundaries. This
family does not qualify biological training or proceed to full-parent fitting.

## Question and estimator

Earlier unconditional trajectories often never produced the requested fragment.
For a fixed target span, this proposal draws each required left/right boundary
cut time independently uniformly on `(0,T)`, sorts the labeled times, forbids
internal target cuts, and simulates all other cuts at their original hazards.
All outside fragments remain in the trajectory. There is no factorial adjustment
for sorting labeled boundaries.

The importance weight is the original restricted path density divided by the
normalized proposal density. Its log is `m*log(T) + sum(log(boundary hazard)) -
integral(forbidden hazard)`, where m is 0, 1 or 2. We retain full original and
proposal event/waiting/terminal-survival log densities and scores. No draw is
differentiated through and no numerical probability is clipped or floored.
The count estimator is unbiased; taking its log and a self-normalized score ratio
introduces finite-sample bias, which is measured rather than called unbiased.
An intact parent at T=0 is handled analytically; other targets at T=0 have exactly
zero probability and explicitly reject the log estimator.

## Fixed numerical experiment and outcomes

Six-residue parent, bond rates `[.02,.07,.11,.03,.15]` per hour, eta `[-1,0,1]`,
times `[.5,2,8]` hours, and targets `[0,6),[0,3),[3,6),[1,5),[2,3)`.
Each state uses K=256/2048/16384, 16 independent groups and the exact seed formula
in launch.json. Primary K=16384 gates are absolute mean relative probability
error <=5%, absolute mean log bias <=.02, mean log-gradient relative error <=5%
and RMS relative error <=25%. This is an absolute **signed mean bias** criterion;
it differs from the later full-parent study's mean absolute error criterion.
The exact producer uses a 32-partition-state generator and Frechet derivatives.

All 45 ordinary primary states pass. Worst absolute mean probability error is
0.002009, absolute mean log bias 0.002004, mean gradient error 0.008873 and RMS
0.244641. The smallest mean ESS is 4,769.1 out of 16,384. The three fixed rare
eta=0 tests alter internal or boundary rates at T=8 with target `[1,5)`:

| Stress case | True log probability | Mean probability relative error | Mean log bias | RMS gradient error | Mean ESS | Pass |
| --- | --- | --- | --- | --- | --- | --- |
| internal_100 | -170.270 | 0.00012827 | 0.00012266 | 0.00058614 | 14624.75 | True |
| internal_1000 | -1682.270 | -0.00085979 | -0.00086506 | 0.00035278 | 14633.71 | True |
| fast_boundaries | -1.680 | 0.066063 | 0.053024 | 0.10032 | 34.55 | False |

The fast-boundary case has mean ESS only 34.55/16384 and fails probability and
log-bias gates. In `internal_1000`, every ordinary linear weight underflows to
zero, but log estimates remain finite near log probability -1682.27. The failure
is retained; counts, seeds and targets were not changed to rescue it.

The unchanged earlier unconditional reference has zero-hit groups in 93/720,
33/720 and 0/720 target/group combinations at its K=1024/8192/65536. See
[reference reanalysis](results/unconditional-reference.json). Its paths are shared
across target queries, whereas this estimator is target-specific and uses other
K values. This is not a matched runtime comparison.

## Verification and runtime

144 conditions, 2,304 groups and 14,352,384 sampled paths complete in 24.81 s,
peak process RSS 318,046,208 bytes. Independent verification uses a separate
21-fragment generator/Frechet reference and a closed-form rare eta derivative.
It replays all 2,304 sufficient-statistic groups, 48 complete primary raw groups
and all 384 saved full event histories. Maximum raw-statistic discrepancy is
1.71e-15; maximum path density/score discrepancy 9.10e-13. This is not a replay
of every unsaved individual draw. Independent verification takes 0.504 s.

Three focused tests pass. An initial pytest collection failed because the
experiment directory was absent from the import path. The corrected test launcher
sets that path explicitly; original and corrected logs and a test-import note are
preserved. The model source and qualification samples were unchanged.

## Physical model and scope

A parent begins as one intact molecule. A live fragment `[a,b)` can split at any
internal bond k with hazard `exp(log_rate[k]) * ((b-a)/20)**eta`. Both children
remain in the state and may split again. All lengths 1 through the complete parent
remain represented, including monomers and products that cannot bind MHC. No
peptide is discarded by a binding filter. Every trajectory conserves residues.
The expected count of one particular source span is its probability of surviving
as a fragment at the observation time. It is not a probability normalized among
observed MS peptides.

These studies qualify inference and gradients for this fixed stochastic law.
They have **zero optimizer updates**, no pretraining, no new biological labels,
no assay losses, no held-out predictive evaluation and no fitted checkpoints.
There is intentionally no validation/test split: the small fixtures are numerical
reference problems, and full-parent work, where present, uses frozen training
inputs only. nM regression and binder-threshold metrics are inapplicable.
Numerical probabilities, gradients, sufficient statistics and saved path arrays
are the appropriate prediction artifacts. This does not establish APC quantities,
MS-response calibration, MHC/TCR performance, transport or clearance.

## Reproduction and preserved evidence

Agent/model: Codex / GPT-6. Registered October 23, 2026; source base commit
`702d66be09a32e753272917650db4f10f8b569c0`, with recorded dirty worktree and
experiment-local source snapshots. Existing unrelated edits were preserved.
Python 3.12.6, Torch 2.14.1, NumPy 2.5.4, SciPy 1.18.1, local macOS arm64 CPU,
float64, one Torch thread; `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1`. No Modal/GPU, installs or shared-environment mutation.

- [Registration](reproduce/launch.json), [prospective plan](reproduce/plan.md),
  [launch command](reproduce/launch.sh), [executed source hashes](reproduce/execution.json).
- [All qualification groups (raw archive)](results/raw-artifacts.tar.gz), [summary](results/summary.json),
  [comparison CSV](results/comparison.csv), [derived summary](results/report-summary.json).
- `results/primary-ordinary-*.npz` / `results/primary-rare-*.npz` contain the complete
  first primary group for each state; filenames are linked from qualification.json.
  [Recorded event histories (raw archive)](results/raw-artifacts.tar.gz) contain eight paths/state.
  Other groups retain all scaled sufficient statistics, not every individual path.
- [Independent verification](results/independent-verification.json),
  [verification command](reproduce/verify.sh), [verification contract](reproduce/verify.json),
  frozen sources in `reproduce/source/`, execution/test logs in `launch_logs/`.
- The report tables for both families are generated by the exponential family's
  `code/report.py`; its `reproduce/report.sh` and `report.json` freeze the command/source.
  `results/closure-hashes.json` inventories the published bundle.

First run `/Users/iskander/code/shared-virtual-env/bin/python reproduce/unpack.py`
to restore exact raw outputs from [the archive](results/raw-artifacts.tar.gz),
checking the [per-file receipt](results/raw-artifacts.json). This refuses to
overwrite differing files. Raw paths below are relative to that restored tree.
Then run `sh reproduce/test.sh`, `sh reproduce/launch.sh`, and `sh reproduce/verify.sh`
from the experiment directory in a fresh copy/output workspace. The qualification
launcher refuses to overwrite immutable outcomes. Preserve the linked prior
reference inputs when replaying full-parent work. Source snapshots and hashes,
not mutable current defaults, define the experiment.

## Handoff

- **Status:** Numerical experiment complete; rare fast-boundary gate fails.
- **Next step:** The separately registered [truncated-exponential proposal](../2026-10-23_1920_codex_boundary-exponential/README.md) tests a density matched to the initial boundary hazard and a proposal-score control.
- **Open questions:** Proposal efficiency in changing fragment states; no new
  Hitlist source/metadata gap arose here. The broader biological goal stays active.
