# Full-support stiff fragment-state solver qualification

Prospective plan, October 23, 2026, Codex / GPT-6. The preceding fragment-state
family qualifies the physical count/trajectory equations but fails its actual
parent runtime gate on four eta=+1 cases. Keep that negative baseline unchanged.
No new biological fitting is authorized until this qualification completes.

## Hypothesis and fixed scope

The first-moment operator is triangular under decreasing fragment length. A
shifted solve `(z I - dt A) x = y` can use row/column accumulators in O(L²):
solve each length from long to short, then add its length-weighted solution to
the corresponding row and column accumulators. All spans remain, with no sparse
support selection. This can replace stiffness-dependent uniformization with a
fixed number of shifted solves in a rational approximation of the exponential.

Implement and compare IPF CRAM16 and CRAM48, coefficients and formulation from
the OpenMC official numerical implementation and its Pusa 2016 reference:
https://docs.openmc.org/en/stable/_modules/openmc/deplete/cram.html
https://doi.org/10.13182/NSE15-26
Use float64/complex128 CPU with the same one-thread environment. Do not add or
upgrade packages. Initially keep the solver in the experiment directory; retain
the production uniformization implementation unchanged.

## Fixed gates before biological training

1. Independently assemble dense fragment operators for lengths 2, 6, 12 with
   eta=-1,0,+1, fixed nonuniform rates and multiple positive/complex shifts.
   Verify shifted solutions/residuals and autograd against dense torch solves,
   including arbitrary RHS and gradients with respect to log-rates, eta and RHS.
   Absolute discrepancy <=1e-10; dense gradient discrepancy <=1e-9.
2. Fixed exponential grid: those sizes, rate scales .01,1,100 and times 0,.5,2,8,24
   hours. Compare both CRAM orders against independent SciPy exponentials and
   Frechet derivatives, plus eta0 closed forms. Expected count absolute error
   <=1e-10, mass absolute error <=1e-8, readout gradient <=1e-8. Save every value,
   minimum, count of negatives and relative errors above expected count 1e-12.
   No clipping, floors or normalization. Any negative output prevents direct
   log-observation admission until a separately specified solution is verified.
3. Replay all 21 preceding actual-parent input cases with exact same sequences,
   rate initializer, artificial readout, time, eta range, full support, 60-second
   wall and 4 GiB memory limits. Save complete counts and gradients. Compare all
   17 available uniformization references and eta0 analytical amounts/gradients.
   For the four prior timeouts, qualify fixed one-vs-two-substep convergence and
   both orders independently; this is empirical agreement, not a rigorous error
   certificate for arbitrary nonnormal matrices.
4. The practical candidate must satisfy numerical gates and finish all actual
   cases within the previous limits. Report both orders' speed/memory and failures;
   prefer the least costly passing formulation, not a posthoc change to support.
   A prospective biological fit still needs a time-grid/parameter-domain contract
   and sampled-loss gradient qualification. Full-path score law and K512 sampler
   remain unchanged references; faster exact math does not imply RL inferiority.

## Data and artifacts

Reuse only the prior frozen runtime-inputs and existing public initializer. No
measurement targets, validation or test labels; no optimizer/pretraining/assay
loss. No biological predictive metrics apply. Register a new timestamped family
before implementation, freeze code/config hashes before each stage, retain all
positive/negative arrays, runtimes and independent checks, write a report and
canonical log entry. A small-fixture pass cannot be called biological validation.
If a gate fails, preserve it and write an explicit amendment before changing the
method. Full APC abundance modulo precursor, machinery, observation and clearance
objectives remain unchanged.
