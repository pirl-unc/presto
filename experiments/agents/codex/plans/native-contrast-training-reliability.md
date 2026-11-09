# Native response failure: training-only acquisition sensitivity

Prepared after the fixed family0630 screen failed both outer gates. Its validation
and test outcomes are now inspected and must not become a fresh confirmation set.
This next family addresses one hypothesis before adding features or RL:

**Hypothesis:** the complete-case same-parent response is unstable across the six
acquisitions, so additional model capacity would mainly fit acquisition-specific
variation. This is an observation-level diagnosis, not a biological noise ceiling.

## Fixed scope and inputs

Use only the 158 training contrasts in the 55 original training dependency
components from recipe0612. No validation/test reads, extra eligibility filters,
new reference choices, sequence ranking, alpha fitting or model comparisons.
Retain the exact component/parent/contrast weights, normalized to one. Pin the
package manifest and all executed diagnostic code in a new timestamped family
before computing the diagnostics. Keep original acquisition labels and source IDs.

For each contrast, take its six original log10 peptide/reference ratios. Confirm
these against the twelve original measurement contributors, independently using
log10 of the intensity ratio. The response is the unpaired difference of three-run
arm means. Numbered runs remain acquisitions with unresolved physical replication
and pairing; do not call them six independent biological samples.

## Prespecified measurements

1. For each arm, subtract its per-contrast mean from each acquisition ratio. Report
   weighted within-arm RMS, per-acquisition weighted residual means and RMS, and
   the weighted RMS of the original mean response. Their ratio is descriptive;
   do not divide by sqrt(3) to claim a biological measurement-error variance.
2. Form six leave-one-acquisition responses: omit one original run, using two
   runs in that arm and all three in the other. Report every contrast's change
   from its original response, weighted absolute/RMS changes, and sign changes
   with explicit zero-target exclusions. Report the weighted fraction retaining
   the same nonzero sign under all six omissions. These are sensitivity analyses,
   not six new training labels or independent evaluations.
3. For each of the nine choices of one control and one KD acquisition, compare
   that two-run response with the response from the remaining two runs per arm.
   Across the fixed training contrasts, compute weighted Pearson and ordinary
   average-tie-rank weighted Spearman, spread and signed agreement. The nine
   partitions are dependent diagnostic views; do not pool them as independent
   observations, bootstrap across them, or claim nine additional replicates.
4. Retain per-component contributions and show whether the pattern is concentrated
   in a few parent groups. Do not discard those groups or choose a better-looking
   acquisition subset. Preserve full tables, including undefined correlations.

## Decision and limits

If the median of the nine defined weighted Pearson correlations is <=0 (or all
are undefined), prioritize source-observation stability and physical replication
metadata before expanding model capacity on this cohort. If it is positive,
register a separate training-component-only comparison of local parent-boundary
context against the frozen 61-feature reference. Either outcome is development
evidence, not biological transfer or a validated ceiling. Positive concordance
can also reflect shared-reference, acquisition or observation effects.

Do not change the original outer result, tune on its 8 validation/18 test groups,
or claim a successful exact-versus-RL comparison. A changed representation needs
new independent biological confirmation. Full APC precursor-relative abundance,
missing-fragment support, clearance and MS observation calibration remain open.

## Execution and verification

Local CPU only, one BLAS thread, no installs or cloud jobs. Keep the 20-minute,
3 GiB RSS and 1 GiB disk reserve plus 32 MiB startup limits. Record environment
versions at stage start and OS peak RSS alongside short-interval process polling.
Independently reconstruct every response/residual/omission/partition metric from
the original source contributors, with absolute 2e-12 / relative 2e-13 arithmetic
tolerance. No predictive validation/test dump or 500 nM classification applies;
explain the training-only diagnostic contract in the README and canonical log.
Close all source tables, summaries, plots and code/hash receipts before choosing
the next scientific step. The study's missing physical-replicate mapping belongs
on Hitlist issue837; no author outreach is authorized.
