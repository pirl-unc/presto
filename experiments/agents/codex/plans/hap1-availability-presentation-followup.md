# Follow-up after HAP1 biological signal qualification

Status: design work to resolve before the next physical-model fit. This is not a
launched experiment or a claim that the full availability model now works.

## Evidence that changes the next action

The 0730 registered recipe and fixed baselines yield a 10.0% grouped-CV RMSE gain
from simple peptide sequence features, improving all ten KO conditions. Keep all
seven frozen baselines, especially sequence ridge0.1, as references. Its signal
is a target the physical model must explain or improve, not evidence that its
coefficients describe cleavage. Reserve the outer validation/test outcomes until
one complete physical comparison is specified and all candidate fits are sealed.

The earlier native-cell processing model gave a small positive result, while
macrophage presentation models failed and their test sets are closed. HAP1 offers
stronger known-machinery response, but measures MHC-eluted peptides. It cannot
alone distinguish supply, transport, loading/stability, recovery and detection.

## Required next model contract

1. Keep an availability submodel without MHC or TCR inputs. Full source protein,
   supplied cellular/machinery context and an explicitly scoped precursor input
   define a distribution or expected amounts over all contiguous fragments,
   including short fragments and nonbinders. Reuse the checked continuous-input,
   independent-cut/removal primitives as the exact reference; do not normalize
   only over observed ligands. State which rate parameters have independent
   digestion/native-peptidome supervision and which remain unsupported.
2. Treat transport and presentation selection as separate from cleavage. HAP1
   loss labels supply component-state assertions; they do not establish zero
   residual activity or authorize every KO to rescale cleavage. Audit each
   intervention's supported target stage in the admitted primary-source contract
   before introducing a parameter. Use the smallest identifiable conditional
   response parameterization. Preserve unresolved or multi-stage interventions.
3. Express observed MHC-elution signal as precursor-relative availability times
   downstream transport/presentation selection and observation response. Work in
   log space with explicit mixture sums over candidate origins. Retain the
   validated source WT denominator and missingness policy. A same-peptide response
   may cancel a shared detector factor only as a stated assumption; do not claim
   an independently trained detector from these ratios. Independent non-MHC MS
   calibration remains necessary for absolute signal interpretation.
4. Use the direct sequence baseline as an explicit observation/selection control,
   not as a hidden residual large enough to make the physical model irrelevant.
   Include a frozen-baseline-plus-physical-change arm and a physical-only arm.
   Audit whether physical terms change predictions and improve grouped validation
   beyond the fixed direct control. Preserve parameter support, Jacobian/gradient
   evidence and amplitude variation separately from predictive performance.
5. Train exact and sampled versions against precisely the same source-weighted
   response objective. Start with exact forward expectations and conditional-score
   sampling as a matched estimator control; label that hybrid accurately. A fully
   discrete simulator must expose the same full fragment support and observation
   model before its result is comparable. Sampling budgets, seeds, reuse/clipping/
   standardization/KL settings must be fixed before fitting; synthetic GRPO wins
   do not justify importing all options into a weak biological objective.

## Checks to finish before launch

- Freeze the source-stage mapping and the minimal trainable parameter list.
  Do not silently declare unavailable labels, molar scales or causal activities.
- Fix the native/purified training-only anchoring contract and audit exact peptide,
  parent, study and reference overlap against reserved HAP1 evaluation components.
  Prior inspected tests cannot become training labels by changing dataset names.
- Verify all spans remain supported without materializing an impractical quadratic
  table. Exact queried amounts and importance sampling must share normalization,
  input/removal conventions and origin mixtures; retain analytic small-protein tests.
- Choose one bounded cohort/feature/optimizer schedule and register its exact
  objective, comparator, runtime/score budget and success rule. No new outer-test
  inspection until the intended comparison and all training attempts are sealed.
- Require measurable physical contribution and predictive improvement, not simply
  nonzero parameter updates. If direct selection explains the result and processing
  does not help, report that outcome and seek independent APC free-peptide evidence.
- Repeat successful biological prediction on independent APC material before
  claiming the requested precursor-relative intracellular availability objective.

## Immediate shared-source integration

Hitlist PR807 is stacked on PR806. The latter conflicts with main after PR804 added
a separate precursor_context family with the same next schema/version numbers.
Reconcile both families and versioning without altering the frozen 0730 experiment
or relabeling its source manifest. Preserve ability to reproduce the original pinned
package. Run required checks and release through the existing upstream workflow;
current drafts and passing source audits are not a published PyPI release.
