# Independent MS prediction closure

Codex / GPT-6, 2026-10-27. Written before accessing recovered test predictions
or implementing this reducer. This implements the already frozen 2030/2036
evaluation contract; it does not revise a scientific gate or fit another model.
The original epoch-three validation metric is already recorded as development.

Require the streaming controller to terminate successfully with all three native
arms and three logistic comparators. Pin every prediction/checkpoint/feature file.
Replay all native normalized observations against their immutable arm manifests,
including permuted training labels. Join all comparator rows by observation ID,
source path/offset, component, peptide, label and partition. Verify the complete
61 sequence features and training-only means/scales, then reconstruct each
logistic prediction from frozen coefficients. Replay 20 deterministic evenly
spaced model inputs per partition against frozen native/parent checkpoints.
Full inference is not independently reimplemented; source/metric reduction is.

Use float64 NumPy BCE/Brier and scikit-learn classification, AUROC and average
precision. Compare all reported canonical metrics: binary metrics to 1e-10;
float32 per-example BCE to 5e-6 absolute plus 1e-6 relative and aggregate BCE to
5e-7 absolute. Probability replay is 1e-12; fresh float32 inference is 2e-5 logits
or log10(nM), reflecting possible batch-size attention-kernel roundoff. Preserve
the actual maximum discrepancies. Do not relax tolerances after a failed check.

The primary ensemble averages the two fixed-final true-label logits. Publish
per-example ensemble validation/test/external predictions and all arms' point
metrics. The permuted arm's training metrics use its training labels and are
explicitly separate from unchanged held-out labels. Every test gate remains:
2% relative BCE gain versus each comparator; AUROC drop no greater than .01;
AUROC at least .02 above the permutation diagnostic. For both test sets, perform
2,000 paired component-bootstrap draws using seed 20261026, drawing the same
number of components with replacement and retaining all rows/molecules/pairs.
Use one PRNG stream in declared order DP test, DQ8 external. Save every draw and
percentile 95% intervals for relative BCE gain and AUROC difference versus the
combined comparator. If a BCE-gain interval includes zero, report inconclusive;
if wholly negative report failure. Report failed gates without model selection.

Verify weighted bootstrap AUROC with independent scikit-learn toy/random fixtures,
including ties and unequal cluster multiplicity, before execution. Test zero-gain
and positive-gain gate boundaries. No production source changes are needed.
Freeze reducer and invocation before actual closure. Local CPU, one numerical
thread/one interop, 300 seconds/2 GiB RSS per verification/comparison action, with
the existing 13/10 GiB available-RAM and 15/10 GiB disk start/floor. No automatic
retry. Run after the native/control controller finishes, not concurrently with
another new fit. Preserve failures; split distinct closure actions if required
before execution rather than silently increasing limits.

Store source verification, numerical discrepancies, all metrics, bootstrap draws,
CSV tables and a compact scientific plot in this family. Close README and the
canonical log. These artificial-background results cannot identify processing,
MS response, natural prevalence, donor independence or causal pathway effects.
