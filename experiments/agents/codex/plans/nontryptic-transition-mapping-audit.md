# Source identity correction and aggregation closure

This is closure of the registered 1857 source-lineage questions, not a new
predictive experiment. The previous goal turn made progress: official exports
were retrieved and 42,941 source cells replayed. Verify attempt 2 completed, but
its S3B lookup incorrectly required bare S3B peptide labels to equal modified
export forms. Its immutable code and outputs remain evidence of that mistake.

## Fixed analysis before execution

1. Read the original S6B workbook using the pinned shared reader and independently
   compare original values with openpyxl. A reported form corresponds to an S6B
   assertion only when bare sequence, explicit labeled residue positions and
   total nominal mass shift all agree uniquely. Preserve the S6B A/B/D cells.
   This is a correspondence between reported labels, not canonical isotope atom
   resolution; S6A conflicts remain open. Never infer an isotopologue from label
   count alone.
2. Recompute every existing product-only/all-transition group from original
   export cells and require exact equality with the stored aggregation result.
   Retain missing/invalid/zero states. Count absent Concat separately from actual
   nonempty disagreements. No filtering or deduplication based on signal values.
3. Join S3 cells by source-declared isotopologue (1H direct, 2H embedded), bare
   sequence, nominal dose and the independently matched S6B assertion. Verify S3
   values against original cells. Compare all reported replicate candidates,
   retaining ordinal-equals-suffix as a diagnostic only. Compare both previously
   registered aggregation modes with unchanged exact, relative 1e-6 and optional
   integer-rounding diagnostics. No best-matching acquisition assignment.
4. Save per-cell candidates and compact per-peptide/mode summaries. A missing
   export replicate remains missing; blank S3 cells stay blank. Report discrepancies
   without selecting transition subsets, fitting scales or changing tolerances.
5. Update the family README, canonical log and handoff from terminal receipts.
   Decide shared source schema eligibility separately from quantitative training
   eligibility. No old held-outs, model fitting, synthetic labels or new retrieval.

## Execution and verification

The executed launch.py cannot be rewritten. Add a byte-equivalent companion
launcher except for allowing the new `mapping_audit` stage. Verify that exact
single replacement before launch. Preserve all guards: 3 GiB startup and RSS,
20 min, 1 GiB disk reserve plus 32 MiB admission margin, one numerical thread,
50 ms sampling and OS peak check. Freeze inputs, code, scope and launch command
in a new request; output files use exclusive creation. The inactive 1 GiB
proposal is not used. Any genuine resource refusal is preserved.

Success means that the source correspondence and numerical disagreements are
auditable, not that all published values match or any biological model succeeds.
No predictive split or validation/test metrics apply to this source-only closure.
