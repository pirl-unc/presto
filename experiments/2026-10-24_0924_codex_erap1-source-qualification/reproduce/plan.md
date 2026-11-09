# Prospective workbook qualification contract

Codex / GPT-6, October 24, 2026. Previous goal turn was a status-only answer:
no new experimental evidence. This executes the already committed plan
`experiments/agents/codex/plans/erap1-independent-source-qualification.md`.

1. Verify the downloaded archive hash, size and every member CRC; preserve both
   workbook byte streams unchanged. Inspect workbook sheets, actual occupied cells,
   formulas and their cached results, row/column labels, merged ranges and units.
   Retain source workbook/sheet/address for each observed field. Never recalculate,
   overwrite cells, fill missing data or turn fitted/simulated curves into raw data.
2. Classify every sheet/figure panel using the primary article and supplement.
   Separate structural/computed data, fluorogenic reporter assays, inhibitor or
   activation curves, peptide disappearance, product fractions and time courses.
   Substrate provenance, enzyme context, units, independent experiments versus
   technical repeats, fitted summaries and unknowns must remain explicit.
3. Identify a supported non-pathogen model-substrate cohort if present. Do not
   expand mixtures into individual peptide labels or infer unreported products,
   rate constants, source-species identity, replicate independence or concentrations.
   Qualification requires a traceable quantity, substrate identity, conditions,
   time/concentration axes and adequate sequence/context variation for the proposed
   output. A reporter-only cohort cannot qualify peptide-specific trimming rates.
4. Independently check workbook observations against a second reader. Inspect
   formula/cached-value differences without recalc. Preserve ambiguity and failures.
   This audit is not a production source curator. Any admitted relations/recipes
   must be implemented in Hitlist under issue809, on a separate branch from806/807.
5. Freeze commands, executed source, runtime, environment, Git state and asset
   hashes. Close with a sheet/panel disposition table and a concrete admission
   decision in README and the canonical log. Update issue809 with source evidence.

No train/validation/test partition or prediction dumps are appropriate for this
source audit. No measured cohort is assumed available in advance, no statistical
power or generalization claim follows from cell counts, and no practical baseline
is promoted. The full APC precursor-relative availability goal remains active.
