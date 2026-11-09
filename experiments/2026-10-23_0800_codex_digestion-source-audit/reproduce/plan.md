# Digestion time-course source audit

Registered before downloading processed archive members, 2026-10-23. Codex / GPT-6.
Owner: Hitlist #770. Builds on controlled-fragment-kinetics.md. No training in this source audit.

## Hypothesis and decisions
Determine whether the original deposited outputs expose independently repeated, unnormalized peptide time courses and contributor layers sufficient to distinguish fragment production/removal from static MS response. Inventory the processed archive before choosing any training endpoint. Do not treat selected 4h/3h panels as a complete peptide universe or non-detections as negatives.

## Work and evidence
1. Resolve original public download URLs from the MassIVE page and exact 505-row inventory; preserve provenance and hashes. Retrieve Identifier.xlsx, methods document and literal substrate FASTAs.
2. Inspect ZIP central directory with verified HTTP byte ranges when supported; preserve each response status/range/hash and ZIP CRC for extracted members. If ranges are unsupported, use an ordinary complete download with exact size verification, never label partial bytes a complete archive. Prefer processed quantitative tables and upstream/raw quantity contributors over spectra and search databases. Record explicit member selection by schema layer, not desired biological results.
3. Inspect workbook/table schemas, original row counts, replicate/run labels, time values, mapping fields, quantity layers and missingness. Download multiple original layers where needed to resolve source transformations. This is a source audit only; shared curation and any training recipe must be implemented in Hitlist before use.
4. Independently verify file decompression/CRC, metadata joins and representative arithmetic if source contributors are available. Report where source derivations cannot be replayed, plus exact missing metadata and blocked endpoint to #770.
5. Freeze the next source-backed admission and model-objective contract only when evidence supports it. No model fits, held-out partition selection, optimization or predictive claims in this family. No validation/test prediction dumps because no predictive evaluation occurs.

## Controls and limits
No synthetic biological labels, no MHC-affinity filter, no source value imputation, no author outreach, no environment installs or cloud training. Do not interpret purified proteasome results as cellular concentrations/APC machinery effects. Distinguish exact substrate construct from canonical sequences, biological digest vs technical MS replicate, continuous vs spliced peptides and I/L ambiguity. Substrate/enzyme inputs and time are measured context only if explicitly supplied. Preserve all original selection and source provenance.

## Closure
README, reproducibility launch/config/source hashes, archive inventory, download receipts, schema/quantity audit, limitations and next action, canonical experiment registry and issue770 update. Report runtime and local artifact locations. Keep full availability/concentration objective active.
