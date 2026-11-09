# Newer MAPP baseline quantity source qualification

Codex / GPT-6, 2026-10-24, before source-table inspection or fitting.

## Question and rationale

Can Goldberg2025 (DOI10.1038/s41586-025-08615-w; PXD052314) supply
source-backed baseline captured-fragment quantities for a prospective peptide
availability comparison? HAP1 KO/WT ratios cannot identify baseline amounts.
Older MAPP test partitions are closed. Hitlist820–823 separately own older
MAPP/PBMC material linkage; do not duplicate that work.

## Fixed source and admission audit

1. Freeze official PRIDE project/file metadata and publisher supplementary ZIP,
   with URLs, timestamps, byte hashes, size and failure receipts. Inventory every
   archive member; inspect Supplementary Tables3/5 for source datasets, quantitative
   columns, sample/run/material labels, reference IDs and confidence fields.
2. Keep this audit about human basal/noninfectious peptide quantities. No
   antimicrobial scoring, pathogen-response modeling, or activity-based peptide
   selection. Other workbook sheets remain inventoried but do not supply targets.
3. Preserve pooled identification versus peptide/run quantities, MAPP captured
   versus secreted versus MHC-bound material, measured versus transformed/imputed
   signals, peptide versus protein LFQ, and exact versus surrogate precursor links.
   Do not infer zeros/negative labels, molarity, specimen pairing or independent
   biological replication. No source ingestion or private Presto training adapter.
4. Inspect all relevant table values as well as headers. Independently reconstruct
   relevant workbook cells from original OOXML and reconcile the read-only parser.
   Publish counts, coverage and explicit unresolved requirements, not original
   biological sequences or activity-ranked results in narrative output.
5. Inspect original processed archive inventory only after metadata establishes
   size and relevance. Retrieval bounded to512MiB per processed asset,1GiB total;
   supplement256MiB, individual metadata20MiB; no RAW downloads. Stop/re-plan before
   relaxing caps. Preserve failed partial attempts and sanitized receipts.

## Decision criteria and downstream design

Availability admission requires individual positive/zero/missing signal fields,
source-backed sample/capture assignments, source sequence/origin evidence, known
quantity grain and processing provenance. A pooled ID list does not meet this.
Independent precursor and capture/detection calibration may remain unresolved;
captured signal is never relabeled free intracellular concentration.
If qualified, implement shared Hitlist source/recipe before any fit, then freeze
overlap-aware grouped evaluation, baseline sequence/length controls, and matched
exact versus sampled distribution models in a separate prospective specification.
If insufficient, identify exact source assets/fields needed and file/update a
source-specific Hitlist owner, preserving existing source-work ownership.

## Execution and closure

Local CPU only, no environment changes or cloud. No training, pretraining,
synthetic biology, optimizer, losses, assay-to-model outputs or held-out metrics
in this source audit. No test split/prediction dump: qualification precedes any
modeling. No existing held-out outcome is accessed. Record elapsed time/RSS,
scripts/environment/git state and exact launch; publish README, summary JSON,
independent verification, raw paths/hashes and canonical experiment-log entry.
No model baseline promotion; full APC availability objective remains unchanged.

- [ ] Freeze source metadata/assets and bounded inventory.
- [ ] Audit baseline quantitative and material coverage.
- [ ] Independently verify source counts/cells.
- [ ] Record shared Hitlist owner and concrete next action.
- [ ] Close report and registry; preserve unrelated task edits.
