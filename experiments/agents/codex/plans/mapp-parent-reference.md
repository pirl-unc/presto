# MAPP parent-reference and span mapping audit

Prospective specification, October22,2026; Codex / GPT-6. Previous goal turn was
progress:4d938e0 reconstructed all MAPP quantities and publisher memberships.
This family qualifies sequence context for the first biological availability
comparison; it is not itself training or a new production curation registry.

## Sources and identity contracts

Reuse original hash-pinned PXD009092 peptides/evidence/group tables and the verified
publisher membership/structural-support ledgers. Keep every source protein and
protein-group alternative, leading-razor assignment, one-based start/end and source
flags. MAPP reference filename is Human_UP_2016_02.fasta; its exact bytes remain
unavailable. A later reference must not be relabeled as the original search database.

Two independent reference representations:

1. Existing human reviewed Swiss-Prot DAT gzip from the Weirdo source cache,
   dated2023 by its original file and entry metadata, SHA256
   defd9fd7fdc3be6f36b5b57b1a2c29978b956612e1437ec15f7065b055bbe9d3.
   Preserve primary/secondary accessions, entry identifier, taxonomy, sequence
   version and canonical sequence. Secondary aliases may be ambiguous; retain
   all candidates. Do not invent isoform sequences by deleting accession suffixes.
2. Official UniProt reviewed human FASTA stream including isoforms, retrieved now.
   Freeze exact query URL, response headers/release metadata, retrieval time, bytes
   and SHA256 before mapping. Preserve exact primary/isoform accession identifiers.
   This reviewed-only scope does not prove uniqueness across unreviewed human
   proteins, historical references, variants or other species. No default alias
   transfer between reference generations.

Use both reference generations read-only; install immutable copies in datacache.
No shared Python environment changes, cloud execution or unbounded downloads.

## Fixed comparisons

- Audit every deposited peptide (including flagged source records as separately
  reported strata) against every source-listed accession in each reference.
  Resolve source-primary/exact-isoform IDs, and explicitly recorded secondary
  aliases only for the DAT generation. Classify missing ID, one/many reference
  candidates, sequence absent, one/many exact occurrences. No approximate/I-L
  matching or sequence alteration. For leading-razor mappings additionally compare
  supplied one-based-inclusive coordinates and source flanks when available.
- Scan the complete declared reference against all deposited peptide strings to
  retain additional exact parent/position alternatives. Use an exact six-residue
  prefix index and verify full substring equality at every candidate. Include
  overlapping occurrences, terminal spans and isoform duplicates. Keep distinct
  identity count separate from distinct full-sequence count. Reference scope is
  part of every uniqueness claim. No affinity-based candidate filtering.
- Report all-source, publisher-membership and prior structural-contrast subsets
  separately. The prior structural subset remains frozen; do not tune source
  admission for improved cohort size or model metrics. Count matched positions,
  ambiguous/unresolved mappings, parent support and peptide length/search coverage.
- Freeze full model-reference sequences and positions only for traceable matched
  parents. A source-coordinate mismatch is recorded separately from an exact
  peptide occurrence elsewhere; do not silently repair source coordinates.
- Build descriptive connected components from all verified parent alternatives
  for future leakage control; do not label these homology clusters or draw/inspect
  predictive train/validation/test splits in this audit.

## Verification and closure

Use an independent Biopython parse of source DAT/FASTA and direct substring checks
for all emitted positive occurrences and source-listed negatives. Cross-check
reference entry counts, IDs, sequences, sequence hashes, source-cell mapping and
coordinate conversions. Test the prefix scan on overlapping/repeated sequences
and boundary cases before full scan. Record missing/ambiguous states rather than
forcing full coverage. Preserve all inputs, scripts, commands, code/dirty state,
runtimes and row-level ledgers under this experiment. Update the canonical log and
shared Hitlist746 acceptance evidence. No model/pretraining/loss, synthetic
training data, GPU, predictive split or predictive metrics/dumps apply; synthetic
strings are software checks of exact matching only. The overall concentration/
APC objective remains active regardless of mapping coverage.
