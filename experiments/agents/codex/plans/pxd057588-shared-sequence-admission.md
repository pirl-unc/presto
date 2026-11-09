# Shared PXD057588 generation: reference and precursor-support admission

Prospective specification, October22,2026; Codex / GPT-6. Previous goal turn was
progress:6cd7edc published the complete MAPP reference audit and Hitlist752.
The newly merged Hitlist749 now materializes PXD057588 in the common package.
This family consumes that actual generation; it does not create another curation
schema or bypass the pending unified training recipe interface in Hitlist739.

## Frozen input and scientific question

Pin the entire87-file Hitlist1.82.22 external generation at
/private/tmp/hitlist-v2-external-18222-audited, manifest SHA256
`daff7930e193c6dbde1fe84c065b024187232147fa006d9348e297766e95e69c`.
Use the exact source adapter/validator from merged PR749 headbac9c7f; preserve
its source/version hashes. Verify the package read-only with its native verifier
and a separate hash/inventory check. No shared environment changes or edits to
another thread's Hitlist checkout. The total physical manifest scope is about
3.93GB, including embedded predecessors; original17external assets total11.5MB.

The primary population is all35,772 typed peptide rows from Tables7–12, comprising
10,662MHC-bound and25,110digest rows; sequences recur between tables. A schema
inspection finds30,587distinct uppercase amino-acid sequences, lengths6–53, with
one nonempty accession-like parent label per row and no delimiters. Preserve
raw labels/sequence and source cell IDs, not an invented parsed multi-parent list.
Mouse species is explicitly reported in the pinned PXD057588 project metadata;
prior primary-paper review scopes these six panels to the mouse experiments.
Do not use the human MAPP reference, or assign tumor-partner losses to macrophages.

Does a verified mouse sequence reference recover the reported parent context,
and which MHC/digest rows actually share possible parent sequences rather than
only string labels? This is necessary for a later precursor-adjusted observable
comparison, not evidence of free intracellular concentration or exact specimen
pairing. Same material labels and channel names remain insufficient.

## Reference contract and fixed analysis

Retrieve the official reviewed mouse UniProt FASTA including isoforms using
taxonomy10090. Freeze exact URL, release headers, time, bytes and SHA256 before
mapping; install through datacache. Preserve original protein headers, primary
and explicit isoform IDs, full sequences/hashes and taxonomy scope. It is a model
reference, not assumed identical to the original study search database. No
suffix stripping, unrecorded alias transfer, I/L equivalence or approximate match.

Run all valid source sequences against every entry with the already verified
six-residue-prefix matcher, retaining every overlapping position, including
termini. Query sequences may be deduplicated by their exact strings; source row
IDs remain separate through a many-to-one row/sequence relation. Record exact
source-ID resolution, missing ID, present-ID/absent-peptide, and additional
reference alternatives. Distinguish parent identities from distinct full sequence
strings. Preserve every unmatched row; do not tune reference/admission for size.

Report all-source and all six table/pool strata. For Tables9/11, quantify literal
label overlap versus exact-ID-and-sequence-supported overlap and full-sequence
alternative overlap. Show parent groups with at least2MHC and2digest rows only
as a support diagnostic, not a source filter. Preserve one measurement identity
through alternative-parent joins. Build descriptive peptide-parent components
from all sequences across tables for later split control; these do not establish
homology or specimen independence. Do not draw predictive splits.

Inspect the shared package's context, scale, material and protocol fields needed
by the later model recipe. Pin source-backed mouse species and assay roles,
record fields still only available as raw assertions, and specify any missing
shared normalization in Hitlist739/752 rather than inventing Presto defaults.
No inference that a source total is an independent precursor amount. A proposed
precursor adjustment must separately specify its peptide-to-protein aggregation
and exact versus condition-surrogate pairing; this audit creates neither.

## Independent verification and closure

Use Biopython for independent FASTA parsing and the already separately verified
character automaton for a complete alternative scan. Compare all occurrence sets,
source mappings, table row counts, sequences/hashes and shared-parent relations.
Verify source-row identity and package content before/after analysis. Preserve
all scripts, exact inputs, commands, code/dirty state, runtimes and row-level
mapping outputs in this registered family. Reuse frozen prior scan code without
changing those historical files. No raw-MS search, model/training/pretraining,
loss, synthetic training, GPU/Modal, predictive split or prediction dumps apply.
Native quantities remain source-relative MHC/digest observations. Report metadata
admission gaps and the next implementable biological comparison, without calling
this concentration validation or completion of the broader availability goal.
