# Shared PXD057588: mouse sequence and precursor-support audit

Completed October 22, 2026; Codex / GPT-6. The actual shared Hitlist generation
now has verified sequence context for **35,771 of 35,772** primary peptide rows.
This advances the biological comparison beyond workbook-only inventories, while
retaining source uncertainty. No model fit or concentration qualification occurred.

## Source and execution contract

Consume the 87-file Hitlist 1.82.22 external generation, manifest SHA256
`daff7930e193c6dbde1fe84c065b024187232147fa006d9348e297766e95e69c`.
Its native verifier from exact PR749 head `bac9c7f` and a separate complete
inventory/hash check both pass. All package bytes were rechecked after mapping.
The input generation totals 3,932,771,529 manifest-listed bytes, including its
predecessors. It is not a newly curated Presto registry. The 34 external-source
files plus the parent manifest and reference are independently copied through
datacache, with distinct source/destination inodes and verified hashes.

The source population is all 35,772 typed primary rows: 10,662 MHC-bound and
25,110 global-digest rows, 30,587 distinct sequences of length 6–53. Original
cell locators, parent labels, sample/channel claims, missingness and quantity
representations survive unchanged. No source quantity was multiplied by a
one-to-many mapping join. The original 309,126 intensity results and 509,850
quantity representations remain separate grains. No held-out split was drawn.

Mouse species is explicit in the pinned PRIDE project metadata; prior source
review scopes these six panels to the mouse experiments. The query uses taxonomy
10090. The official response declares UniProt release 2026_03 (September 2, 2026),
retrieved October 22 at 19:16:18 UTC. The 18,126,470-byte FASTA has SHA256
`41bc0fa6cb9118ba95f330a27b18c39559e9397b6d99d1e73cc094869c99787a`.
It contains 25,758 reviewed entries including isoforms, 25,737 distinct full
sequences and 15,203,685 residues. This is an explicit model reference, not a
claim to have recovered the original study search database. No secondary-alias
transfer, isoform-suffix stripping, approximate or I/L matching occurs.

## Complete mapping results

Every exact overlapping occurrence is retained. Source labels are separately
classified as missing ID, resolved-ID/peptide-absent, or exact sequence match.

| Table | Source rows | Any reference match | Source-parent match | Multiple full parent sequences |
|---|---:|---:|---:|---:|
| 7 | 2,824 | 2,824 | 2,819 | 1,014 |
| 8 | 3,554 | 3,554 | 3,548 | 1,250 |
| 9 | 1,309 | 1,309 | 1,306 | 463 |
| 10 | 2,613 | 2,613 | 2,609 | 952 |
| 11 | 25,110 | 25,109 | 25,103 | 7,926 |
| 12 | 362 | 362 | 362 | 120 |

Across all rows, 35,747 match their reported parent ID; 24 reported IDs are
missing and one resolves to a sequence without the peptide. There are 52,277
position alternatives over the deduplicated sequence universe. 24,046 source
rows match one distinct full sequence, while 11,725 match multiple sequences.
Reference identity and full-sequence counts remain distinct. All unmatched and
ambiguous source rows remain in the ledgers.

The complete peptide↔parent graph spans 5,374 connected components; the largest
contains 862 distinct measured sequences across tables. Components retain paths
through other assays and define possible split constraints. They are not homology
clusters, independent biological subjects, or evidence of exact specimen pairing.

## Ligand/proteome support in Tables 9 and 11

| Parent definition | Shared parent keys | Keys with ≥2 ligand and ≥2 digest rows | Distinct ligand/digest rows in those multi-peptide groups |
|---|---:|---:|---:|
| Literal source label |663|110|258 /1,919|
| Source label plus verified peptide occurrence |661|110|258 /1,919|
| All exact reference identity alternatives |1,136|236|277 /2,128|
| All distinct full-sequence alternatives |1,134|236|277 /2,128|

The verified-source-label intersection covers 827 ligand and 6,057 digest rows in
total. Additional isoform/parent alternatives expand candidate support, not the
number of independently measured proteins. The support ledger retains exact
source-row sets for each definition, so parent multiplicity cannot inflate the
number of measurements.

These results support designing a **condition-level precursor-surrogate
comparison** once the shared recipe interface is ready. They do not authorize an
exact-specimen join: Tables 9/11 use different channel maps, and the material links
remain `assay_channel_unresolved_specimen`. Table 11 measures digest peptides;
aggregation into a precursor-protein estimate requires a declared policy and
sensitivity analysis, not a renaming of the source quantity.

## Remaining recipe requirements and next biological hypothesis

Use the shared Hitlist #739 typed query/recipe path and carry this reference mapping
as evidence in that system. Species currently survives in the original project
asset; the external typed source-assertion predicates do not include a normalized
species assertion. Keep molecular/APC/partner roles and source evidence explicit,
not implicit human defaults. The current package has no exact specimen links.
Table 10 loss belongs to tumor partners, not macrophage processing machinery.
Preserve the Table 8 polarization and Table 10 replicate-description conflicts.

The next narrow hypothesis is whether sequence/context information improves
held-out **observed condition-dependent MHC/digest signal** beyond a source-parent
and condition baseline. A separately registered recipe should compare unadjusted
signal with an explicitly labeled condition-surrogate precursor adjustment on
verified support, using the same holdouts and observable loss for exact and
REINFORCE estimators. Preserve alternative-parent policy and complete-case
selection. No missing peptide is a biological zero. Validation must include
nonconstant predictions, baseline comparisons, sample/work/runtime accounting
and source-linked prediction dumps; a positive result would still concern the
combined processing/presentation/observation pathway.

This source alone does not identify free intracellular concentration, native
processing flux, independent MS response, class II processing or APC machinery-loss
effects. MAPP ingestion is separately tracked in Hitlist #752. Those limitations
remain part of the full active availability objective rather than being silently
removed from its completion criteria.

## Independent verification, artifacts and reproduction

The producer reuses the frozen six-residue-prefix scanner from the MAPP audit.
The independent verifier uses Biopython and the separate character-level
Aho-Corasick algorithm over the complete reference. It verifies every emitted
span/flank, all 35,772 source mappings, all 30,587 sequence summaries, all 26,449
parent-support records and all seven table/global summaries. It also compares
71,544 sequence/parent cells directly to the original six publisher workbooks.
All 36 copied input hashes and 10 primary output hashes pass. Matched parent sequences,
source mappings, position alternatives and support sets are preserved, including
missing IDs and the one sequence-negative resolved candidate.

Runtime: full package verification 10.899 s; mapping 7.591 s; independent replay 8.975 s.
Local CPU only, Python 3.12.6. Exact package/code/environment/source receipts are in
`reproduce/`; immutable inputs are indexed by `results/input_paths.json`.
No environment modifications, GPU/Modal, pretraining, model, loss, synthetic
training, validation/test split or predictive metrics apply. This is an admission
and sequence-context audit; prediction dumps are inapplicable.

[Summary](results/summary.json), [independent replay](results/independent-verification.json),
[package verification](results/package-verification.json), and
[context/quantity contract](results/context_and_quantity_contract.json) contain
machine-readable evidence. Frozen code/snapshots preserve dirty launch commit
6cd7edc. `reproduce/launch.sh` records the exact three commands; use a fresh family
under this repository with the same pinned inputs to rerun. Completed outputs are
protected against overwrite. Large reusable inputs stay in datacache.

Data attribution: Cui et al., Cancer Research 2025,
[DOI 10.1158/0008-5472.CAN-24-4674](https://doi.org/10.1158/0008-5472.CAN-24-4674),
PXD057588; publisher supplements CC BY 4.0 as recorded by the shared source.
Reference data: The UniProt Consortium, [UniProtKB](https://www.uniprot.org/),
[CC BY 4.0](https://rest.uniprot.org/help/license); derived tables select/reformat
records and add exact-match annotations without altering sequence strings.

Status: complete verified shared-source reference audit. Biological recipe
admission, fitting and the overall APC availability/concentration goal remain open.
