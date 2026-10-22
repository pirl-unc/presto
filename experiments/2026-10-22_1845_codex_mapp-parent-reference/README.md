# MAPP parent-reference and exact-span audit

Completed and independently verified, October 22, 2026. **Codex / GPT-6.**
This audit supplies sequence-context evidence for a biological differentiable/RL
availability comparison. It does not fit a model or admit a concentration endpoint.

Of the frozen 2,272 structurally supported untreated/TNF+IFN contrast peptides,
**2,269 have exact matches in the reviewed human isoform-inclusive reference**.
However, 1,256 match multiple distinct full protein sequences, and only 1,676
have a source leading-parent match at the reported coordinates with matching
flanks. Preserve these as different facts: source assignment, verified span,
possible parent, and unique origin are not interchangeable.

## Inputs and reference contracts

The original PXD009092 search names `Human_UP_2016_02.fasta`; its bytes remain
unavailable. Neither reference below is a reconstruction of that search database.

| Reference contract | Entries | Distinct full sequences | Residues | Isoform IDs |
|---|---:|---:|---:|---:|
| Local 2023 reviewed human DAT, canonical sequences; recorded secondary aliases retained | 20,422 | 20,347 | 11,392,335 | 0 |
| Official UniProt 2026_03 reviewed human FASTA, including isoforms; exact IDs only | 42,562 | 42,480 | 24,515,652 | 22,131 |

The first snapshot is dated by its local file/entry metadata, not a verified
original release manifest. Its 108,046,188 compressed bytes have SHA256
`defd9fd7fdc3be6f36b5b57b1a2c29978b956612e1437ec15f7065b055bbe9d3`.
The official response declares release 2026_03, released September 2, 2026;
retrieved October 22 at 18:43:18 UTC. Its 29,338,408 bytes have SHA256
`8c4459cc1e190caad54150511e5a0b5779863557f88387e5a86647a48d19a9d4`.
The response headers, query URL, completion log and retrieval receipt are retained.
The download tool's final exit status was not recovered, so transport success is
not inferred from a missing handle: the complete HTTP200 body was hash-pinned and
parsed independently in full by both implementations.

The 2023 DAT contains 362 accession aliases that resolve to multiple records.
All alternatives are retained. Isoform suffixes are never stripped, and secondary
aliases are not silently transferred to a different reference generation.
The two contracts differ in release, isoform scope and alias handling; their
coverage difference is not a controlled estimate of an isoform-only effect.
Uniqueness is limited to each declared reviewed-human reference, excluding
unreviewed entries, historical sequences, variants and other species.

Five original deposited table members, the structural-support ledger, and publisher
membership ledger retain the hashes/locators from the preceding
[source audit](../2026-10-22_1630_codex_mapp-source-admission/README.md) and
[publisher reconciliation](../2026-10-22_1736_codex_mapp-publisher-reconciliation/README.md).
The mapping population is all 10,673 deposited peptide rows. The 8,717 rows
without reverse/contaminant flags, publisher populations of 4,983/5,311, and
frozen structural subset of 2,272 are reported separately. No eligibility policy
was tuned. Reverse/contaminant rows stay flagged even if they match human sequence.
All original inputs are installed through datacache; immutable paths are in
[results/input_paths.json](results/input_paths.json).

## Verified coverage

Here, a match is literal amino-acid substring equality, including overlapping
occurrences. There is no I/L equivalence, approximate search, affinity filter,
sequence repair or assumed biological origin. Coordinates are one-based inclusive;
`-` denotes a reference sequence boundary in derived flank fields. Original
coordinate/flank cells are preserved alongside comparisons.

| Population / mapping result | 2023 canonical | 2026_03 with isoforms |
|---|---:|---:|
| All deposited peptides with any exact match | 8,753 / 10,673 | 8,790 / 10,673 |
| Unflagged source peptides with any exact match | 8,654 / 8,717 | 8,691 / 8,717 |
| Publisher untreated peptides with any exact match | 4,960 / 4,983 | 4,973 / 4,983 |
| Publisher TNF+IFN peptides with any exact match | 5,286 / 5,311 | 5,301 / 5,311 |
| Frozen contrast peptides with any exact match | 2,262 / 2,272 | 2,269 / 2,272 |
| Contrast peptides matching a resolved source-listed parent | 2,261 | 2,258 |
| Contrast peptides matching resolved leading parent anywhere | 1,703 | 1,700 |
| Contrast peptides matching leading coordinates and flanks | 1,678 | 1,676 |
| Contrast peptides with a resolved leading-coordinate conflict | 26 | 26 |
| Contrast peptides matching one distinct full sequence | 2,030 | 1,013 |
| Contrast peptides matching multiple distinct full sequences | 232 | 1,256 |
| Contrast peptides with additional non-source-listed matches | 18 | 1,114 |
| Distinct possible parent identities in contrast subset | 640 | 1,232 |
| Contrast peptide-position alternatives | 2,613 | 4,529 |
| All-source connected components intersecting contrast subset | 540 | 542 |
| Largest such component, counted within contrast subset | 175 peptides | 175 peptides |

The three current-reference unmapped contrast peptide IDs are 5883, 8209 and 9663.
Two source IDs cannot be resolved in this reference; the third resolves but its
peptide sequence is absent. Their source records remain present and unresolved.
The 1,013 single-full-sequence matches comprise 1,012 single-identity matches
plus one peptide matching different identities with an identical full sequence.
Identity counts and sequence counts are both retained.

Across all source rows, the two scans emit 14,258/23,463 exact position alternatives
and freeze 1,796/3,537 matched parent entries with full sequences. The source-ID
ledger has 32,269 rows per reference, including absent IDs and resolved sequences
with no match. A position elsewhere in a protein does not repair a disagreeing
reported position. A new reference match does not supersede a source assignment.

Connected components use the complete all-source peptide↔parent graph, not just
the contrast subset. Subset counts refer to intersections with those components;
paths through other deposited peptides are retained. These are exact-parent-sharing
groups, **not homology clusters or validated independent partitions**. No train,
validation or test split was drawn or inspected.

## Consequence for biological fitting

The isoform-inclusive generation provides broader candidate context and is the
preferred prospective model-reference basis, with all ambiguities retained.
Do not turn one measured peptide into multiple independently weighted labels for
its alternative parents. A subsequent model must either marginalize a declared
parent mixture or use a prospectively specified unambiguous diagnostic subset,
with explicit coverage costs. A leading-razor label cannot supply a measured
parent-mixture weight. The graph gives a necessary splitting constraint; further
sequence/homology and specimen checks remain necessary.

The biological label remains **relative proteasome-associated captured signal in
HEK293**, not free intracellular peptide concentration or processing flux. The
original search covers 6–40 residues; lack of reported shorter fragments provides
no absence labels. The full availability objective still includes nonbinding and
short products, APC machinery/context effects, precursor abundance and downstream
presentation/detectability. This audit does not establish those endpoints.

Remaining data work includes the shared source-generation import, the conflicting
peptide-FDR statements, original-reference lineage, and the unassigned parallel
tryptic-proteome runs. Source ingestion can preserve incomplete evidence; endpoint
admission must specify which unresolved facts matter to its quantity and split.
The merged [Hitlist PR #749](https://github.com/pirl-unc/hitlist/pull/749) implements the
shared external-source mechanism and PXD057588; MAPP remains a contract
candidate rather than an imported source generation. This audit supplies evidence
for that same system, not a second curation schema.

## Verification and reproducibility

The producer uses a six-residue prefix index with complete substring checks.
Its boundary/overlap/nested-prefix/absent tests passed before the full scan.
The independent verifier uses **Biopython 1.88** for both original references and
a character-level Aho-Corasick automaton for the full peptide/reference scan.
It reproduces all **37,721** emitted positions, including complete-set equality
against a scan of every reference residue. It also checks every source candidate
with overlapping regular expressions, every emitted span/flank, all source-row
hashes, all reference identifiers/sequence hashes, every per-peptide summary,
all 14 subset summaries and connected components via a separate graph traversal.

- 62,984 reference records independently parsed.
- 64,538 source-accession mapping rows independently replayed.
- 26,978 source-positive candidates and 14 source-negative candidates checked.
- Nine input hashes and 15 primary output hashes checked.
- Producer 11.537 s; independent verification 17.734 s, local CPU only.
- Source audit only: no model, pretraining, synthetic training data, loss,
  assay-output fit, GPU/Modal job or predictive evaluation. No validation/test
  metrics or prediction dumps apply because no model or split exists.

Exact executed sources and git/environment receipts are under `reproduce/`.
Launch commit 4d938e0, dirty only the already recorded documentation/experiment
work. The shell commands and source snapshots pin both executed implementations.
The reference retrieval is separately recorded in `retrievals/receipt.json`.
Reproduction uses the frozen inputs, not whatever a future mutable UniProt URL
returns; [launch.sh](reproduce/launch.sh) takes a fresh experiment directory under
this repository and refuses to overwrite an existing result.

Primary artifacts: [summary.json](results/summary.json),
[independent-verification.json](results/independent-verification.json),
[subset_summary.csv.gz](results/subset_summary.csv.gz), source-field and accession
mapping ledgers, complete position ledgers, reference inventories, and matched
parent sequences. Large original references live in datacache; the experiment
contains derived ledgers plus explicit immutable paths/hashes.

Sequence/reference data attribution: **The UniProt Consortium**,
[UniProtKB](https://www.uniprot.org/), under its
[CC BY 4.0 database license](https://rest.uniprot.org/help/license).
The derived files select and reformat records and add exact-match annotations;
original sequence strings are unchanged. Accession/isoform distinctions follow
the [UniProt manual](https://web.expasy.org/docs/userman.html).
MAPP measurements originate from Wolf-Levy et al.,
[Nature Biotechnology 2018, DOI 10.1038/nbt.4279](https://doi.org/10.1038/nbt.4279),
PXD009092. No source endorses the derived mapping or modeling interpretation.

Status: reference audit complete; biological source admission and the broader
APC availability/concentration objective remain open.
