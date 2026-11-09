# MAPP deposited sources and processed-evidence admission

October 22, 2026; Codex / GPT-6. The recovered PXD009092 archive contains a
concrete untreated-versus-cytokine cohort candidate: **2,272 peptides across 572
protein groups** meet the prespecified structural coverage diagnostic. **No rows
are admitted for training yet.** Confidence claims, processing semantics and
shared metadata/export remain unresolved. This is a source audit, not a model fit.

## Source contract and experiment identity

The [plan](../agents/codex/plans/mapp-source-admission.md) was registered before
computation. Sixteen immutable original assets total **53,293,496 bytes**: project
and file metadata for seven PRIDE deposits, the verified PXD009092 SEARCH.7z, and
an institutional copy of the [primary accepted manuscript](https://weizmann.elsevierpure.com/ws/files/111310290/ym_NatBio_RevealingTheCellularDegradomeByMassSpectrometry_AM2018.pdf).
Originals are in datacache under `data/availability_sources/`; [input manifest](reproduce/input_assets.json)
and [paths](results/input_paths.json) pin bytes and locations. Five extracted TSV
members have separate byte hashes and immutable derived paths. Their source URL
fragments identify archive members, not standalone downloadable files; the
launcher reconstructs them from the pinned archive before installing the bundle.

The paper describes proteasome-associated capture and parallel bottom-up
proteomics, with peptide filtering and protein inference. This captured pool is
not a direct measurement of free intracellular concentration or production flux.
The manuscript methods were checked on complete rendered PDF pages 22–23
(printed pages 20–21). The paper's 4,983 enriched-peptide result is **not yet
reconstructed**; the deposited peptide table contains a broader source population.

| Deposit | Deposited RAW files | Processed archive | Verified scope at this checkpoint |
|---|---:|---|---|
| PXD009092 | 18 | SEARCH.7z, 51.74 MB | Actual unspecific MAPP run map inspected |
| PXD009360 | 9 | tryptic-search.tar.gz, 733.58 MB | Candidate standard proteomics; table/sample mapping pending |
| PXD007263 | 12 | search.tar.gz, 205.72 MB | Project title identifies time points; metadata only |
| PXD009181 | 24 | tryptic-search.tar.gz, 1,971.27 MB | Time-points part 2; metadata only |
| PXD009438 | 14 | SLE-search.tar.gz, 199.79 MB | PBMC/SLE project; metadata only |
| PXD009112 | 14 | tryptic-search.tar.gz, 1,202.69 MB | PBMC/SLE project; metadata only |
| PXD009508 | 18 | search.tar.gz, 386.08 MB | Controls project; metadata only |

These are source inventory descriptions, not verified cross-deposit material
joins. The PXD009092 run map has six source-label groups of three: NO-DSP,
NO-INH, UT, CHX, TNF+IFN and BEADS. Preserve exact labels and source-backed
interpretations separately. All 18 raw-file names match the PRIDE deposit.
The other 19 summary rows are experiment aggregates and a Total row, not more
acquisitions. The complete map corrected the preliminary impression from the
first three protocol-control rows: the desired cytokine comparison is present.
A numbered suffix and a matched condition do not independently establish a paired
specimen, aliquot or donor.

## Source structure and quantitative findings

| Item | Result |
|---|---:|
| Peptide rows / unique sequences | 10,673 / 10,673 |
| Protein-group source records | 2,039 |
| Indexable unique numeric protein-group IDs | 2,003 |
| Quarantined group records | 36: 24 empty, 12 with content but no ID |
| Evidence rows | 149,959 |
| Quantitative peptide columns | 15; bead-control intensities absent |
| Peptide/run quantity comparisons | 160,095 |
| Both positive, agree with source evidence sum within rounding tolerance | 95,336 |
| Both zero | 55,483 |
| Reported zero, positive underlying evidence | **9,276** |
| Both positive but disagree / positive reported with zero evidence | 0 / 0 |

The peptide quantity columns explicitly include `MAPPbeadFilt`, while evidence
and identification columns retain all 18 experiments. The diagnostic tolerance
was prespecified at relative 1e-4 / absolute 1 intensity unit. The 9,276 zeroed
entries cannot be treated as simple lack of acquired signal. The exact filtering
recipe and its relation to the paper's mock-enrichment rule still need explicit
reconciliation. Neither original nor filtered intensity is calibrated molecules.

There are 396 reverse-flagged and 1,562 contaminant-flagged peptide rows; these
counts can overlap and are not an additive exclusion total. Source evidence types
are MULTI-MATCH 74,573; MULTI-MSMS 72,496; MULTI-SECPEP 1,599; MSMS 1,291.
Keep direct-MS/MS evidence separate from transferred identification. The reported
peptide PEP is not interchangeable with a dataset-level FDR threshold.

Three confidence assertions need independent scope and provenance: repository
project metadata says peptide FDR 0.1, the paper says peptide FDR 0.01, and the
search parameter table reports PSM FDR 0.01 and protein FDR 1. A PSM threshold
cannot silently resolve a peptide-level discrepancy. Source reference is named
`Human_UP_2016_02.fasta`, but its exact bytes/hash are not supplied here.

The malformed protein-group rows retain source locators, occupied-field summaries
and row hashes in [quarantine](results/quarantined_group_rows.json). No adjacent-row
repair was attempted. All peptide/evidence and indexable group IDs resolve in the
executed source audit, with no reported peptide/evidence sequence or reciprocal
peptide-group mismatch. This does not prove every long group-list cell is intact.
The group's fragment-derived intensity/LFQ columns are **not independent precursor
protein measurements**, despite the `proteinGroups.txt` filename.

## Prospective contrast support, not training eligibility

The registered diagnostic requires a canonical peptide, no reverse/contaminant
flag, exactly one protein-group ID, positive reported intensity in at least two
of three runs in each UT/TNF+IFN arm, and at least one direct-MS/MS-bearing run in
each arm. It finds **2,272 peptides / 572 groups**. Of those, 271 groups contain
multiple supported peptides, covering 1,971 peptide rows. This is an observed
support set, not an estimate of all native products or their molecular probability.

The [row ledger](results/peptide_support_ledger.csv.gz) retains all 10,673 peptides
with separate flags and a deferred admission decision. No unreported precursor
value, missing peptide, loss condition or biological replicate identity is invented.
The earlier 129 synthetic fits remain optimizer evidence only; no biological
model or APC generalization has been demonstrated by this audit.

## Reproduction, failure preservation and verification

Started from dirty Presto `30cc5368bfa58b936339c3e5600309625831d851`.
[launch.json](reproduce/launch.json) records source/input hashes, overrides and dirty
state. Frozen experiment-local code contains all executed logic; no model is
imported. Attempt 1 stopped on nonunique empty protein-group IDs. Its source,
invocation, log and partial outputs remain under `attempts/01-duplicate-empty-group-ids/`.
The amended plan requires row quarantine and unique numeric IDs before retrying.
Attempt 2 completed in **17.170 s**; independent verification took **3.008 s**.

```bash
AUDIT_PYTHON=/Users/iskander/code/shared-virtual-env/bin/python bash experiments/2026-10-22_1630_codex_mapp-source-admission/reproduce/launch.sh
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_1630_codex_mapp-source-admission/reproduce/source/verify.py
```

Local macOS arm64 CPU, Python 3.12.6; independent verifier pandas 3.0.6 /
NumPy 2.5.4. It does not import the audit code. It re-parses original tables,
verifies 16 original hashes, five member hashes, ten result hashes, 149,959 source
evidence rows, all 160,095 quantity comparisons, all 10,673 support-ledger rows,
and the 36 quarantined group records. [Verification passed](verification/receipt.json).
Pandas emitted deprecation warnings for `set_index(verify_integrity=...)`; they
were not errors or suppressed checks. Explicit Ruff E4/E7/E9/F passes.

Artifacts: [source/header inventory](results/table_inventory.json),
[deposited series](results/deposited_series.json), [run maps](results/runs.json),
[summary](results/summary.json), [all quantity comparisons](results/quantity_comparisons.csv.gz),
[parameters](results/parameters.json), [output hashes](results/output_hashes.json),
[runtime](results/runtime.json), source snapshots and `launch_logs/`.

No training/pretraining, architecture, parameters, loss weights, synthetic labels,
assay-to-head mapping, requested/observed GPU or Modal applies. There is no
train/validation/test split or predictive dump: this audits source membership and
quantities, not generalization. IC50 correlations and 500 nM binding metrics are
inapplicable. No optimizer winner is selected.

## Status and next step

Source inspection and independent audit are complete. Biological admission remains
pending: resolve confidence and filter semantics, material/context assertions and
reference mappings through the shared Hitlist system. Inspect the candidate
PXD009360 processed archive before assigning independent precursor abundance or
exact pairing. Then freeze an endpoint based on observed same-peptide condition
changes and compare exact versus sampled optimization under the identical target
and protein-group holdouts. Keep that endpoint separate from the full concentration
goal and reserve professional-APC evidence for later validation.


### Completed proteome retrieval and run-header check

PXD009360 retrieval finished successfully. All **733,580,954 bytes** match the
source checksum (matching SHA-1), gzip integrity passes, and 18 archive-member
paths are safe and unique. [Receipt](retrievals/tryptic_receipt.json) pins SHA-256
`45b1f2b278937b43a40913798b96c7fb31a0d1a7753ebcf99951287c0ebe91c7` and the immutable datacache path. Integrity verification
took 8.349 s. The original 16-input MAPP audit remains unchanged;
this later candidate has a separate receipt and is not an admitted source table.

The [run-header check](retrievals/tryptic_headers.json) took 6.938 s:
19 source summary records include **nine actual Trypsin/P, Specific runs**, all
matching deposited RAW files. Their experiment labels are bare numbers448–456;
no treatment or exact MAPP specimen mapping is established. Do not infer arm
assignments from numeric order. Search parameters report PSM and protein FDR0.01
and matching between runs enabled. Quantitative tables have not yet been audited.
[Retrieval invocation](reproduce/tryptic-retrieval.json) preserves the curl command,
source metadata and verifier command. Header reproduction:

```bash
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_1630_codex_mapp-source-admission/reproduce/source/peek_tryptic.py
```

Next obtain the source-backed sample map and inspect quantities; an available
tryptic archive alone does not establish independent precursor pairing.
