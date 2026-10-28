# Shared multi-output training and 2026 study holdouts

Status: **CPU source preparation is running; predictive training has not started.**
Agent/model: Codex / GPT-6. Date: 2026-10-28 UTC.

[Prospective specification](../agents/codex/plans/2026-10-28_temporal-multi-output.md)
· [live report](http://127.0.0.1:54502/dashboard.html)
· [canonical registry](../experiment_log.md).

## Source and admission contract

Fresh canonical IEDB and CEDAR full exports were downloaded with Hitlist 2.0.40,
10,066,712,903 and 9,188,295,774 bytes respectively; see
[acquisition receipt](results/acquisition.json). The 5,208 source reference entries
include 45 distinct papers dated 2026, with PubMed publication-date XML retained.
A metadata-only inventory finds **no direct IC50, KD or EC50 measurements in those
2026 exports**. The user therefore authorized older whole-assay-study holdouts.
These are new from-scratch study splits, not claims that historical data were never
used in earlier architecture development.

Use canonical Hitlist indexing/export and the shared typed Presto adapters.
Admit unmodified peptides with resolved human MHC groove sequences from the pinned
mhcseqs catalog. Preserve every reported candidate MHC; do not assign a cognate
allele to a multi-allelic sample. Ordinary explicitly allele-resolved human DR may
use a declared DRA*01:01 reference, preserving original typing. Explicit partners,
construct exceptions, DP/DQ and unresolved serotypes retain stricter rules.
Molecular MHC species, antigen/source organism and experimental host are separate.
This is the compatible human-MHC cohort, not a claim to train on every raw export row.

Physical measurements remain distinct: IC50, KD, EC50, KD-via-IC50, KD-via-EC50,
half-life, off-rate, on-rate and td50. Literal qualitative binding and observed
elution are separate binary outputs. Bounds keep their original inequality;
training scales use exact TRAIN values only. Allocate only supported TRAIN heads.
Shared source implementation is pinned at `c8f032a7a200a1ea87121b016eb9dbb22c90e3db`;
[complete runtime file hashes](reproduce/runtime-pins.json) include 211 files.

## Frozen selection and model recipe

Prespecified 2026 test PMIDs: 41534530 (class-II MS), 42069060 (class-I MS),
42048449 (class-I KD-via-IC50), subject to actual source admission. A deterministic
metadata-count rule adds older assay studies for test and validation. Source-study
alias components remain whole; exact peptides are disjoint, with test > validation
> train priority. Final membership, exclusions and numerical budgets are pending
canonical admission and will be hashed before any fit.

One shared-v5 wide 192/3/6/768 model and the canonical observation-manifest trainer,
seeds 42/43, no pretrained weights, dropout .1, AdamW 3e-4/decay1e-4, batch128.
Relative label frequencies: 60% elution,20% IC50,15% other physical outputs,5%
qualitative binding; within families, endpoint/class frequencies use square-root
counts. Canonical weighted quotas cover all unique TRAIN rows. Budget is one full
weighted epoch or6,000 updates, whichever is larger. Validate every1,000 updates;
equal-endpoint loss selects the checkpoint. Full validation/test dumps; a fixed
16,384-row cap applies only to post-fit TRAIN diagnostic predictions.

All outputs use the same peptide/groove encoder and full ordered ligand context.
Unsupported source-protein/cellular-context inputs are not allocated. Their source
annotations remain available for stratified reporting. This task does not qualify
native DO:DM, TCR or independently identified processing/detector mechanisms.

## Ligand comparison populations

Shared canonical backgrounds match positive length, MHC candidates, context and
study: one per TRAIN positive, five per evaluation positive, seed20261028, with
replacement. Use exact same-protein mapping when available, otherwise report the
frozen human Ensembl112 proteome population separately. Exclude all known positive
peptide sequences and other-split sequences. These are unobserved comparisons,
not measured non-binders. Report length-only controls and same-protein/fallback
AUCs separately. The complete positive blacklist includes held-out sources for
conservative background exclusion only.

Source-reported MS negatives are retained as **assay-scoped nondetections**, with
study-specific AUCs separate from generated backgrounds. A large historical source
is HLA-B27 transgenic rat/ERAP1 data, so human MHC must not be relabeled as human
host or a universal non-presentation claim. Mixed/unknown conditions remain explicit.

## Execution and verification

Source preparation: CPU8,96GiB RAM. Attempt2 failed at Hitlist's default16GiB
provenance scratch limit; raw downloads survived. Attempt3's explicit256GiB disk
request was rejected before a worker started. Attempt4 asserts128GiB scratch,
16GiB reserve and512GiB requested ephemeral disk. All attempt receipts/logs are
preserved. Dropped local log/controller connections do not restart the remote build.

Predictive resources, when launched: requested `H100!`, maximum two workers,
CPU8/64GiB RAM each. Actual GPU/memory and runtime metrics will be recorded by
workers. There are no predictive results or preferred new checkpoint yet.

Software evidence: the current shared suite passes2,805 tests/four skips;
the literal purified-method and reported-MS-negative admission checks passed79
and89 focused tests, respectively. Seven artificial preparation/report fixtures pass,
including the real ObservationIndex/common-trainer preflight, independent source/background audit and separation of
reported-negative versus generated-background AUCs. Artificial tests are not
biological performance evidence.

## Reproduction and artifacts

- `code/acquire.py`, `build_index.py`, `admit.py`: canonical source stages.
- `code/partition.py`, `prepare_worker.py`, `prepare.py`: fixed cohort preparation.
- `code/launch.py`, `worker.py`: two-seed common-trainer fits and live validation.
- `reproduce/`: source snapshots, exact invocations, failed-attempt dispositions,
  dependency call IDs and pinned package/reference hashes.
- Modal volume `presto-checkpoints`, `/runs/2026-10-28_1540_codex_temporal-multi-output/`:
  source downloads/index, admission SQLite, prepared manifests and eventual raw fits.
- Local large artifacts: repository `artifacts/2026-10-28_1540_codex_temporal-multi-output/`.

Final closure still requires: all fits, complete validation/test prediction dumps,
independent metric/source checks, selected-checkpoint TRAIN-probe serving replay,
per-output/class/study summaries and uncertainty, canonical registry/model-reference
updates and the shared recipe handoff. PR131 stays draft pending scientific gates.

## Fresh source build and admission progress

The full canonical build completed at 2026-10-28 17:58 UTC, yielding 5,844,451 MS,
821,470 binding and26,694 other-assay rows. The source inventory is large (298.8MB)
and lives at the hashed location in `results/source-inventory-location.json`.
The canonical upstream build excludes class-II peptides shorter than12 residues,
including150,931 MS and3,273 binding rows. This is an upstream curation policy,
not a claim that shorter class-II ligands cannot bind.

Admission attempt4 failed during remote launcher import, before data processing;
local filesystem upload paths are now constructed only locally. All three
launchers passed remote-path import checks. Attempt5 is running against the
completed source build. The finished binding pass admitted263,432 candidate
physical/qualitative measurements, including I/II direct IC50 counts19,167/38,543.
61,548 rows across endpoints use the declared reference DRA. Final split totals
and any lineage-conflict exclusions remain pending.

The source also contains468,548 high-throughput multiplexed binding records not
covered by the frozen soluble-affinity/qualitative-purified adapter. Of these,
418,890 are the peptide microarray study PMID32903714; its exported labels are
not numerical affinity measurements. Other display-based studies share the broad
method label. See `results/multiplex-assay-inventory.json` and
`results/unsupported-assay-inventory.json`. A future typed microarray/display
readout must preserve its measurement scope instead of converting it to IC50.
Primary source: https://pubmed.ncbi.nlm.nih.gov/32903714/ ; source assay metadata:
https://www.iedb.org/assay/17059219 . These rows are explicitly excluded from the
current frozen admission contract, not silently counted as trained examples.
