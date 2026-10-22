# PXD057588 quantitative transform reconciliation

October 22, 2026; Codex / GPT-6. **All 200,724 published primary fold changes
reconcile to the prespecified assay-specific calculation** within 1e-8 log2 units.
The maximum absolute error is 1.63e-14. An independent scalar calculation from the
original workbooks verifies every source coordinate and comparison value.
Applying the opposite normalization gives 200,724 discrepancies.

## Frozen source and measurement contract

Use the six publisher Tables 7–12 from the [preceding source audit](../2026-10-22_1410_codex_availability-source-audit/README.md).
Its 17-asset manifest remains unchanged: SHA-256
`14419b00aeff8caebdc67917522e329b63b07d1ea85bbb59e15ef0c07bc3a27c`.
Originals are immutable datacache inputs under `data/availability_sources/`;
URLs, sizes and hashes remain in that audit's `reproduce/input_assets.json`.
The [prospective plan](../agents/codex/plans/pxd057588-quantity-reconciliation.md)
and [executable contract](reproduce/contract.json) specify all 32 contrasts before
calculation. Source-method citation: [primary paper, MS data processing](https://pmc.ncbi.nlm.nih.gov/articles/PMC12616400/).

The 10,662 MHC-I peptide rows and 25,110 global-proteomics digest-peptide rows
retain complete-case selection, native parent labels and source channel maps.
No new confidence filtering, zero filling, censor conversion, parent aggregation,
sequence-reference mapping or exact specimen pairing is introduced. The source's
reported q≤0.05 search policy remains an eligibility consideration. Independent
significance panels on the same spreadsheet rows are excluded from this primary
quantity comparison; they are not row-aligned measurements.

For elution Tables 7–10 and 12, calculate log2(treated / arithmetic mean of the
same peptide's untreated controls). For global-proteome Table 11, first divide
intensities by each channel's total over all retained rows; the published common
mean-total multiplier cancels in the ratio. Table 12 keeps separate GL261 and CT2A
controls. The opposite-policy diagnostic applies total normalization to elution
and omits it for global proteomics. It is deliberately incorrect, not an alternate
training target. Selection/normalization must be frozen before making subsets.

## Results

| Table | Published policy | FC cells | Mismatches at 1e-8 | Opposite-policy RMSE, log2 | Reported FC SD, log2 |
|---|---|---:|---:|---:|---:|
| 7 | unscaled | 8,472 | 0 | 0.606716 | 1.313518 |
| 8 | unscaled | 10,662 | 0 | 0.318651 | 0.460952 |
| 9 | unscaled | 7,854 | 0 | 0.162149 | 0.416465 |
| 10 | unscaled | 20,904 | 0 | 0.383605 | 0.436803 |
| 11 | total_signal | 150,660 | 0 | 0.082619 | 0.337851 |
| 12 | unscaled | 2,172 | 0 | 0.678239 | 0.951572 |

The labels have measurable variation. This does not show that a model can learn
it or explain the earlier IC50 underfitting: these are different assays and cohorts.
Wrong normalization can introduce substantial systematic target shifts, including
0.678 log2 RMSE for Table 12. Preserve the original scale and transformation as
quantitative evidence in Hitlist, not undocumented adapter behavior.

Table 9 contains 1,309 observed ligands across 1,096 literal parent labels;
928 labels have only one ligand. Tables 9 and 11 share 663 labels (830 ligand
rows), but only **110** have both multiple ligands and multiple digest peptides,
covering **258** ligand rows. Parent strings are not verified protein mappings.
A conditional distribution restricted to observed singleton support supplies no
within-parent shape information. None of these counts defines the unobserved
pool, a peptide response factor or a precursor-normalized concentration.

## Reproduction and verification

Launched from dirty Presto commit `51b895eb3925de87df7710c7bd6c78d131c25697`;
[launch.json](reproduce/launch.json) records exact dirty status and overrides.
The experiment-local frozen script and contract contain all executed calculation
logic; no uncommitted model code is imported. Python 3.12.6, NumPy 2.5.4,
openpyxl 3.1.5, local macOS arm64 CPU. Calculation: **6.432 s**;
independent scalar verification: **5.067 s**. No requested or observed GPU/Modal.

```bash
AUDIT_PYTHON=/Users/iskander/code/shared-virtual-env/bin/python bash experiments/2026-10-22_1601_codex_pxd057588-quantity-reconciliation/reproduce/launch.sh
/Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-22_1601_codex_pxd057588-quantity-reconciliation/reproduce/source/verify.py
```

The verifier shares only immutable input retrieval with the calculator. It uses
`math.fsum`/`math.log2` and original workbook cells independently of the NumPy
calculation, verifies 200,724 dump rows and aggregate errors, all six parent-support
tables and overlap counts, the contract/source hashes, and five output hashes.
Maximum scalar/vector difference is 1.63e-14 log2; [verification passed](verification/independent_verification.json).
Explicit Ruff E4/E7/E9/F passes. Source workbooks were never modified.
The first verifier and its 5.041 s receipt are retained; the final verifier writes
outside `results/` so rerunning the calculator cannot inventory a stale verification
receipt. Only that output path changed, followed by a full successful replay.

Artifacts: [summary](results/summary.json), [per-channel diagnostics](results/channel_reconciliation.json),
[parent support](results/parent_support.json), [all cell comparisons](results/cell_comparisons.csv.gz),
[runtime](results/runtime.json), [hashes](results/output_hashes.json), frozen code in
`reproduce/source/`, and stdout in `launch_logs/`. The dump is a calculation replay,
not model predictions. No model, pretraining, training parameters, synthetic labels,
losses, head mapping or winner applies. There is no train/validation/test split:
this audits all published primary values and makes no predictive generalization
claim. IC50 regression and 500 nM classification metrics do not apply.

## Decision and outstanding admission

The arithmetic portion of these published targets is qualified. Keep presentation
and global-digest quantities as separate endpoints. Carry derivation input IDs,
control membership, normalization universe and operation order through the shared
[Hitlist curation contract](../../docs/study_curation_contract.md). Source context,
protein references, exact versus surrogate modality joins, uncertainty and split
independence still require normalized export and endpoint-specific admission.
The authoritative Hitlist context implementation is in a separate worktree; this
experiment does not create a competing curation schema. Biological model fitting,
relative-availability identification and concentration calibration remain open.
