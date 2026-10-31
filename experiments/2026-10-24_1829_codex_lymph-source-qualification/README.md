# Independent lymph/DC quantitative source qualification

Codex / GPT-6, October 24, 2026. **Source qualification completed with usable quantities and explicit unresolved context; no model has been trained or evaluated.** The registered [plan](reproduce/plan.md) freezes source semantics and resource limits. This stage does not improve the biological prediction scores.

## Recovered primary material

Nanaware2024, PMID38848214 / PMC11233987, DOI10.1016/j.celrep.2024.114311. All seven original quantitative workbooks, primary XML and license metadata were recovered from the official PMC cloud inventory, version PMC11233987.319. Nine processed assets total 80,601,182 bytes; original bytes remain in `data/lymph_source_qualification/2026-10-24_1829_v1`. License metadata states CC BY-NC-ND. Exact retrieval URLs, byte counts, hashes and failures are under `reproduce/` and `results/`.

The primary Key Resources table links PXD051024 native human lymph peptidome, PXD044801 human lymph proteomics, PXD044885 mouse control/DSS lymph, PXD051618/MSV000094586 mouse lymph proteomics, and PXD052267/MSV000094761 dendritic-cell I-Ab immunopeptidomics. Complete public repository inventories are independently reconciled in the subsequent1903canonical admission. Source-to-specimen mapping remains unresolved.

## Independently checked source contents

All nine asset SHA256 and source-metadata MD5 hashes, all seven workbook ZIP CRCs, 72 sheet identities and 14,567 original header/early-row cells pass independent OOXML-versus-openpyxl checks. Verification takes 1.129seconds, peak0.157GiB locally. These are source checks, not predictive metrics.

| Original block | Peptide-bearing rows | Distinct literal peptide forms | Positive area cells | Missing area cells |
|---|---:|---:|---:|---:|
| S6(a), native human lymph |1783|1138|1783|0|
| S4(e), DC DIA database search |1581|1315|2536|6950|
| S4(e), DC DIA library search |1020|632|4632|1488|

These are raw source-block counts, not admitted training examples or independent specimens. No sequence canonicalization, confidence filtering, normalization, imputation, zero-to-negative conversion or source-quantity aggregation was performed. The two DIA search blocks analyze overlapping acquisition material and are not independent replicates.

The human native area block and its side-by-side protein summary have independent row orders; a row join would be invalid. The protein summary is derived from the same native peptides and is not independent precursor abundance. S4(e) contains six area columns, three cervical and three mesenteric, plus separate intensity fields; broader descriptive replicate/control labels still need reconciliation. Do not conflate area, intensity, total TIC or spectral counts. Native lymph is extracellular; these measurements do not directly label the free intracellular APC peptide pool. Human and mouse, analytical digest and native products, and MHC-selected and unselected populations remain separate. Anatomical similarity does not establish paired specimens.

## Preserved verification failures

An initial checksum check incorrectly interpreted multipart S3 ETags as MD5; original source-metadata MD5 verification corrected that assumption. Independent cell decoding then encountered shared formulas, elapsed-time styles and time-of-day styles. Every executed failed source/log remains under `results/attempt-*`, with frozen correction rationale in `reproduce/`. The successful checker translates relative shared formulas independently and preserves the original numeric values of all20 inspected formatted-time cells. No biological quantities or eligibility rules changed. Exact successful command, source hashes and exit status are recorded in `reproduce/verification-recovery.json`.

## Pending admission and modeling

Complete repository inventory reconciliation, independent quantity-count replay, sample/replicate/normalization qualification and the shared Hitlist source/recipe remain pending. No private Presto training curator has been introduced. No train/validation/test split, assay model, pretraining, loss, synthetic labels, predictive score or prediction dump applies to this source-only stage. A fresh prospective modeling contract is required before fitting; previously inspected MAPP/native/macrophage/HAP1 tests remain closed to tuning. Local CPU only; no Modal, RAW acquisition or shared-environment modification.

## Closure and canonical successor

The follow-on [1903source admission](../2026-10-24_1903_codex_lymph-canonical-admission/README.md) independently reproduces every emitted quantity and source cell, and all five repository inventories. It also recovers original human/DC mzIdentML references and a conflicting community SDRF instrument assertion:19total processed source assets. Shared Hitlist source implementation is under verification; [Hitlist817](https://github.com/pirl-unc/hitlist/issues/817) records source ownership and metadata limits. The source qualification is complete, while training-recipe admission and biological fitting remain pending. Earlier serialization failures stay preserved.

The published workbook inventory omits individual clinical rows from S5(g); these do not enter the modeling source. The complete local inventory and original workbook remain available by hash. To reproduce the full inventory after retrieving the pinned original assets, run `python analysis/inventory_workbooks.py` from this experiment (with its existing absolute RAW path); then run the independent workbook verifier. The original full inventory is deliberately not duplicated in Git.
