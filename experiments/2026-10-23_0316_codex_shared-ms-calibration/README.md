# Shared MS calibration evidence and protocol integration

**Source admission and conservation pass; no predictive fit.** The shared Hitlist source now preserves PXD004886 nominal peptide loads, raw/normalized response, distinct reporter identities, original curves and acquisition context through EvidenceStore. All existing MAPP source files and recipe contents are conserved. Training eligibility remains a separate, unfinished recipe decision. This does not resolve the unsuccessful MAPP availability fits or earlier IC50 underfitting.

## Execution and reproduction

- Date: October23,2026; agent/model: Codex / GPT-6.
- Presto registration: `3f1ae1b6fb9de1f175fbfc99f14682793b4f4a88`, existing dirty task notes/untracked work recorded in [launch.json](reproduce/launch.json).
- Successful source implementation: clean Hitlist `04f7369d64e6d099cc5b8ce5163e6825389e1c1c`, version2.0.3 candidate stacked on canonical MAPP integration2.0.2. These are source commits, not released-version claims.
- [Original specification](reproduce/plan.md), [prospective correction after first failure](reproduce/replan-after-attempt-1.md), [exact command](reproduce/launch.sh), [configuration and frozen source hashes](reproduce/launch.json), [independent verifier registration](reproduce/verify.json).
- Local CPU only; no Modal or GPU. Python3.12.6, PyArrow26.0.0, macOS26.5 arm64. Overrides: `PYTHONPATH=/private/tmp/hitlist-ms-calibration`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`. No shared-environment installation or upgrades.
- Large raw source generation: `/private/tmp/hitlist-v2-calibration-203-canonical`; exact inventory and lineage in [source-manifest.json](results/source-manifest.json). Original assets: `/private/tmp/hitlist-pxd004886-assets`; assets are caller-supplied, not bundled in the Hitlist wheel. Their retrieval and publisher/archive equivalence receipts are in the [earlier source audit](../2026-10-23_0216_codex_ms-calibration-admission/README.md). Current frozen source receipt adds Data13/14/15 and article XML hashes.
- Input reference: `/private/tmp/hitlist-v2-reference-200-final`, manifest SHA256 `d99f67e625c6c33a2054a56dea4f98fb8248ca5f717cc98af3662f228655fa6a`. Prior MAPP generation: `/private/tmp/hitlist-v2-mapp-202-canonical`, manifest SHA256 `83a21da963d7d938c68a9570d7b868e578107302ba466ac55df09c839309b693`.
- A fresh schema8 generation is built from the same actual schema7 reference with both MAPP and calibration sources. The prior MAPP top-level manifest is kept separate; no claim of recursively appending to schema8. All157 prior inventory files match exactly and have independent inodes.

The launcher writes a new immutable destination and must not overwrite this completed generation. For a new run, register new destination/recipe paths and retain this invocation. The independent verifier uses direct Parquet reads plus a separately implemented original-XML/shared-formula parser; it imports neither Hitlist nor openpyxl.

## Dataset and curation contract

PXD004886 is a selected synthetic peptide panel spiked into a human HEK293 background digest after digestion. It provides an **instrument-response observation**, not APC processing, peptide yield, free intracellular concentration or MHC capture. Original standards were chosen for LC/MS performance and grouped using measured response; group assignment must not become a sequence-response predictor. Publication/supplement license is CC-BY4.0 subject to third-party credit exceptions.

Original Supplementary Data1/2/4/8/13/14/15 and article XML are pinned. Every populated source cell retains original cached value, expanded formula, coordinate, canonical XML serialization and missingness. Original asset bytes retain the literal source XML. Physical absence, explicit empty string, empty formula cache and zero remain distinct; no zero is present in the source S4 response matrices. No omitted slot becomes a negative or imputed measurement.

| Shared relation | Rows | Meaning |
|---|---:|---|
| Source sheets / cells | 7 / 13,561 | Original evidence and layout |
| Nominal standards / loads | 30 / 25 | Sequence-level panel and group×S1–S5 design, fmol on column |
| Reporter forms | 32 | Distinct original labels across assets;31 within each |
| S4 source slots | 99 | Original column positions under11 site headers; no invented run/day identity |
| S4 measurement slots | 6,138 |31×99×2 raw/normalized representations |
| Source summary measurements | 186 | Avg, SD and CV, including empty states |
| Curve rows / positions | 682 / 3,410 | Actual source RSQ ranges and plotting amounts |
| Protocol parameters | 520 |35 configurations/footnotes,165 gradient slots,320 window parameters |
| Source assertions / normalization diagnostics | 34 / 99 | Scoped facts, explicit unknowns/conflicts and derived quality checks |

There are24,823 calibration rows across distinct grains; these are not24,823 independent experimental examples. The canonical catalog exposes84 materialized evidence types. All31 MAPP/calibration relations were queried without truncation:1,853,831 rows, independently matched to direct Parquet fingerprints.

Specific retained boundaries:

- Nominal amount is **fmol on column**, not stock molarity or certified molarity of each molecular form. CNNVLYIR CAM/PCm remain separate; form fractions, purity, lot and amount uncertainty are unknown.
- Four `_HUMAN` reporter labels omit the isotope suffix present in the nominal panel. Their exact labels/annotations are retained; a sequence match alone does not establish heavy/light equivalence. The prospective raw-S4 recipe will exclude these unresolved form joins.
- First materialization (`f7ebc08`) failed on an invalid assumption that curve and S4 reporter-label sets were identical. Data4 A21/A56 has `SMD2.NNTQVLINC[CAM]R[+10]`; Data8 A25 and the corresponding22 blocks add `.sum4`. The corrected adapter retains both reporter IDs and an unresolved equivalence assertion. [All first-attempt code/config/logs/results](attempts/attempt-1) are retained. No published source generation came from that failed attempt.
- Data4 is S4 only. Raw contains2,372 positive values,568 explicit empty strings and129 absent cells; normalized contains2,398 positives,572 explicit empty strings and99 absent cells. The26 normalized-only positives and39 nonuniform ratio columns remain unrepaired source discrepancies. Raw and normalized are not independent examples.
- Curves preserve their source RSQ ranges and plotting amounts. A sample-to-nominal-load join is unresolved; curve omissions may reflect selection/quantification and saturation handling. There are2,416 positive curve responses; no full dilution-curve training is admitted.
- Data13 site LC/MS/column settings, Data14 initial-QC versus main-study gradients, and Data15 the64-window scheme share one source-linked protocol-parameter relation. Literal “As above/below” directives remain literal. No phase or per-run context is guessed.
- Article Methods Sec12 supplies a2µL **reference-protocol injection volume**. This is not an individual-run receipt. Human background, porcine trypsin and synthetic standards retain separate roles. APC species/machinery and MHC fields are not applicable to this assay, not missing biological labels.
- Data8's workbook title saysData9; both source and publication labels survive.

## Verification and comparisons

[Execution receipt](results/execution.json), [independent audit](results/independent-verification.json), [canonical catalog](results/catalog.json), [baseline audit](results/audit.json), [full query fingerprints](results/query-all-added-sources.json).

| Stage | Wall time | Result |
|---|---:|---|
| First source materialization |16s by UTC receipt|Fails closed on cross-asset aggregation identity; retained |
| Corrected source generation |65.72s|Pass |
| Open/catalog/baseline audit |22.04s|All32 baseline audits match |
| Full MAPP/calibration queries |8.90s|All31 relations conserved |
| Three original MAPP recipes |27.77s|Two API, one CLI; common and CLI verification pass |
| Independent XML/Parquet/conservation audit |13.66s|Pass |

Producer peak process RSS:3,457,449,984bytes (3.22GiB). Independent audit:691,044,352bytes (0.64GiB). Producer receipt spans125s including the final CLI verification. These are data-pipeline timings, not neural training throughput.

The independent audit checks all13,561 source cached values, formulas, XML elements/types, missingness and coordinates; allS4/curve responses and nominal joins;162 reproducible source means/SD/CVs; protocol values/scopes; aggregation multiplicity; normalized-only and ratio diagnostics; current generation inventory;157 prior files/inode isolation; every query row; and every original MAPP recipe content file. The three recipe exports retain all original targets, splits, coverage, parents, contributors and run observations byte-for-byte. Only their source-manifest envelopes change. [Complete recipe exports](results/recipe-exports.tar.gz) contain28 independently byte-verified files; [archive receipt](results/recipe-archive-verification.json). Uncompressed recipe paths are recorded in [recipes.json](results/recipes.json).

Format/lint,56 focused tests and strict docs pass. Required normal `test.sh` refuses before starting pytest:0.65GiB available versus2.5GiB required for one worker; no guard bypass or full local-suite pass. [Software checks](results/software-checks.json) and logs preserve the distinction. Presto registration head3f1ae1b has both full lint/test and both docs CI jobs passing; new Hitlist-head CI is separate.

## Training and decision

No model, pretraining, optimization, synthetic labels, losses/weights or assay-head mapping was used. No new validation/test split, predictive metric, checkpoint or per-example prediction dump is applicable: this experiment qualifies original evidence and preserves previously exposed splits. IC50 regression and500nM binder metrics do not apply to MS intensity. Successful source preservation does not improve the earlier predictive results.

Preferred source contract: retain both aggregation identities and their provenance. Reject the original cross-table equality assumption. [Hitlist#762](https://github.com/pirl-unc/hitlist/issues/762) now explicitly owns the aggregation-equivalence limit and the recovered protocol metadata as well as calibration admission. Related precursor/run links remain in#757 and controlled processing panels in#759.

[Detailed next-stage specification](../agents/codex/plans/raw-s4-nominal-response.md). Next: define and independently admit a bounded **raw-S4 nominal-load response** recipe through Hitlist, with explicit form exclusions, source-cell dispositions, peptide/site grouping and restricted interpretation. Only then fit a local response diagnostic. Full curve calibration still requires sample/aggregation/selection reconciliation. A30-standard selected panel cannot establish broad sequence detectability, SWATH-to-MAPP transfer, or the intracellular concentration model; the availability goal remains open.

Publication: shared source is [draft Hitlist#767](https://github.com/pirl-unc/hitlist/pull/767), stacked on#760. Published headad60a8f changes only documentation/task closure after executed04f7369; frozen implementation hashes remain the execution contract. Hosted CI/review and release are separate gates.
