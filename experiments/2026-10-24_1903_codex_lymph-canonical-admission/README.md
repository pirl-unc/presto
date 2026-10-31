# Canonical lymph/DC source admission: original quantities and references reconcile

Codex / GPT-6, October24,2026. Source admission succeeds through the shared Hitlist API. **No biological model was trained or evaluated.** The [prospective plan](reproduce/plan.md) predates materialization; the [source owner](https://github.com/pirl-unc/hitlist/issues/817) records scientific metadata gaps. [Hitlist PR818](https://github.com/pirl-unc/hitlist/pull/818) is published; CI/merge/release remain pending separately.

## Source and curation contract

Nanaware2024 PMID38848214 / PMC11233987.319, DOI10.1016/j.celrep.2024.114311. Nineteen original public processed assets include seven quantitative workbooks, primary XML/rights metadata, community human SDRF and nine deposited mzIdentML files. All assets are pinned by URL, exact bytes and SHA256. Raw originals are in `data/lymph_source_qualification/2026-10-24_1829_v1`; the canonical immutable package is `data/lymph_source_qualification/2026-10-24_1903_canonical`.

The shared source projects only human S6(a) A:Q and the two distinct DC S4(e) database/library blocks. Other workbook blocks remain original source assets. Keep every peptide-bearing row, original cells, modified/ambiguous forms, charges, confidence fields and reference alternatives. Areas and intensities remain separate. No filtering, normalization, aggregation, missing-value imputation, synthetic biological labels or negative generation occurs. Formula/invalid/nonfinite/zero/missing states remain distinct. The source is not an endpoint-specific training recipe.

| Relation | Rows |
|---|---:|
| Original assets |19|
| Original source-block rows / peptide-form observations |4384 each|
| Typed modality/anatomical contexts |3|
| Exported sample labels |7|
| Measurement cells, including missing |32995|
| Deposited protein-reference entries |3430|
| Deposited peptide-evidence alternatives |23455|
| Source-scoped contextual assertions |14|

Human native lymph has1783positive area cells. DC database and library searches have2536/4632positive area cells respectively and the same positive intensity-cell counts;6950/1488cells of each kind are missing. The algorithms reuse acquisitions and share sample IDs; these counts are not independent specimens or unique peptides.

Human material is extracellular lymph from IBD subjects; Sample1 donor/pooling identity remains unresolved. The adjacent protein summary derives from these peptides and is not independent precursor abundance. Primary Q-Exactive HF instrument context conflicts with community SDRF Fusion Lumos. DCs were expanded by B16-FLT3-L treatment, then isolated from cervical/mesenteric nodes for I-Ab capture. They are not untreated baseline APCs. Exported3+3channels do not resolve the description's3+6replicate/control counts. No per-sample machinery loss is supplied; unknown is not wildtype. Mouse S4(n) no-enzyme searches reuse files described under analytical digestion and remain ineligible as native-processing labels. MHC-selected DC peptides do not directly label free intracellular concentration.

## Implementation, verification and resources

Actual execution imported isolated Hitlist2.0.27 development source over clean main68dea0be320906e311763db553cee54dccb6fa07, with exact changed modules and contract snapshots under `reproduce/hitlist-source/`. Presto basee8d5daf362151b84c81da92d95b962b733391338, with registered task/source dirt. Shared installed distribution remains1.82.2; no environment replacement occurred.

The writer snapshots original bytes, verifies them, compiles nine typed relations, replays the original sources and atomically creates a new package. EvidenceStore exposes all nine relations and declared same-grain/context links. Independent OOXML decoding then compares121346original cells and every32995quantity cell to the original workbooks, including sample/observation links and missingness. A separate lxml decoder verifies every3430reference entry and23455identification alternative. These checks do not establish measured parent origin or perfect identification confidence.

Complete PRIDE inventories reconcile4/13/50files for PXD051024/PXD044801/PXD044885. Independent MassIVE inventory parsers reconcile132/44files for MSV000094586/MSV000094761. Only processed files were downloaded; no RAW acquisition or reprocessing occurred.

| Stage | Seconds | Peak process GiB |
|---|---:|---:|
| Shared source build, replay, API queries |15.217|0.461|
| Independent cells/references/inventories |2.267|0.557|

Local CPU, one numerical thread, Python3.12.6; all600second/3GiBstage caps pass. No Modal, GPU or model parameters. Full package manifestSHA256: `6e82a782b0511ba56425855f0764c4c40127e715fdb10b10e6c12118ab539c4c`.

Focused source tests pass8cases. Initial unused-import lint and an incorrect shared-Sink call signature were fixed before the real-data launch. The required full test runner initially could not access its system memory probe inside the sandbox; the original refusal is retained, then the same one-worker checks were rerun with normal probe access. Initial full regression had3459passes and seven assertions expecting catalog28instead of29. Updating only those global-version expectations yields43focused passes and a complete required rerun of3466passes in294.19seconds, maximum processRSS870432768bytes. Format/lint pass. Historical package versions and real-source bytes did not change. PR818head8657c9acd637e63d7ec175e1353aebca94a76c99contains the verified source; CI/merge/release remains pending.

## Reproduction and next biological comparison

[Launch](reproduce/launch.sh), [exact source/env receipt](reproduce/launch.json), [verification](reproduce/verify.sh), both source snapshots, [admission](results/admission.json), [full package manifest](results/source-manifest.json) and [independent reconciliation](results/independent-verification.json) preserve execution. Launch refuses an existing output; replay requires a fresh destination generation with recorded frozen inputs. Original quantities remain in the canonical package rather than duplicated in experiment source code.

No pretraining/training contract, loss terms, assay-head mapping, held-out scores, binding threshold metrics or prediction dumps applies because this is source-only admission. Do not claim new biological accuracy. Next: resolve an explicit observable-composition recipe, source grain/form aggregation and full-parent dependence; freeze appropriate fresh training/evaluation splits and compare shared-observer exact and sampled methods with simple controls. Cross-study processing/MS chemistry transfer and biological identifiability must be tested, not assumed. The full APC availability/concentration objective remains active and unachieved.

## Source structure and sequence-annotation checks

A frozen structural audit compares API rows directly with materialized tables, preserving every observation. Exact full accession labels initially match none because of spelling conventions. A separate explicitly declared token parser handles bare accessions, accession|name and sp|accession|name without changing original labels, stripping isoforms or resolving ambiguous origins. It finds exact peptide occurrences for1612human rows across186candidate full-parent sequences,875DC database rows across196parents, and941DC library rows across191parents. All1612supported human rows have an occurrence matching reported coordinates; alternatives remain. No duplicate keys under the declared block/raw-peptide/parent/charge/start/end definition were found. Structural-only checks take5.282s/0.436GiB and5.120s/0.438GiB. These are source support counts, not training/test examples.

Every2601DC row reports a Length one residue shorter than the literal peptide. Most mapped Start positions are shifted by+1while End agrees. A separately frozen mass hypothesis check supports the **full literal strings**: all1581database and1020library rows match the original reported mass within0.1Da, including explicit rounded modification deltas; none matches the one-residue-shortened alternative. Pyteomics and Biopython unmodified monoisotopic masses agree within0.002Da. Deposited identification alternatives contain729/1008full strings. This takes0.135s/0.232GiB. All per-row masses, residuals and deposit-presence flags are retained; source strings/lengths/coordinates remain unchanged. Mass agreement does not establish an exact quantity/acquisition join or identification correctness. A subsequent recipe should preserve the annotation discrepancy and derive candidate positions from actual source sequences, rather than removing a residue or excluding every DC row by length.

The human parser preserves102unsupported/noncanonical strings without inventing sequence replacements;69recognized human rows lack an exact accession-token reference match. DC missing-reference counts are706/79. Missing original references are not supplied from a current external database. Source scope and full-parent alternative grouping must survive future training admission. No predictive metric or new model fit occurs in these diagnostics.

The exact tracked Hitlist patch is retained as `reproduce/hitlist-tracked.diff.gz`; decompression preserves the original context whitespace byte-for-byte. The uncompressed working copy is not separately versioned.
