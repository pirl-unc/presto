# Quantitative HAP1 machinery panel admission

Original quantitative assets are recovered and their main quantity definitions
have been checked against the deposited search results. The shared Hitlist source
package is now implemented, built and checked against those original cells. The
endpoint training recipe, full-suite validation and release remain pending.
**No model was trained or evaluated in this family.**

Codex / GPT-6, October 24, 2026. Source state is Presto
`c7d538cfc64f2295e70c2bf693e333f08d1f34b6`, with recorded task dirt, and Hitlist
`fda1eff79d7a77f195e6a5f3a4c5abbee5cfd7c4` (2.0.18). Registration and the source
replay specification preceded their respective retrieval/arithmetic steps; see
[plan](reproduce/plan.md) and [configuration](reproduce/launch.json).

## Source and scope

Shapiro et al. 2025, PMID 40113210, PMC12090245, DOI
10.1016/j.mcpro.2025.100951, PRIDE PXD056426. The panel comprises wildtype and 11
individual machinery knockout conditions in human HAP1 cells, with three declared
biological replicates per condition. HAP1 is a CML-derived cell line, not a
professional APC. The observed quantities are MHC-I elution and global proteomics;
they do not directly measure intracellular free-peptide concentrations.

Original article XML and eight supplemental assets were retrieved through
EuropePMC. The PRIDE inventory contains 167 files: 164 raw acquisitions, two
processed-search archives and a community SDRF. The 130,237,651-byte canonical
`Search.zip` and SDRF were downloaded; the noncanonical search archive and raw
acquisitions were only inventoried. All downloaded assets have size/hash receipts.

| Supplement | Original grain | Size |
|---|---|---:|
| mmc5.xlsx | DDA, DIA and proteomics run/condition/replicate maps | 61 / 67 / 36 runs |
| mmc6.csv | Canonical peptide × biological replicate | 20,696 peptides × 36 samples |
| mmc7.csv | Noncanonical-search peptide × biological replicate | 27,184 peptides × 35 samples |
| mmc8.csv | Protein × sample, three quantity families | 6,437 proteins × 36 samples |

Source hashes, headers, literal missingness counts, sample rows and file inventory
are retained in [source-inventory.json](results/source-inventory.json). The source
root is `data/hap1_quantitative_source_admission/2026-10-24_0612_codex_hap1-quantitative-admission/`;
raw bodies and large deposited files remain there, outside the experiment bundle.
The article is CC BY 4.0. Its original filtering, background exclusions, binding
predictions and analysis-only imputation must remain distinguishable from measured
quantities in the eventual upstream representation.

## Quantity replay

[replay.py](analysis/replay.py) reads the original archives, independently of a
production curator. It streams all 607,928 DIA rows from 69 runs. Of these, 67 runs
map to the supplement, yielding 567,399 unique run/precursor keys with no duplicate
keys and 489,076 run/stripped-peptide aggregates. Two additional runs remain
unmapped (40,529 rows); they were counted and deferred, not assigned a condition.

| Tested interpretation | Compared numeric cells | Agreement |
|---|---:|---|
| Peptide: normalized precursor sums, mean over all declared technical runs | 296,483 | All agree; maximum absolute error 7.45e-9 |
| Peptide: same quantities, mean over observed technical runs only | 296,483 | 212,472 within declared tolerance |
| Peptide: raw precursor quantities | 296,483 | No matches within declared tolerance |
| Protein: MaxLFQ intensity | 166,028 | All exactly equal |
| Protein: raw intensity | 166,028 | 1,033 exactly equal |

The peptide supplement's `sampling_density` suffix therefore does not describe
its reconstructed numeric scale: these cells reproduce normalized precursor
intensities. Averaging over all declared technical runs incorporates absent run
contributions into the denominator. This describes the published aggregation; it
does not authorize turning wholly missing peptide observations into measured zeros.
The protein supplement's `intensity` fields reproduce MaxLFQ, not raw intensity.

The initial replay took 55.03 seconds and peaked at 0.635 GiB process RSS. Complete
per-cell comparisons and numerical summaries are retained in
[results/replay](results/replay). Retrieval took 117.73 seconds for Search.zip and
0.64 seconds for the SDRF. Execution was local CPU; no Modal/GPU or shared-environment
changes were made.

## Remaining admission questions

- All 36 peptide supplement `NA` cells with a reconstructed deposited quantity have
  quantity zero. Preserve both representations and their source, rather than
  relabeling the supplement as an observed zero.
- Of 65,704 protein supplement `NA` cells, 42,608 have zero deposited MaxLFQ and
  **23,096 have positive deposited MaxLFQ**. The latter need explicit source-filter
  lineage; do not silently fill the supplement's missing cells.
- The two unmapped DIA runs are absent from both inspected source sample workbooks.
  Their acquisition context remains unknown; run-name appearance is not proof.
- All 36 proteomics run names have unique punctuation-normalized deposit aliases.
  These are candidate filename links, not proof of specimen pairing with elution.
- Per-run and library Q-values are at most .01 for mapped rows, but 2,059 mapped rows
  have global Q-value above .01. Preserve the different confidence fields and
  source policy; do not silently impose another threshold.
- Noncanonical quantities, protein sampling density, coverage, complete sequence
  origins and exact material/control lineage still require canonical admission.

These observations are reproduced from the retained cell comparisons and recorded
in [discrepancies.json](results/replay/discrepancies.json). That file also retains the
full distribution of source MaxLFQ values for missing protein cells. The follow-up
summary was calculated after replay; it is not a separately preregistered model test.

## Reproduction and decision

Retrieval entry point: [reproduce/launch.sh](reproduce/launch.sh). Exact URLs,
commands, environment settings, responses and source hashes are in retrieval and
deposit receipts. The retrieval launcher, deposited-file launcher and replay
launcher are retained under [reproduce/source](reproduce/source). From a fresh
experiment output directory with those assets retrieved, run the registered Python
interpreter on `analysis/inventory.py`, then `analysis/replay.py`. Replay refuses to
overwrite its existing output directory; preserve original results when extending it.

This is a source audit, so there is no training/pretraining, synthetic data, loss,
assay-to-model-output mapping, train/validation/test split or predictive dump.
No binding or concentration metrics apply. Published source measurements are not
counted as model predictions. The prior APC baseline and its closed test remain
unchanged.

Continue through existing [Hitlist #726](https://github.com/pirl-unc/hitlist/issues/726):
complete the endpoint recipe and source release before the biological comparison.
This work does not implement a Presto-only source curator or complete that upstream
request. Independent APC availability
and concentration validation remain open.

## Shared source package implementation and verification

Draft [Hitlist PR #806](https://github.com/pirl-unc/hitlist/pull/806), commit
`2f4bbe0`, on branch `codex/hap1-quantitative-726-20261024` implements
`write_hap1_quantitative_package` and `verify_hap1_quantitative_package`, with ten
relations exposed through EvidenceStore. Base is e8faa7f (2.0.19); the change bumps
to 2.0.20, package schema 20/catalog 25. Original source layers, metadata claims,
reference entries and candidate occurrences remain queryable. Protein sampling
density joins to elution context; MaxLFQ and coverage join to proteomics context.

The first original-source build, [v1](results/admission-v1/result.json), failed
before quantity projection because the community SDRF repeats a column name.
The parser now preserves positional headers and values; its regression checks
both claims survive. The failed source snapshot and exception remain recorded.

The successful [v2 receipt](results/admission-v2/result.json) and
[manifest](results/admission-v2/manifest.json) pin the package to
`88cad10d7bc2a638305cb0d37ad96ac2826c3badaf7b3e0aad7928820df5c4b8`.
Its directory is `canonical-package-v2` under the source root. Construction plus
complete source replay took 379.92s and peaked at 1.163 GiB.

| Shared relation | Rows |
|---|---:|
| Supplemental quantity cells, including missing | 2,391,692 |
| Deposited protein quantity cells | 558,216 |
| Original DIA precursor rows | 607,928 |
| Original source and metadata records | 62,496 |
| Search reference entries, including decoys/standards | 40,964 |
| Exact occurrence or unmapped records | 60,447 |
| Source sample groups / runs / modality-condition contexts | 83 / 166 / 24 |

The 83 sample groups include additional WT/TAP1/TAP2 biological labels used in
DDA library construction, beyond the 36 quantitative elution and 36 proteomics
groups. These are not extra quantitative replicate measurements. Seventeen
workbook condition labels differ literally (ERp57/PDIA3 and GIIa/GANAB) while raw
filenames and replicate labels agree; both source claims remain preserved.

[Independent verification](results/admission-v2/independent-source-check.json)
compares every supplemental cell, deposited protein quantity and DIA row to the
original files without adapter parsing helpers. It checks source-cell conservation,
raw/numeric values, sample/run links and each of the 59,726 emitted exact sequence
matches; 721 source peptide rows remain unmapped. Runtime 96.90s, peak RSS
0.987 GiB. This checker does not independently prove search exhaustiveness,
specimen independence or predictive accuracy.

An actual query with unresolved contexts exposed shared
[Hitlist #805](https://github.com/pirl-unc/hitlist/issues/805): all-null optional
joins raised an Arrow type error, and mixed joins could match null identities.
Three regressions fail before correction. The fix retains typed null outputs and
keeps required relationships strict. [Actual package queries now pass](results/admission-v2/evidence-store-smoke.json).
The source package and quantitative replay are unchanged by this consumer fix;
the query receipt records the corrected consumer hash separately.

Format/lint and 55 focused package/catalog/query tests pass. Draft PR806's docs
and lint checks pass; full tests are tracked by run40088549411 at exact2f4bbe0.
`./test.sh` was
attempted but its capacity gate refused to launch with 0.22 GiB free against a
2.5 GiB minimum; no guard override was used. Full-suite/hosted validation and
merge/release remain pending. Each build attempt freezes invocation, environment,
git diff and changed source files before execution in `results/admission-v*/`.
Independent source/query checkers are under `analysis/`. No biological fit has
been performed on this package.
