# Native intracellular peptidomics: source replay passes; semantic admission fails

The public workbooks are available and the newly merged shared Hitlist package
reproduces their values. Training admission remains deferred: the audit finds
broken cross-paper matching, sequence/annotation errors and incorrect acquisition
provenance. These findings are filed as [Hitlist791](https://github.com/pirl-unc/hitlist/issues/791).
This completes the available-assets audit, not full admission or concentration
validation. No new biological model trained; the MAPP transfer stop rule remains.

## Source and execution contract

Agent/model: Codex / GPT-6. Registered October24,2026 before retrieval/schema
arithmetic at Presto132404d1d0a2b714dcff9804e7e052b909f28f6d, with recorded dirty
task state. Thirteen public PLOS assets downloaded successfully with timestamps,
status, hashes and sizes. The original epoxomicin Europe PMC XML request returned
HTTP500; a publisher-page browser request returned HTTP403. Original2012 source
lineage remains unresolved. No same-URL retrieval retries, author outreach or
access bypass occurred.

Primary publications:
- [Dasgupta2014](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0103604):
  primary S1 (.s001), selected heatmap S2 (.s002).
- [Gelman2013](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0053263):
  primary S1 (.s004), heatmap S2 (.s005), derived dose/time S3/S4 (.s006/.s007).
  Initial .s001/.s002 leads are figures; the initial registration is preserved,
  and retrieval.json records corrected identities.
- [Fricker2012](https://doi.org/10.1021/pr2012076), original epoxomicin study:
  bibliographic/reuse evidence only; its source table was not retrieved here.

During the audit, [Hitlist790](https://github.com/pirl-unc/hitlist/pull/790) merged.
The public package writer and EvidenceStore are executed from pinned commit
f4d3a6d0f1933162f9ac6eac42ac1b134c8a27dc (code2.0.15, package schema15/catalog20),
in an isolated checkout. No shared environment/user checkout changes or parallel
Presto training curator. PyPI publication was not verified by this audit.

## Verification and source counts

The independent reader decodes zip/XML relationships, shared strings, original
cell types/values, styles and sheet layouts, then compares every explicit cell
with openpyxl. Compressed JSONL preserves original cell XML, including formatted
blanks and spreadsheet errors. Layouts retain row/column/merge/hidden attributes
and original style/comment references. There are no formulas; `=Avg` annotations
are literal strings. Canonical row arrays are compared against those independent
cells; numeric/bound/missing/overlap/error parsing is checked separately.

| Check | Result |
|---|---:|
| Workbooks independently decoded | 6 |
| Explicit OOXML cells / parser disagreements | 271,778 / 0 |
| Primary source rows | 10,318 |
| Canonical source rows including deferred heatmaps | 10,564 |
| Canonical row values replayed, including blanks | 328,880 |
| Ratio slots replayed, queried and parse-verified | 61,908 |

| Source | Primary rows | Deferred heatmap rows |
|---|---:|---:|
| Dasgupta2014 | 6,073 | 73 |
| Gelman2013 | 4,245 | 173 |

These are source-grain counts, not independent specimens, unique peptides or
admitted training examples. A primary row expands to three treated/control slots,
two control/control slots and one derived average, including empty slots. Derived
heatmaps/dose/time tables are not additional independent targets. Worksheet
dimensions include formatting and are not biological counts.

| Ratio-cell interpretation | Slots |
|---|---:|
| Reported numeric exact | 34,763 |
| Explicit bound | 15,753 |
| Empty source value | 4,113 |
| `nd` | 7,217 |
| `po` | 60 |
| Spreadsheet `#VALUE!` | 2 |

All raw values/operators remain unchanged. Three reported numeric values are
nonpositive; a log-ratio recipe must explicitly handle their source status, not
silently add a pseudocount. Numeric exactness is separate from identity confidence:
16,807 exact slots lack a sequence and213 have outer-parenthesized tentative
identities. Including an uncertain-modification annotation gives219 exact slots
with parentheses.

## Findings affecting biological inference

1. **Reuse is missed.** The canonical matching key includes the paper ID, making
   cross-paper matches impossible; all61,908 lineage records say `source_unique`.
   Independent fingerprints find830 possible repeated groups /1,660rows: identical
   raw cell line, sequence, charge, observed mass and first five ratios, requiring
   at least two numeric ratios. Of these,552 also have equal case/whitespace-normalized
   experiment labels;278 differ and need adjudication. These are possible reuse
   links, not proof of specimen identity or a ready deduplication rule. Both papers
   explicitly describe republication/reanalysis. All candidates/locators are retained.
2. **Identity can be corrupted.** DasguptaK587 `(Ac-AGKQAVSASGK)` becomes
   `AAGKQAVSASGK`: the acetyl prefix is checked before removing parentheses, then
   arbitrary uppercase letters are extracted. The legends mark outer parentheses
   as tentative MS/MS assignments. Annotation letters must not become residues.
3. **Modifications are lost.** 540 slots contain `Mox` but its oxidation annotation
   is absent from the normalized modifications. `pyroE-` and uncertain dehydration
   syntax also need explicit supported/unresolved status. Keep biological PTMs,
   analytical labels, charge and identification confidence separate.
4. **Heatmap sequence projection is wrong.** All173 Gelman heatmap rows project
   columnE (adjacent residue), while sequences occupy columnF. Raw JSON is correct
   and heatmaps are deferred, but the exposed field is unsafe for lineage joins.
5. **Instrument scope is wrong.** The package assigns SynaptG1 universally.
   Gelman Methods nameG1; Dasgupta Methods nameG2. Reused data require original
   acquisition provenance, not automatic inheritance from the republishing paper.
6. **Timing needs a timeline.** 5,041 primary rows contain a wash annotation in the
   average-label column. The2014 Methods retain inhibitor during washing, supporting
   35minutes total exposure (20minutes treatment plus15minutes handling). This is
   not drug-free washout or a reason to shorten exposure. The2013 Methods retain
   inhibitor in some experiments; unresolved row-level assignments remain explicit.

Source arithmetic agrees, but the package's own replay also reproduces these
semantic errors. No source values were repaired locally or selected by model fit.
The original issue body and refinement after full Methods review are retained.

## Homogeneous metadata and future endpoint

| Dimension | Shared representation / limitation |
|---|---|
| Material | Native whole-cell peptide extract; neither MHC elution nor proteasome capture. No free-cytosol or professional-APC claim. |
| Species/cells | Human HEK293T and SH-SY5Y source labels, with original spelling and material scope. |
| Perturbation | Compound, vehicle, dose, treatment/harvest intervals and compound presence. Exposure is not KO, complete loss or known residual activity. |
| Measurement | TMAB peak ratio to within-experiment mean control; distinguish treated, control and derived average. No absolute abundance. |
| Replication | Rep1/2/3 and shared denominator; isotope assignments vary across experiments. A generic labeling figure is not a universal channel map. |
| Acquisition | Study protocol assertions and original acquisition lineage separately, including G1/G2 and DDA. |
| Digestion/MS response | Native extraction without an imposed digest; analytical labeling and detection belong to the observation model. Ratios do not calibrate general detectability. |
| Identity | Raw sequence, tentative/confident/unresolved status, PTMs, TMAB tags, charge, direct or transferred identification evidence. Unknown peaks are not sequence labels. |
| Parent | Original precursor/gene/flanking-residue claims. Modern full-length mapping and isoform ambiguity need qualification; historical IPI version remains incomplete. |
| Lineage | Reused/reanalyzed measurements, charge states, shared controls and derived tables. Different papers are not independent cohorts. |
| Other assays | Lysate/purified enzyme activities have separate scope; they do not establish in-cell machinery activity. |

These belong in Hitlist's common source/context/measurement/lineage relations and
recipe APIs. Issue791 records concrete corrections. Unknown specimen/run/channel
identities must not be invented. The original2012 supplement and exact reuse
adjudication remain unresolved. A future ratio fit tests relative peptide response;
precursor influx, clearance, absolute concentration and APC transfer remain open.

## Reproduction, resources and next action

Registration, source contracts and reconciliation are frozen in `reproduce/launch.json`,
`retrieval.json`, `plan.md` and `upstream-audit-plan.md`. Execution receipts contain
commands/environment, source hashes, timing and terminal status; source snapshots
are in `reproduce/source/`. Run `launch.sh`, `run_audit.py audit`,
`run_audit.py canonical_audit`, then `code/qualify.py` with the recorded Python and
one-thread environment. Scripts refuse existing outputs: rerun in fresh family/
asset directories with explicit path relocation and the pinned Hitlist checkout.

Original inputs/package live under
`data/native_peptidome_source_audit/2026-10-24_0117_codex_native-peptidome-admission/`.
Receipts give exact URLs/paths/hashes. Canonical manifest SHA256:
`b15de0b80c9fff1aad5b60f66e4e5b083c7f585ea01e723209a482182e26c32b`.
All source cells/layouts, candidate groups and summaries are retained in `results/`.

Local CPU only; no Modal/GPU/environment changes. Workbook audit14.64s/332.33MiB;
canonical build/query/replay14.60s/997.98MiB; quantity/projection verification1.02s.
All respect registered caps. Retrieval failure is preserved; all three executed
audit phases pass without retry. Experiment Python is explicitly checked with
`ruff --isolated --select F,E9` because normal repository lint excludes it.

There is no model/pretraining/optimizer, synthetic biological data, assay head,
loss or predictive split. Validation/test predictions, correlation/RMSE/binder
metrics do not apply before source independence and identity qualification.
No winner/checkpoint is promoted. Correct the shared source semantics, qualify a
recipe, then run the bounded native-response comparison. Full APC concentration
goal remains active and unachieved.
