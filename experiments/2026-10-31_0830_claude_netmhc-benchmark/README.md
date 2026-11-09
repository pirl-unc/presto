# External NetMHC benchmark: Presto vs NetMHCpan-4.1 / NetMHCIIpan-4.3 / MHCflurry 2.3.16

Date 2026-10-31. Agent: claude (Claude Sonnet 5.5, Claude Code). Issue #140 (external-tool part), PR #149.
Pre-registration (committed before any tool ran): `experiments/agents/claude/plans/2026-10-31_external-netmhc-benchmark.md`.
CPU only, local, no Modal. Git base: `f00d4633` (pre-registration commit) on `codex/netmhc-benchmark`; results commit follows it.

## Result in one paragraph

On identical peptide/MHC rows, within study and assay detector, both Presto families are **far behind** the
external BA predictors on held-out test studies. Mean-of-study test Spearman is about 0.30-0.50 for Presto
against 0.58-0.70 for NetMHCpan-4.1/MHCflurry (class I) and 0.43-0.77 for NetMHCIIpan-4.3 (class II). Gaps of
+0.14 to +0.34 Spearman (tool minus Presto) are mostly positive with bootstrap CIs that exclude zero where more
than two studies exist, but only 1-5 studies per pool exist, so the CIs are wide or degenerate. Thresholded
AUROC at <=500 nM and RMSE tell the same story. The train-mean baseline has no within-study ranking and is
beaten on RMSE by Presto in most pools but not all. **The external tools were very probably trained on many of
these rows (the studies are public IEDB measurements); the overlap audit was impossible, so every external
number is UNAUDITED and the gap is an upper bound on Presto's true deficit.**

## What was run

* Dumps (per-example, validation and test): `2026-10-30_1154_codex_direct-factorized-affinity` seeds 42/43
  (`direct-s42`, `direct-s43`) and `2026-10-29_1956_codex_curated-data-retrain` seeds 42/43
  (`curated-multi-s43` is the validation-selected headline of that family). Input hashes in
  `results/run_summary.json`. Tool predictions depend only on (peptide, allele), so one tool run (3,488
  NetMHCpan/MHCflurry pairs, 3,825 NetMHCIIpan pairs) serves all eight dumps.
* Code: `evaluation/external_benchmark.py`, `scripts/external_benchmark.py`, `tests/test_external_benchmark.py`
  (11 unit tests on synthetic data, no binaries). Parallelism <=4 (thread pool of tool subprocesses).
  Study subsampling cap (20,000) never triggered (largest study 1,375 rows).
* Reproduce: `reproduce/launch.sh` (frozen manifest in `reproduce/launch.json`, bootstrap seed 20261031).
* Tool versions (`results/tool_versions.json`): NetMHCpan 4.1b (data/version "NetMHCpan version 4.1b"),
  NetMHCIIpan 4.3e, mhcflurry 2.3.16 with the `models_class1_presentation` 2.3.0 download
  (`models_class1_presentation.20260928`) via `mhcflurry-predict --affinity-only`. Binary SHA-256:
  netMHCpan `cc9a4f95...b8c7`, NetMHCIIpan-4.3 `1f36179d...4ac10`. Licences: DTU academic-use licence
  (internal research only; bundle ships the NetMHCIIpan licence text, sha256 in the versions file; the
  NetMHCpan-4.1 directory has no licence file, only the readme). MHCflurry: Apache-2.0 upstream.
  Tool outputs are third-party licensed predictions, so per-example tool dumps stay in `artifacts/`.
* Per-example dumps (every tool plus Presto, one row per scored/excluded row, with exclusion reason):
  `artifacts/2026-10-31_0830_claude_netmhc-benchmark/predictions_<run>_<split>.csv.gz`, plus raw
  `tool_predictions.csv.gz`, supported-allele lists, run logs. SHA-256 of each in
  `results/prediction_dump_hashes.json`. (`artifacts/` is gitignored; regenerate with launch.sh.)

## Row counts after exclusions

Rules as pre-registered. **No row was excluded by any rule in any dump**: every in-scope row was single-allelic,
four-digit, standard residues, in the length window, supported by every applicable tool, and received a finite
prediction from every tool (`results/<dump>__census.csv`; zero `multi_or_ambiguous`, `bad_residue`, `length`,
`unsupported_allele`, `tool_failure`). Out-of-scope endpoints (`ec50`, `td50`, `half_life`, `koff`, elution) are
not BA targets and were not loaded. The curation of both families had already restricted BA rows to single
resolved alleles, which is why the exclusion counts are zero; this is a property of the data, not evidence that
the rules are untested (unit tests cover each path).

| dump | class I exact / censored / qualitative | class II exact / censored / qualitative |
|---|---|---|
| direct-s42, direct-s43 test | 1313 / 118 / 290 | 982 / 353 / 151 |
| direct-s42, direct-s43 validation (developmental) | 509 / 569 / 689 | 1768 / 153 / 446 |
| curated-multi-s42, -s43 test | 1156 / 63 / 290 | 492 / 311 / 151 |
| curated-multi-s42, -s43 validation (developmental) | 481 / 394 / 689 | 1727 / 70 / 446 |

The curated family contains fewer test rows than the direct family (curation differs), so the families are not
scored on identical populations and must be compared only through the external tools as common reference.

## Headline: within-study test Spearman (mean of per-study Spearman, 95% study-level bootstrap CI)

Pool = class x assay detector; each study contributes the mean of its (endpoint) units with n >= 20 rows.
Never aggregated across detectors or classes. Held-out TEST. "studies/rows" is the pool size; with 1-2
studies the percentile CI is degenerate or very wide, so treat the CI as uninformative there.

| class | detector | studies/rows | Presto direct s42 | Presto direct s43 | NetMHCpan-4.1 / NetMHCIIpan-4.3 | MHCflurry |
|---|---|---|---|---|---|---|
| I | fluorescence | 3 / 607 | .371 [.186, .694] | .402 [.142, .751] | .595 [.383, .724] | .548 [.480, .652] |
| I | radioactivity | 2 / 692 | .324 [.207, .441] | .301 [.299, .303] | .638 [.455, .822] | .580 [.369, .791] |
| II | fluorescence | 5 / 804 | .389 [.245, .541] | .368 [.233, .538] | .697 [.531, .817] | n/a |
| II | radioactivity | 1 / 178 | .053 | .167 | .434 | n/a |

| class | detector | studies/rows | Presto curated s43 (selected) | Presto curated s42 | NetMHCpan-4.1 / NetMHCIIpan-4.3 | MHCflurry |
|---|---|---|---|---|---|---|
| I | fluorescence | 2 / 450 | .398 [.211, .585] | .408 [.246, .569] | .700 [.677, .724] | .582 [.513, .652] |
| I | radioactivity | 2 / 692 | .501 [.302, .700] | .504 [.376, .633] | .638 [.455, .822] | .580 [.369, .791] |
| II | fluorescence | 3 / 314 | .553 [.504, .596] | .535 [.468, .605] | .766 [.686, .867] | n/a |
| II | radioactivity | 1 / 178 | .283 | .238 | .434 | n/a |

Paired gap (tool minus Presto, Spearman, same studies and bootstrap draws; full table in
`results/headline_pooled_all.md`): direct s43 test: class I fluorescence NetMHCpan +.192 [-.027, +.535],
MHCflurry +.146 [-.099, +.371]; class I radioactivity NetMHCpan +.337 [+.156, +.519], MHCflurry +.279
[+.070, +.488]; class II fluorescence +.329 [+.141, +.507]. Curated s43 test: class I fluorescence NetMHCpan
+.302 [+.092, +.513]; class I radioactivity +.137 [+.122, +.152]; class II fluorescence +.213 [+.090, +.308].
Presto beats a tool nowhere in the pooled mean-of-study Spearman on test. Validation (developmental) gaps are
smaller for the curated family (class I +.10 to +.13 vs NetMHCpan; CI includes 0 for seed 42), and class II
validation fluorescence for the direct family has Presto at .61-.67 vs .85.

Pooled-study RMSE (log10 nM) and <=500 nM AUROC/balanced accuracy/F1/precision/recall/accuracy/AUPRC are in
`results/headline_pooled_all.csv` and `results/<dump>__units.csv` (per class x endpoint x detector x study, with
row-bootstrap CIs for Spearman and RMSE). Direct s43 test AUROC (class I fluorescence) Presto .583 vs
NetMHCpan .754 / MHCflurry .757; class II fluorescence .653 vs .871.

Study-level contrast worth noting (direct s43 test, per-unit): the single largest class I unit
`pmid:42048449` (592 kd_via_ic50 radioactivity rows) has Presto rho .303 vs NetMHCpan .822 / MHCflurry .791;
curated s43 reaches .700 there. Several tiny units (n=2, n=12) are excluded from pooling (n < 20).

## Censored rows (separate from regression)

`results/<dump>__censored.csv`: per class x detector x study x endpoint x direction x bound source, with n,
satisfied fraction, violation count, median/mean log10 violation. All test censored rows are explicit lower
bounds (`>`); bound-source `inferred` rows exist only where the curation marks them (none in test). Example
(direct s43 test, satisfaction fraction): class I fluorescence lower bounds n=118: Presto .347, NetMHCpan .051,
MHCflurry .051; class II fluorescence n=311: Presto .045, NetMHCIIpan .572. Tool predictions rarely exceed
~5e4 nM, so tools cannot satisfy the very large (>=1e5 nM) floors; this measures the bound, not binder
ranking.

## Qualitative (literal Positive/Negative `qualitative_binding`, identical rows, BA score)

`results/<dump>__qualitative.csv`. Test: class I `pmid:17429413` (36 pos/254 neg) AUROC Presto .831 (direct
s43) / .900 (curated s43), NetMHCpan .887, MHCflurry .902; class II `pmid:11433380` (72/52) Presto .627/.813,
NetMHCIIpan .644. Not a ligand-vs-generated-background test. EL scores and generated-background AUROC are
out of scope here.

## Baselines

TRAIN-only per-endpoint mean log10(nM) from each run's frozen `scales` (computed on all exact TRAIN rows).
Constant, so within-study Spearman/Pearson are undefined and AUROC is 0.5. Pooled test RMSE, Presto direct s43
vs train-mean: class I fluorescence 1.684 vs 1.429 (Presto WORSE than a constant), class I radioactivity
1.216 vs 1.548, class II fluorescence 1.190 vs 1.358, class II radioactivity 0.869 vs 0.947 (see
`results/headline_pooled_all.csv` for every pool and family). A class x detector assay-only baseline cannot be
fit because full TRAIN labels are not in the dumps (the TRAIN probe dump has 16,384 rows, about 94% elution);
this is a stated limitation, not substituted.

## Overlap audit: not possible (results UNAUDITED)

* NetMHCpan-4.1 / NetMHCIIpan-4.3: the bundle `data/` holds pseudosequences, synlists, thresholds; the training
  peptide sets are not shipped (`netMHCpan-4.1/data/README` points to a DTU-internal path).
* MHCflurry: only the 2.3.0 `models_class1_presentation` weights are installed, no curated training table.
* Partial evidence only: each tool prints a per-allele "Distance to training data". Across the alleles used it is
  0.000 for nearly all (class I mean .006, max .137 over 61 alleles; class II mean .003, max .064 over 23), i.e. the
  alleles are in the tools' training. Peptide-level overlap is unknown. Several studies (e.g. `pmid:17942551`,
  `pmid:17291638`, `pmid:17429413`) are classic public IEDB panels that are very likely in these tools' BA
  training data. Treat the gap as an upper bound on Presto's deficit on genuinely new data. A fresh confirmation
  set is still required before any broad claim (issue #140).

## Deviations from the pre-registration

1. The curated-retrain dump has no structured `assay_method`; the detector is read from the IEDB assay-method
   string (`.../fluorescence|radioactivity`) when `assay_method` is absent (tested). The first run used detector
   "none" for that family; it was replaced before any result was recorded here, because pooling across detectors
   is forbidden by the pre-registration. Results here are from the corrected run (`results/run_summary.json`).
2. Pooling unit: units are class x endpoint x detector x study; the study-level bootstrap resamples studies, and a
   study with several endpoint units contributes the mean of them (pre-registration said "per study").

## Interpretation limits

* 1-5 studies per pool: CIs are fragile; do not read the CIs of one-study pools at all.
* Test and validation studies were seen in earlier reports; none of this is a fresh confirmation set.
* Presto predictions come from the track selected by the row's assay method; no recalibration was applied.
* Seed variation within a family is small compared with the gap to the tools, except class I radioactivity in the
  direct family (.30-.32) versus the curated family (.50).
