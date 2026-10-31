# Pre-registration: external NetMHC/MHCflurry benchmark (issue #140)

Registered 2026-10-31 BEFORE any tool was run on held-out rows and before any
Presto-vs-tool metric was computed. Only row counts, allele names, peptide
lengths and field formats of the existing prediction dumps were inspected.
Changes after this commit must be listed in the experiment README under
"Deviations".

## Question

How far are held-out Presto affinity predictions from NetMHCpan-4.1 (class I),
NetMHCIIpan-4.3 (class II) and MHCflurry 2.3.16 (class I) on identical
peptide/MHC rows, measured within assay detector and study?

## Inputs (per-example dumps, CPU only)

* `2026-10-30_1154_codex_direct-factorized-affinity`: seeds 42 and 43,
  test and validation dumps.
* `2026-10-29_1956_codex_curated-data-retrain`: seeds 42 and 43 (validation
  selects s43; s43 is the headline, s42 is shown for completeness), test and
  validation dumps.
* Validation results are DEVELOPMENTAL (used for checkpoint selection).
  Test results are the held-out numbers, but the test studies were also seen
  in earlier reports, so none of this is a fresh confirmation set.

## Row eligibility (identical for every method)

1. Endpoints `ic50`, `kd`, `kd_via_ic50`, `kd_via_ec50` (binding affinity,
   `raw_unit` nM). `ec50`, `koff`, `td50`, `half_life`, elution and generated
   background rows are not BA targets and are not scored.
2. Regression comparison: `qualifier == "="` only (literal exact value).
   Censored rows (`>`, `>=`, `<`, `<=`) are reported separately by bound
   satisfaction and violation magnitude, never in regression metrics.
3. Allele rule (prospective). Exactly one MHC entry whose
   `provenance.catalog_alleles` resolves with mhcgnomes to human HLA alleles at
   four-digit resolution.
   * Class I: one allele, written `HLA-A0201` for NetMHCpan and `HLA-A*02:01`
     for MHCflurry.
   * Class II: DR = `HLA-DRA*01:01` plus one DRB1/3/4/5 allele, written
     `DRB1_0401`; DQ/DP = one alpha plus one beta of the same locus, written
     `HLA-DQA10501-DQB10303`.
   * Anything else (several alleles, bags, non-human, non-four-digit, any other
     gene combination) is counted as `multi_or_ambiguous` and dropped.
   * Alleles absent from a tool's own allele list are `unsupported`. A row
     unsupported by ANY tool applicable to its class is dropped for ALL methods.
4. Peptides: only the 20 standard amino acids. Length window class I 8-15
   (NetMHCpan-4.1 / MHCflurry), class II 9-30 (NetMHCIIpan-4.3). Outside the
   window the row is dropped for all methods and counted.
5. Tool failure rule: a row missing a finite prediction from any applicable
   tool after the run is dropped for all methods and counted by tool.
6. A study with more than 20,000 eligible rows is subsampled to 20,000 by
   ascending SHA-256(observation_id) and the subsample is recorded. (Not
   expected to trigger.)

## Predictions compared

* Presto: the dump's `prediction` (log10 nM, track selected by the row's
  assay method). Not re-fit, not recalibrated.
* NetMHCpan-4.1b: `-BA`, `Aff(nM)`. NetMHCIIpan-4.3e: `-BA`, `Affinity(nM)`.
  Peptide input, default settings, one allele per invocation, up to 4
  concurrent processes.
* MHCflurry 2.3.16: `mhcflurry-predict`, `mhcflurry_affinity` (nM) from the
  installed default models.
* Baselines (TRAIN only): per-endpoint TRAIN mean log10(nM) read from the
  run's frozen `scales` (computed on all exact TRAIN rows). Full TRAIN labels
  are not stored in the dumps, so a class x detector assay-only baseline cannot
  be fit without leakage or a TRAIN probe sample; this is reported as a
  limitation, not substituted. A constant baseline has no within-study
  Spearman/Pearson/AUROC; RMSE and threshold accuracy are reported.

## Metrics

* Unit of analysis: class x endpoint x detector x study (`provenance.study_id`).
  Per unit: n, Spearman, Pearson, RMSE in log10(nM); binding at <= 500 nM
  (truth from the raw nM value): accuracy, balanced accuracy, precision,
  recall, F1, AUROC, AUPRC. Per-unit 95% CI: row bootstrap (2,000 draws,
  seed 20261031).
* Pooled within study, per class x detector (never across detectors or
  classes): unweighted mean of per-study metric over studies with n >= 20 and
  a defined value, plus Fisher-z mean of Spearman weighted by n-3. 95% CI: study-level
  cluster bootstrap (2,000 draws, studies resampled with replacement, seed
  20261031), with the paired Presto-minus-tool difference from the same draws.
* No metric is pooled across assay groups as a headline.
* Censored: per class x detector x study x direction, n, satisfaction rate and
  median/mean log10 violation per method, with `inferred_panel_limit` bounds
  separated from explicit bounds.
* Qualitative (`qualitative_binding`, literal Positive/Negative): AUROC per
  class x study on the identical rows, tool score = -log10(nM) BA, Presto score
  = the dump prediction for that endpoint. Not a ligand-vs-background test.
* EL scores and generated-background AUROC are out of scope for this PR.

## Overlap audit

NetMHCpan/NetMHCIIpan training peptide sets and the MHCflurry curated training
table are not present in the bundle (`data/` holds pseudosequences, synlists and
thresholds only; the bundle README points to a DTU-internal path). The
(peptide, allele) overlap audit is therefore impossible and all external
results are labeled UNAUDITED. As partial evidence only, each tool prints a
per-allele "Distance to training data"; it is recorded.

## Outputs

`experiments/2026-10-31_0830_claude_netmhc-benchmark/` with README,
reproduce/{launch.sh,launch.json}, results tables, per-example prediction dumps
(gz CSV), tool versions and hashes; entry in `experiments/experiment_log.md`.
Code: `evaluation/external_benchmark.py`, `scripts/external_benchmark.py`,
`tests/test_external_benchmark.py`.
