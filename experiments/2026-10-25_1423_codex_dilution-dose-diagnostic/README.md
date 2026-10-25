# Conditional nominal-dose response diagnostic

Completed 2026-10-25 by Codex / GPT-6. All five stages passed; the simple
unit-slope model gives nonconstant dose responses with 49.75% lower development
error than the constant baseline. This is conditional observation calibration.
The previous turn completed tested Hitlist2.0.40 publication and source closure;
that was progress, not a wait. This family implements the registered next step.

The [frozen plan](reproduce/plan.md), [exact launch configuration](reproduce/launch.json)
and [source snapshots](reproduce/source/) define the contract. Execute stages in
order using `reproduce/launch.sh STAGE --attempt N`: test, admit, audit, fit, verify.
Every attempt is non-overwriting and records code/environment/commit/resource state.
Source data remain once at the canonical package path in launch.json. A small
shared generic recipe bundle is written under data/dilution_dose_recipes/.

The ten intended heavy standards contain nine library-supported forms and 90
aligned slots. Admission requires positive finite signal and exact feature linkage;
all exclusions and the absent intended standard stay explicit. DIA1/DIA2 remain
different methods. Supplied nominal solution concentrations are not certified
injected amounts; the twofold-versus-1:10 source conflict remains unresolved.
All preparation, aliquot, matrix, source/protocol assertions and source coordinates
are retained through shared Hitlist queries. No confidence cutoff or missing-value
imputation is added. Accepted data are conditional on the deposited query library.

Four fixed linear arms fit log10 signal: per-group constants, fixed unit dose slope,
shared learned dose slope, or method-specific slopes. Groups are standard/exact
form/charge/method. Weighted least squares gives each group equal total weight;
all allocated coefficients are estimated, with no encoder or frozen unused heads.
There is no pretraining, synthetic biological data, optimizer, regularization or
hyperparameter selection. Only conditional MS signal is supervised; affinity,
restriction, TCR recognition, APC machinery and availability are not outputs here.

Five leave-level-out development folds hold both methods at the level together;
20 fold fits and four descriptive full fits were completed. No independent test is
claimed because this selected panel was inspected during source qualification and
lacks independent biological/preparation replication. Binding-threshold metrics
and log10(nM) error are inapplicable. Per-example development and descriptive full
predictions, coefficients, membership, error/correlation/spread metrics and exact
fixed-gate decisions will be preserved. Original native outer data stay closed.

Local CPU only, one BLAS thread; no Modal or GPU. Each stage keeps the existing
20-minute, 3-GiB process-tree RSS, 1-GiB disk reserve plus 32-MiB startup margin,
50-ms monitor and OS child peak checks. macOS admission uses free+speculative,
excluding inactive RAM. No shared Python environment installation changes.

The independent audit reads checksummed canonical tables, reconstructs every
standard-slot decision and metadata/target join, then admits fitting. Independent
scalar grouped covariance formulas replay all coefficients and predictions;
metrics are recomputed from archived values with explicit rank-tie arithmetic.


## Observed result and decision

All five stages passed on their first attempts. The generic shared recipe retains
90 candidate slots: 87 accepted positive exact-feature matches, one positive slot
excluded for missing feature identity, and two zero slots excluded with both link
and value-status reasons. All ten intended standards remain recorded, including
one absent from the library. There are 18 supported standard/form/charge/method
groups and five reported levels: 0.2, 0.1, 0.05, 0.025 and 0.0125 fmol/µL.
Acquisition/dose links remain explicitly filename-derived.

The following are OOF **development** results, in log10 arbitrary signal units.
The same 87 rows and five whole-dose folds are used for every arm.

| Arm | Fitted coefficients | Equal-group RMSE | Pooled RMSE | Pearson | Spearman | Centered prediction / target SD |
| --- | --- | --- | --- | --- | --- | --- |
| Per-group constant | 18 | 0.665914 | 0.653779 | 0.5316 | 0.5038 | 0.2757 |
| Fixed unit dose slope | 18 | 0.334620 | 0.321510 | 0.9028 | 0.9400 | 0.8022 |
| Learned shared slope | 19 | 0.340878 | 0.330683 | 0.9003 | 0.9348 | 0.9226 |
| Learned slope per DIA method | 20 | 0.343272 | 0.332993 | 0.8989 | 0.9366 | 0.9237 |

The primary shared-slope comparison improves equal-group RMSE **48.81%** over
constant and passes all four prespecified gates. Its five fitted slopes range
**1.01096–1.23128**, all positive; centered prediction spread is **92.26%** of
observed spread. Equal-group RMSE improves in both DIA methods: **0.719868 to
0.400029** for DIA1, and **0.607185 to 0.269021** for DIA2.

The preferred diagnostic is the **fixed unit-slope arm**: it is simplest and has
the lowest development RMSE, **0.334620**, a **49.75%** reduction from constant.
Its method-specific errors are 0.397497 (DIA1) and 0.256782 (DIA2). Extra learned
slopes slightly improve full-fit loss but do not improve this held-level result.
This descriptive arm choice is not a new independent test or evidence of a
statistically significant difference between the dose-aware arms.

The full-data learned shared slope is **1.098346**; the two descriptive method
slopes are **1.139477 / 1.055775**. The unit-slope model learns only 18 group
intercepts. Its useful shape is `log signal = group offset + log nominal dose`.
It supports roughly proportional dose sensitivity under this assay protocol.
It cannot predict an unseen standard's offset from sequence. The group offset
also contains nominal amount/response/recovery effects and is not an independently
identified universal peptide detectability coefficient.

Constant-arm OOF predictions vary because every fold refits its intercepts on a
different set of levels. Their negative within-group centered correlation is a
leave-one-level-out mean effect; the constant model itself has zero dose slope.
Centered plots/metrics are computed after prediction and never inform fits.
No sample, standard or dose was removed after inspecting residuals.

## Independent verification and runtime

Six production-path software tests pass: known shared/method slopes, equal group
weights with unequal row counts, held-target leakage prevention, identifiability,
positive-value transformation boundaries and actual shared recipe dispositions.
The independent pre-fit audit replays 12 checksummed canonical tables, 4,936
metadata cells, all 90 candidate slots and 261 quantitative transforms. It
reconstructs exactly the 87 admitted IDs before fitting.

Independent scalar grouped-covariance formulas reconstruct all **24 fits** and
**696 development/descriptive predictions**. There are **4,075 numerical checks**;
maximum absolute disagreement is **2.66e-14**, below the fixed 2e-10 tolerance.
Every reported metric and gate recomputes from archived predictions. Exact raw
and centered rank conventions are separately checked; scalar-versus-array
centering changes no reported rank correlation in this run.

Post-fit artifact bookkeeping also verifies all **71,688 copied source fields**
across the 696 prediction records and all 76 comparison-CSV cells. This adds no
model or scientific analysis. All 18 frozen source/template/launcher hashes and
the original plan match their pre-execution values.

| Local stage | Wall seconds | OS peak RSS MiB | Result |
| --- | --- | --- | --- |
| test | 4.999 | 245.86 | passed |
| admit | 33.712 | 1712.45 | passed |
| audit | 1.088 | 1854.28 | passed |
| fit | 1.021 | 139.41 | passed |
| verify | 0.194 | 51.19 | passed |

Total measured stage wall time is **41.01 seconds**. This is a small CPU linear
model diagnostic, not a throughput or GPU benchmark. All unchanged resource
limits pass. No biological training was run on Modal or a remote GPU.

## What this enables and what remains open

Retain unit proportionality as a qualified protocol-specific observation control.
Stop adding complexity to this nine-standard panel. Further progress toward
availability requires peptide-dependent response/amount anchors with independently
supported quantities and compatible protocol transfer, alongside native/APC
precursor and processing data. The existing Hitlist#762 follow-up owns the broader
calibration-source requirements; #845 owns this source/recipe applicability.
Neither an RL optimizer nor exact span enumeration resolves that observation
identifiability gap. No new biological exact-versus-RL comparison is claimed here.

These selected post-isolation heavy standards do not measure intracellular
processing, native extraction recovery, pMHC selection, clearance or amounts of
undetected/sub-ligand products. Results do not fix the older full-model IC50
underfit or qualify a general MS observer. The full goal remains unachieved.

## Retained artifacts

- `results/metadata.json`, `queries.json`, `decisions.json`, `admitted.json`:
  complete panel/context/source provenance and all 90 decisions.
- `results/recipe-resolved.json` plus the generic recipe manifest path/hash in
  `results/admission.json`: shared API materialization, not a Presto raw adapter.
- `results/fold-models.json` and `full-models.json`: every learned coefficient,
  rank, training membership, equal-group weights and training loss.
- `results/development-predictions.json` and `descriptive-full-predictions.json`:
  all per-example predictions; the latter are explicitly not held out.
- `results/summary.json` and `comparison-lf.csv`: complete overall, per-method,
  per-standard and per-level metrics, fixed gates and null-correlation policy.
- `results/admission-verification.json`, `verification.json`,
  `artifact-copy-verification.json`, all runtime receipts and `launch_logs/`.
- `reproduce/closure.json`: required published files and hashes. The original
  shared canonical package and small generic recipe bundle remain once in data/;
  their exact paths and source manifest identities are in the frozen configuration.

The original CSV uses CRLF. `results/comparison.csv.gz` preserves its exact bytes;
`comparison-lf.csv` has identical parsed cells and Git-compatible line endings.
If the original CSV is absent, run `gzip -dk results/comparison.csv.gz` from this
family directory before rerunning artifact-copy verification. No frozen numerical
output was rewritten; `reproduce/csv-archive.json` records both representations.
