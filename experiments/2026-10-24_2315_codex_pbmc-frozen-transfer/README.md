# Frozen A549-to-PBMC transfer: support fails before model scoring

Codex / GPT-6, October 24, 2026. [Prospective contract](reproduce/plan.md),
[source/model/environment pins](reproduce/launch.json),
[complete decision](results/decision.json), and [all source-cell dispositions](results/cohort/decisions.parquet).

**Neither cohort passes the registered minimum of 20 sequence/parent dependency
components. No model was scored or fitted.** This is an independently verified
source/endpoint qualification result, not a negative or positive prediction result.
The prior A549 result remains within-study evidence; this family supplies no
external-validation claim. No threshold, grouping policy or eligibility rule was
relaxed after seeing support.

| Original source cohort | Components | Accession/run groups | Measurements | Unique peptides | Support |
|---|---:|---:|---:|---:|---|
| CTRL, primary | 6 | 27 | 234 | 82 | Fail |
| LUPUS, descriptive secondary | 7 | 20 | 161 | 66 | Fail |

Each cohort required at least 20 components, 20 groups and 100 measurements.
Components constrain shared sequence/accession dependence; they are not counts of
independent biological specimens. Repeated acquisitions do not establish physical
specimen independence.

## Source and endpoint

Original PXD009438 MaxQuant export: 985 peptide rows, 8,851 evidence rows, 283
protein-group rows, 14 acquisitions, and 13,790 peptide/run intensity cells.
Seven original assets and the 199,785,170-byte search archive are bound by SHA256;
the archive also matches its PRIDE SHA1. Existing local bytes were reused without
a download. The source append preserves every byte of the predecessor reference
generation using independently verified copy-on-write snapshots; those reference
rows are not PBMC training examples.

Source generation SHA256 is
`63752e0221b53b8e67ccdfa21c87b8babeb58296a458c9bfa2c6159275318b67`.
It was written with the frozen shared PBMC source adapter on Hitlist commit
`5d648602326412cbad79bc47b1ea70fc3bb1c45d`, version 2.0.31. The separate new
registered recipe uses candidate Hitlist 2.0.32; its runtime files are frozen in
`reproduce/recipe_source/`. The installed shared environment was unchanged.
Presto registration base was `c7fa66826883dc953bc5e3e0deabe261177f07ef` with recorded
unrelated task dirt and new frozen source/admission/audit code.

The shared recipe `pxd009438_observed_baseline_composition_v1` requires positive
finite original reported-run intensity, canonical sequence, explicitly blank
peptide reverse/contaminant flags, peptide PEP in [0, 0.01), original per-run
`By MS/MS`, positive direct evidence, all linked evidence intensities reported,
and one distinct supplied parent accession. Every otherwise eligible duplicate
sequence within an accession/run is excluded; none occurred in the actual
eligible population. Groups require three remaining distinct sequences.

All original source sequence/accession nodes, including excluded bridges, enter
the dependency graph before eligibility. Stable namespaced hashes and output
ordering were documented before admission. Weights are equal component, then
group, then peptide, independently within CTRL and LUPUS. Targets are centered
log10 intensity and observed-set fractions. All 13,790 decisions remain available;
580 cells pass initial eligibility, 185 more are excluded from small groups, and
395 remain. Reasons overlap and must not be summed as disjoint counts:

| Exclusion reason | Measurement cells |
|---|---:|
| No positive direct-MS/MS evidence | 10,597 |
| Per-run identification is not `By MS/MS` | 10,423 |
| Nonpositive/nonfinite original run intensity | 6,137 |
| Not one distinct supplied parent | 5,782 |
| Peptide PEP outside [0, 0.01) | 5,040 |
| Peptide flags not explicitly blank | 4,956 |
| Incomplete evidence intensities | 185 |
| Fewer than three remaining sequences per group | 185 |

The deposited CTRL/LUPUS labels are source groups. PBMC is a mixed immune-cell
population, not purified APCs. Publication PBMC versus repository blood-serum
metadata conflict remains explicit. Human is project/search scope; the full
original `Human_UP_2016_02.fasta` and per-parent species are unverified. No donor,
physical specimen, aliquot, clinical normalization or precursor pairing is inferred.
This differs from the prior A549 source-header human-only eligibility axis.

Captured MS intensity is not free-peptide concentration, complete intracellular
availability or a calibrated detector response. Positive observed-set conditioning,
possible molecular-form aggregation and capture/detection effects remain. Missing
or zero intensities are not negative examples; no synthetic biological labels,
imputation, censor bounds or nM/binder metrics were introduced.

## Frozen modeling contract and stop decision

Before admission, the A549 model bundle and prediction/input seals were hashed.
The only planned models were uniform, the validation-selected length ridge and
sequence ridge, exact-gradient training, and REINFORCE seeds 42/43/44. All 61 input
features, original scaling, SVD projection and coefficients would remain frozen.
No additional optimization, pretraining, calibration or sampling was planned.
Other Presto processing/MHC/TCR/context/detector branches were not trained.

This was external-test-only by design: no PBMC training or validation partition.
Both support gates failed before inference. Consequently there are no model
predictions, predictive metrics, bootstrap intervals or model comparisons in this
family. Prediction/evaluation stages were not executed or implemented after the
failure. The saved cohort examples are measured outcomes, not prediction dumps.
No inspected-label adaptation or replacement family is launched here.

## Verification, resources and software status

Independent CSV decoding reproduces 692,452 original fields. A separate NetworkX
graph and original-column/evidence replay verifies all 13,790 decisions, 1,756
component nodes, 395 targets and their weights. Maximum target/weight discrepancy
is 1.78e-15. Original reported-versus-evidence-sum relative discrepancy is
6.50e-5, consistent with source rounding. Closure additionally checks all 14
literal run aliases and exact label-free/label-file projections. Frozen A549
model/seal hashes are unchanged. Source and recipe manifests are retained.

| Local stage | Seconds | Peak polled RSS, GiB |
|---|---:|---:|
| Source append and qualification | 62.66 | 0.897 |
| Shared recipe and input sealing | 17.16 | 0.720 |
| Independent original-source audit | 0.83 | 0.258 |
| Support-failure closure | 0.35 | 0.079 |

One local CPU numerical thread, no cloud/GPU/installations. All 4-GiB RSS,
20-minute stage and one-hour family limits pass. Bounded physical allocation stays
under 128 MiB and free disk above 1 GiB. The preexisting source/reference is a CoW
snapshot, not a new 3.8-GiB physical copy. These timings are single runs.

Twenty-eight focused PBMC/Goldberg source and recipe tests, required format and lint pass.
Initial two lint findings and a fixture's incomplete reciprocal protein-group IDs
were corrected before admission without weakening source checks. The full local
suite remains deferred below the separately registered 2-GiB headroom threshold;
about 1.42 GiB was available. [Shared recipe PR830](https://github.com/pirl-unc/hitlist/pull/830) is published at
`1619c374f421cc7775b4c6619e748384e81be346`; all frozen runtime files remain
byte-identical after rebasing onto merged MAPP recipe829. Hosted CI, merge and
verified recipe release remain unfinished. Source Hitlist2.0.30 is published to
PyPI and both distribution bytes match the tested clean-main artifacts. The canonical supervisor gained only the
`closure` stage after the original launch; original supervisor and executed stage
snapshots are retained.

## Reproduction and artifacts

Use fresh source/recipe/output destinations; the launcher refuses existing stage
receipts and package paths. From the pinned tree/environment, run in order:

```sh
for stage in source admit audit closure; do
  sh experiments/2026-10-24_2315_codex_pbmc-frozen-transfer/reproduce/launch.sh "$stage"
done
```

Exact source, predecessor, recipe, Python and frozen-predictor paths are in
`reproduce/launch.json`. Shared raw inputs remain under
`/private/tmp/hitlist-pxd009438-source-20261024/`; canonical generations are under
repo-level `data/pbmc_transfer/`. This directory contains the recipe outputs,
original/recipe manifests, support summary, input hashes, executed code, resource
receipts and logs. No test prediction dump exists for the reason above.

The next biological comparison needs a prospectively qualified independent
cohort with adequate support and explicit observation context. The full APC
concentration/availability objective remains active and unachieved. Existing
Hitlist material-context issue #820 retains the unresolved physical-link scope;
closed source-admission issue #822 is not reopened by this endpoint failure.

[Hitlist820 follow-up](https://github.com/pirl-unc/hitlist/issues/820#issuecomment-6390908317)
records support and unresolved reference/material needs. The source issue stays closed.


Software closure update, October24,23:56UTC: all hosted unit3.10–3.14, integration,
lint, docs and coverage checks pass for1619c374. PR830mergededf061a2; clean-main
release40144589370is pending behind MAPP2.0.31run40143097533. The original local
headroom deferral and biological support failure remain unchanged.
