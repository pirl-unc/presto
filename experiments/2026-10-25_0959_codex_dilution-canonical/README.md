# PXD024809 canonical nominal-dose evidence — source replay passed

Date: 2026-10-25. Agent/model: Codex / GPT-6.

The canonical source package now builds and independently reconciles to the
original files. Build attempt 3 passed in 80.89 seconds with 1.70 GiB peak RSS;
independent replay attempt 1 passed in 83.24 seconds with 2.23 GiB peak RSS.
All 29,039,248 literal source cells and 297,876 numeric comparisons reconcile.
The required full default local Hitlist suite passes 3,571 tests. No predictor
was trained, and source validation does not establish peptide concentrations.
Shared Hitlist release and the separately registered training contract remain
pending. The full peptide-availability objective is unachieved.

## Source and scientific contract

The native response-only model failed its fixed development gates, so feature
and optimizer search on that cohort remains stopped. Independent observation
constraints are required before interpreting intensity as peptide availability.
PXD024809 supplies nominal-dose standards in a serially diluted JY HLA-eluate
matrix. Post-isolation standards do not measure native intracellular abundance.

Seventeen originals, totaling 402,761,771 bytes, remain under
`data/dilution_observer_sources/` in source families0813/0844. The exact paths and
SHA256 values, prior independent audit hashes, package destination and source
revision are frozen in `reproduce/launch.json`. The full canonical package is:
`data/dilution_observer_canonical/2026-10-25_0959_codex_dilution-canonical/`.
Its manifest SHA256 is `e81fc00bbe8b952fb37ddd63c0c369b31936bc78f1cbb109c9354254bbf5f486`.

The schema preserves literal numeric/identifier values and missingness without
filtering, imputation or training-label conversion. Library and scored/aligned
identifiers occupy distinct namespaces. The full ten-standard panel includes
one absent library standard, which is not a negative. Nine standards have query
support; they do not establish broad sequence transfer. Keep nominal solution
versus injected/certified amount, the twofold-versus-1:10 source conflict and the
different DIA1/DIA2 windows explicit. Filename evidence does not establish a
physical specimen, independent replicate or an actual per-run instrument receipt.

## Execution and reproducibility

Presto execution commit: `d92606a4ec2cde5cd9f9edd255a62108203f6556`, with unrelated
working task edits preserved. Hitlist candidate2.0.40:
`1b3ad5639a7bbbea7a24fdde41c25e12a1bc2511`, [PR846](https://github.com/pirl-unc/hitlist/pull/846),
[issue845](https://github.com/pirl-unc/hitlist/issues/845). The local candidate
was clean; formatting left all340files unchanged. No shared installation changed.

Python3.12.6, pandas3.1.0, pyarrow26.0.0 and psutil7.2.2 on macOS26.5/arm64.
Local CPU only; no Modal/GPU/cloud training. The frozen environment sets
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1` and all of OMP/OpenBLAS/MKL/
VECLIB/NUMEXPR numerical thread counts to1. Exact executable and source snapshots
are in `reproduce/source/` and `reproduce/hitlist-source/`.

Executed successful commands, from the modular Presto worktree:

```sh
./experiments/2026-10-25_0959_codex_dilution-canonical/reproduce/launch.sh build --attempt 3
./experiments/2026-10-25_0959_codex_dilution-canonical/reproduce/launch.sh verify --attempt 1
```

Each executed configuration is separately retained in `reproduce/*-launch.json`.
These are historical invocations: do not rerun their attempt numbers or overwrite
the existing package. A reproduction needs a newly registered destination and
attempt identifiers while preserving the pinned inputs and explicit configuration.

Both source stages retain20minutes,3GiB process-tree RSS,1GiB disk reserve plus
32MiB startup margin,50ms polling and OS child-peak checks. Memory admission uses
free plus speculative pages on macOS and excludes inactive pages. Build and local
tests ran sequentially. No tolerance, threshold, source or resource limit changed.

| Condition | Result | Runtime / memory |
| --- | --- | --- |
| Format / lint | Pass; no formatted changes |340files|
| Earlier catalog/source fixture sets |48pass and26pass (overlap)|1.63s /1.68s|
| Independent checker/launcher boundaries |13pass|0.13s; tiny synthetic software fixtures|
| Full default local suite attempt3 |3,571passed;392warnings|379.80s pytest /388.38s wrapper;901,644,288-byte maximum RSS|
| Build attempt1 |Preflight refusal; no child|69.88MiB available versus3GiB floor|
| Build attempt2 |Preflight refusal; no child|774.23MiB available versus3GiB floor|
| Build attempt3 |Pass|80.89s;1,827,373,056-byte OS peak RSS|
| Independent replay attempt1 |Pass|83.24s;2,394,259,456-byte OS peak RSS|

The full default local suite is non-integration, as defined by `./test.sh`.
Hosted workflow40183613623 separately passes all five Python3.10–3.14unit jobs,
lint, integration and combined coverage; docs build40183613700 also passes.
The local gate receipt records the exact commit, command, environment, exit and
log hash. `results/local-full-default-tests-attempt3.log.gz` restores the complete
original log byte-for-byte. The earlier local preflight refusals remain in the
Hitlist worktree. Passed tests need not be repeated on unchanged pinned bytes.

## Independent reconciliation

The public writer recompiles from original sources. The independent verifier
imports no compiler/schema/numeric/identity helpers: it reads literal pandas
chunks, checks prior independently verified0844digests, treats IDs as opaque keys
and compares arithmetic using `fractions.Fraction`.

All17original/snapshot hashes and independent inodes, 1,031,853tabular source rows,
29,039,248literal cells, metadata locators/fields, all raw and parsed measurements,
unique keys and required links passed. Numeric comparisons297,876passed. Source
records include1,361metadata records in addition to the tabular rows.

| Canonical relation | Rows |
| --- | ---: |
| acquisitions | 10 |
| aligned_quantities | 64,070 |
| assertions | 113 |
| assets | 17 |
| comparisons | 297,876 |
| feature_links | 64,070 |
| library_precursors | 12,851 |
| library_transitions | 891,222 |
| material_contexts | 13 |
| material_links | 20 |
| nominal_levels | 5 |
| nominal_standards | 10 |
| scored_features | 76,561 |
| source_records | 1,033,214 |
| standard_memberships | 10 |

Feature-link dispositions are49,646exact matches,10,939missing feature IDs and
3,485IDs absent from deposited score exports. Ten library labels preserve empty
versus literalNA states;64,060agree literally. Independent metadata checks cover
13contexts,20material links,10standards,5levels and15protocol assertions. Public
EvidenceStore discovery and representative joins passed after the full replay.
Existing small tamper fixtures exercise rejection after manifest rehash; this
run does not claim an additional full-size corruption experiment.

## Decision and retained evidence

Source conservation is complete for this pinned revision. No model, pretraining,
loss weights, assay-output mapping, synthetic biological training or train/val/test
partition applies. There are no prediction dumps or held-out metrics because no
predictor was fitted; original native outer partitions remain closed. No exact/RL
comparison or model winner is claimed. Next: review, merge and verify the tested
PyPI release, then register a nominal-dose observation/selection diagnostic with
its limited standard support and separate biological-transfer requirements.

`results/build.json`, `results/verification.json`, both successful runtime receipts,
all refused attempts and `results/summary.json` retain the full results. Stage logs
are under `launch_logs/`. Current closure is `reproduce/closure-source-replay.json`;
the original `reproduce/closure.json` describes preparation at commitd92606a.
The original source data are kept outside this experiment directory, with exact
paths in the frozen configuration. Earlier audits: [0813](../2026-10-25_0813_codex_dilution-observer-admission/README.md),
[0844](../2026-10-25_0844_codex_dilution-scoring-lineage/README.md).

Amendment1 preserved attempt1's configuration and added the already frozen source
contract to explicit hash checks before any child launched. Subsequent attempts
used unchanged code/configuration. After user-reported memory recovery, the local
suite completed; concurrent activity reduced headroom before build2. A read-only
process audit identified active work in other repositories, which was left intact.
Memory recovered again before build3. No abandoned Presto test worker was found
among readable project processes; process visibility was not exhaustive.
