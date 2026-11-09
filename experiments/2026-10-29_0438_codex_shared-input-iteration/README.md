# Shared sequential input iteration

Status: **completed; shared sequential reader improvement retained**.
Agent/model: Codex / GPT-6. Date: 2026-10-29 UTC.

The shared `ObservationIndex.__iter__` now holds one input stream for each
contiguous source segment. It still checks the pinned inode, size and modification
time for every row and runs exactly the same evidence normalization. Random
access, model, loss, optimizer and source membership are unchanged. There is no
cached preflight result or skipped admission. Existing temporal H100 fits retain
their frozen aa2b3f7a runtime.

## Dataset and execution

First 5,000 TRAIN lines from each corrected temporal physical-assay, native-MS and
generated-background file: **15,000 rows, 104,398,183 bytes**. The inputs are exact
source prefixes, not a representative accuracy cohort. Full-source hashes are
inherited from the verified staged materialization; every captured prefix and
archive has a checked hash. No validation/test data, new synthetic data, model,
pretraining, training loss, optimizer, or inference is used. Existing background
records (5,000 of the 15,000 rows) are read as original bytes solely for timing.
The executed invocation's `synthetic_data: none` means no new generation;
`results/metadata-clarification.json` makes the existing-comparator scope explicit. Predictive metrics and
prediction dumps are inapplicable to this reader-only experiment.

CPU contract: local Apple M1 Pro, Python 3.12.6, warm file caches after a full-content
comparison. No GPU computation or new Modal training app. The capture reads the
already staged TRAIN files in the existing worker and writes only a temporary
archive; it does not mutate or commit training state.

Both conditions traverse the same admitted index. Three paired repeats alternate
order: original/sequential, sequential/original, original/sequential. Timings include
iteration, normalization and an identical ID digest. Full normalized-row equality
is verified separately outside timing. Complete timings are in `results/timings.csv`.

| Comparison | Original median | Sequential median | Throughput ratio |
|---|---:|---:|---:|
| Initial comparison | 1.1730 s | 0.9156 s | 1.2812× |
| Frozen-baseline replay | 1.1795 s | 0.9292 s | 1.2694× |

The predeclared gate was >=1.10× median throughput plus all integrity checks.
Both comparisons pass. The replay corrected the launcher's dependency on mutable
Git HEAD for its baseline; initial results remain under `attempts/initial-reader-benchmark`.
These timings do not establish a full-cohort, cold-cache or H100 fit speedup.
A small timing table is sufficient; no scientific plot is needed for six repeats.

## Integrity and source evidence

- All 15,000 normalized observations match exactly, including provenance and order.
- 28 focused integrity/resume tests pass: mid-stream edits, atomic replacements,
  next-source edits, independent iterators, explicit closure, exact fitted weights,
  optimizer state, coordinate evidence, selected predictions and resume behavior
  on the existing ten-row artificial software fixture. These tests do not measure
  biological prediction.
- 212 pinned runtime files are unchanged; only the common source iterator differs.
  Model and trainer bytes are identical. See `results/runtime-parity.json`.
- The first large CLI stdout capture and a chunked retry were incomplete. Inspection
  found that Modal 1.6.1 CLI exec awaits process exit without draining background
  output printing. A direct SDK transfer awaits both PIPE streams and exit status,
  then verifies archive size/hash. A first SDK attempt needed the documented-in-code
  synchronizer event-loop bridge. All three failed captures are preserved; no rows
  from them entered the benchmark. The successful wrapper and remote program are
  frozen under `reproduce/source/`; no shared environment package was edited.

## Reproduction

Baseline repository commit: `393b3a7e` (scientific runtime aa2b3f7a plus experiment
records). The only candidate source change is frozen in
`reproduce/source/modular_sources.candidate.py`; the original is alongside it.
The exact executed benchmark and capture programs, hashes, Python/platform and
invocation are in `reproduce/`. The later source formatting does not change the
executed snapshots. To replay, use a fresh result directory/family copy, retain the
capture receipt and original artifacts, restore the frozen benchmark into `code/`,
and copy the candidate source to `data/modular_sources.py` in that pinned checkout.
`reproduce/launch.sh` invokes the comparison; completed summaries are protected
against accidental overwrite.

Raw prefixes/archive and the manifest are in
`artifacts/2026-10-29_0438_codex_shared-input-iteration/`, referenced by absolute paths
and hashes in `results/capture.json` and `reproduce/launch.json`. If those local
prefixes are absent, their first-5,000-row contract can be reconstructed from the
immutable reviewed cohort on Modal volume `presto-checkpoints` under
`2026-10-28_1540_codex_temporal-multi-output/revisions/kinetic-review-v2/prepared/`.

Decision: keep the measured common-reader improvement. The full suite passes
**2,860 tests, four skips**, in 339.84 seconds; targeted integrity/resume checks
pass 28 tests. No biological-performance or model-promotion decision is made here.
