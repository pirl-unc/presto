# PXD024809 canonical nominal-dose evidence — execution pending

Date: 2026-10-25. Agent/model: Codex / GPT-6.

The independent replay and source-build commands are frozen, but **the original
source package has not been materialized or independently replayed**. Attempt 1
refused before launching a child: free plus speculative memory was 73,269,248
bytes (69.88 MiB), below the unchanged 3 GiB startup floor. Disk headroom was
14,606,356,480 bytes. This is an execution-admission result, not a scientific
success or failed biological fit. No model prediction has changed.

## Scientific purpose and fixed source contract

The native response-only model failed its development gates, so further feature
or optimizer search on that cohort remains stopped. Before interpreting signal
as peptide availability, the MS observation process needs independent constraints.
PXD024809 contributes nominal-dose standards in a serially diluted JY HLA-eluate
matrix. Post-isolation standards do not directly measure intracellular abundance.

The source contract pins 17 admitted originals totaling 402,761,771 bytes. The
previous independently verified audits establish 1,031,853 tabular source rows,
29,039,248 literal cells, 76,561 scored features, 891,222 library transitions,
12,851 library precursor memberships and 64,070 aligned slots. Those are
**expected source-conservation counts**, not new canonical-package results.

All original data stays under `data/dilution_observer_sources/` in the 0813 and
0844 source families. `reproduce/launch.json` records exact asset locations,
hashes, prior-audit hashes and the new, non-overwriting package destination:
`data/dilution_observer_canonical/2026-10-25_0959_codex_dilution-canonical/`.

The source revision preserves all raw values and missingness. It applies no
confidence filter, imputation, censor threshold or assay-to-training-label
conversion. The full ten-standard panel includes one absent library standard;
absence does not become a negative. The twofold versus 1:10 source conflict,
nominal solution versus injected amount, and different DIA1/DIA2 window
definitions remain explicit. No physical specimen or independent replicate is
inferred from a filename.

## Software and reproduction

Hitlist candidate 2.0.40, commit
`1b3ad5639a7bbbea7a24fdde41c25e12a1bc2511`, is in draft
[PR #846](https://github.com/pirl-unc/hitlist/pull/846), under
[issue #845](https://github.com/pirl-unc/hitlist/issues/845).
The production source bytes match 7c9fa84; subsequent commits correct seven
global catalog expectations and retain the hosted failure log. Historical
package envelopes are unchanged.

Presto preparation base: `50a2af40f32cad0f297d71884c9215310f17011d`, with existing
unrelated `tasks/todo.md` edits preserved. No shared environment installation.
The exact Hitlist files used, immutable contract, local scripts and plan are
snapshotted in `reproduce/`. Local CPU only; no Modal, GPU or cloud training.
All numerical library thread overrides are fixed to one in `launch.json`.

Software checks completed:

| Check | Result |
| --- | --- |
| Hitlist format / lint | Pass |
| Six catalog test files plus source fixtures | 48 passed, 1.63 s |
| Catalog subprocess correction plus source fixtures | 26 passed, 1.68 s; overlaps the previous set |
| Independent-checker numeric and launcher boundaries | 13 passed, 0.13 s |
| Full local default suite | Pending; earlier preflights refused at 0.25 and 0.08 GiB |
| Real canonical source build / independent replay | Not launched |

The 13 boundary tests use tiny synthetic software fixtures only, including
numeric missing/zero/overflow states, rational tolerance boundaries, exclusion
of inactive memory, fail-closed probing, and refusal without child launch or
overwriting a previous receipt. They do not validate the biological source.

Before running source stages, use the same Hitlist checkout and shared Python
PATH to run `./format.sh`, `./lint.sh`, then `TEST_SH_MAX=1 ./test.sh`. Preserve a
new full-suite log without overwriting the two refusals. Only after actual
exit-zero completion, record `results/local-software-gates.json` with
`full_default_suite_exit_code: 0`, the exact `hitlist_commit`, and retained
log path/hash. Hosted CI alone does not satisfy this local prerequisite.

From the Presto modular worktree, after the same memory and software guards pass:

```sh
./experiments/2026-10-25_0959_codex_dilution-canonical/reproduce/launch.sh build --attempt 2
./experiments/2026-10-25_0959_codex_dilution-canonical/reproduce/launch.sh verify --attempt 1
```

Do not rerun attempt 1 or overwrite an existing package. Every later attempt gets
a new number and retains failures. Changed source/verification code requires a
prospective amendment and newly frozen invocation; executed snapshots stay intact.

Each source stage retains 20 minutes, 3 GiB process-tree RSS, 1 GiB disk reserve
plus 32 MiB startup margin, 50 ms process monitoring and OS child peak reporting.
Startup memory on macOS means free plus speculative pages, excluding inactive
pages. The build and full test suite must not run concurrently.

## Independent verification coverage prepared

`build.py` uses the public writer, including its full source recompile. The
separate verifier imports no compiler/schema/identity/numeric helpers. It uses
pandas literal chunks and prior independently verified row digests, treats
canonical identifiers as opaque keys, and uses `Fraction` arithmetic for exact
numeric differences. Its pending checks cover:

- All original/snapshot hashes and independent inodes; the full package inventory.
- Every tabular and metadata record, header, cell order, locator and source index.
- All library transitions and form/charge/target memberships, every scored and
  aligned raw/parsed measurement, exact feature links and 297,876 comparisons.
- The intended standard panel and nominal levels, all context claims/links,
  scoped unknowns, both acquisition methods, and 15 source-backed protocol claims.
- All 15 table counts, primary-key uniqueness and required relationship targets;
  representative public EvidenceStore joins after source reconciliation.

These checks are specified and implemented, but have **not** passed on the real
package. Corruption rejection is established only in the existing small Hitlist
fixtures until the original-source work can execute.

## Training, evaluation and decision

No model, pretraining, synthetic training, loss terms, target mapping or
validation/test partition applies to this source-conservation family. No
held-out metrics or prediction dumps exist because no predictor was fitted.
Original native outer partitions remain closed. There is no winning model.

The continuation produced draft PR publication, corrected catalog software
contracts and a frozen independent replay; it does not improve biological
prediction quality. Next: complete local source verification, all software gates,
review and tested PyPI publication before registering a nominal-dose likelihood
and selection contract. Nine selected standards cannot alone establish general
sequence response, native MS1 calibration or APC concentrations. The full
availability objective remains open.

Evidence: `results/software-boundaries.log`,
`results/build-attempt1-runtime.json`, `results/summary.json` and
`reproduce/launch.json`. Original audits:
[0813](../2026-10-25_0813_codex_dilution-observer-admission/README.md),
[0844](../2026-10-25_0844_codex_dilution-scoring-lineage/README.md).

Hosted checkpoint: at Hitlist1b3ad56, all five Python3.10–3.14unit jobs, lint
and docs pass. Integration121139728815/run40183613623 remains live. Exact
observed job states are in results/hitlist-ci-observed.json; this does not
certify local full-suite execution or independent original-source replay.

Configuration amendment1 retains attempt1's exact config separately and adds
the already frozen source-contract JSON to the future runner's explicit hash
checks. No source bytes, executable code, thresholds or limits change; no real
source stage had launched. See reproduce/amendment-1.json.
