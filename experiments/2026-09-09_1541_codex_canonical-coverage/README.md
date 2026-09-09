# Canonical full-source output coverage

- Agent/model: Codex / GPT-6; date: 2026-09-09.
- Status: upload verified; `merged_measured` is running as attempt
  `serialized_entry`, after two preserved startup failures before data loading.
- Production base: merged PR #64, `190360a7e2596eb5f62682088f1ef63270bf5eeb`.
- Plan: [detailed specification](../../tasks/canonical_coverage_evidence_spec.md).
- Prior evidence: [source inventory](../2026-09-09_1106_codex_output-coverage/),
  followed by the independently audited routing, lineage and selector repairs.
- Local raw root: `artifacts/2026-09-09_1541_codex_canonical-coverage/`.

This family measures supervision incidence after canonical curation, complete
MHC resolution, augmentation and peptide-disjoint train/validation/test splitting.
It addresses #48/#50 and establishes eligibility for #53. It does not establish
prediction quality. The source contract changed after the earlier inventory;
its results and launch receipts remain preserved in that earlier directory.

## Frozen conditions and interpretation

`conditions.json` records every resolved trainer argument for separate merged,
exclusive Hitlist and Hitlist-plus-bulk conditions, each measured and augmented.
No modality caps; data seed 17; split seed 42; peptide groups 80/10/10; strict
complete MHC with canonical unresolved filtering; ambiguous source flanks masked.
The existing classifier owns supported assay families and censor/unit handling.
Unsupported source rows remain in the funnel. Bulk-derived detectability/excision
and organism-derived foreignness retain their proxy/auxiliary status. Generated
wrong-enzyme and negative families are counted separately. Aliases do not create
independent endpoints; the actual objective registry records loss multiplicity.

No fitting/pretraining is performed in the census. Validation/test partitions
provide support counts, so prediction dumps and predictive metrics are inapplicable
to this phase. A later update-diagnostic phase must freeze its training-only
candidate IDs, configurations and optimization budget before launch. #53 retains
the requirement for fresh fitting and held-out loss, regression/classification
metrics, baselines and per-example predictions.

## Reproduction and hardware

Launch only from a clean commit after instrumentation tests pass. Freeze the
source tree, all resolved arguments, Linux package environment, source files and
MHC catalog/index hashes. The first condition is `merged_measured`, on Modal CPU:
4 cores requested / 8 limit; 65,536 MiB requested / 196,608 MiB hard limit;
four-hour timeout; no GPU. Inspect actual resource use before further launches.
The installed Modal 1.1.4 client is isolated from the training image. The training
image uses Python 3.12, CPU torch 2.7.0 and the inventory release pins. Record
the actual image environment and all platform-specific dependencies.

The source volume prefix and per-condition result paths are unique to this
family. The launcher records a durable app/call handle. Full SQLite evidence and
logs remain in raw storage; copy all resulting summary JSON/CSV and receipts into
this directory before calling a condition complete. Failed runs retain their own
receipts. Do not overwrite prior results or repair shared caches during an audit.

## Handoff

The live attempt uses clean `d6a690a510750ee66b9224ba4faa76c647e4f0b6`:

- App: [`ap-PPrESGuLvIKrTYEYNnwTPG`](https://modal.com/apps/iskandr/main/ap-PPrESGuLvIKrTYEYNnwTPG).
- Call: `fc-01M23HN1ZCVA1M5NF5Q1P0GFNJ`; container:
  `ta-01M23HN2CS1NEKZ4DS3Z430S7R`.
- [Launch and startup receipts](results/merged_measured/attempts/serialized_entry/).
- Remote output: `presto-checkpoints`, path
  `2026-09-09_1541_codex_canonical-coverage/merged_measured/attempts/serialized_entry`.
- Worker log: same volume,
  `2026-09-09_1541_codex_canonical-coverage/logs/merged_measured-serialized_entry.log`.

The worker entered successfully and reached the canonical merged-data loader
after checking all frozen source/input hashes and the pinned Hitlist curation
hash. Direct container log inspection confirms the full merged input path;
support counts are still pending. Observed environment is Python 3.12.1,
torch 2.7.0+cpu, Hitlist 1.59.1, mhcseqs 2.5.12, mhcgnomes 3.41.0 and Modal 1.1.4.
`/proc/meminfo` reports 201,326,592 KiB (192 GiB); the queried cgroup-v2 files
were unavailable. No GPU was requested. Preserve the current call if a local
watcher expires; inspect its actual state before considering any retry.

The census is incomplete and no fitting or predictive evaluation has run. Copy
all available final summaries/receipts and reconcile retained SQLite counts,
source funnels and split support before scheduling the other five conditions.

The user explicitly approved the upload on 2026-09-09 ("Upload please"). All ten
files, **1,464,177,529 bytes**, were uploaded without overwriting existing objects
to workspace `iskandr`, environment `main`, volume `presto-data`, under this
family's prefix. Local hashes matched before/after transfer; remote file names
and byte counts match the manifest exactly. The census verifies content hashes
again before data construction. [Upload receipts](results/upload/) preserve the
actual launcher snapshot and its two-line dirty diff against `340e208` (a receipt
variable shadowing correction), exact command/environment and user authorization.
Nineteen focused audit/launcher checks passed after that correction. The reviewed
`340e208` CI passed 2,061 tests / 3 skipped. These were prelaunch checks.
CI at `37cbb17` also passed 2,061 tests / 3 skipped (865.08 seconds); its full
PR log is preserved in the raw root. Later-head CI remains a separate gate.

The first launch at clean `37cbb17` reached Modal app
[`ap-2NjIAHmEYvAgM1ghTB4SEz`](https://modal.com/apps/iskandr/main/ap-2NjIAHmEYvAgM1ghTB4SEz)
but image construction failed: `package directory './inference' does not exist`.
No worker call was submitted and no data loaded. The source archive's duplicated
package list omitted a declared package. [Initial receipts](results/merged_measured/)
and `artifacts/2026-09-09_1541_codex_canonical-coverage/merged-measured-launch.log`
preserve that failure. Runtime, observed census hardware, support counts and
predictive metrics are unavailable for this attempt. Dependency layers completed.

The launcher now derives archive package roots from `pyproject.toml` and isolates
explicit retries with `--attempt package_manifest`. Initial receipts and remote
logs remain unchanged; repeat attempt names are rejected. A local wheel build
from a copy of the next clean snapshot is required before retrying the same
condition/data/hardware contract. No additional input upload is needed.
The corrected launcher/instrumentation passes 28 focused tests in 13.15 seconds
and Ruff 0.16.0. An initial expanded check passed 32 tests but exposed that the
canonical packaging test discovers preserved raw source snapshots under local
`artifacts/`. Testing the isolated source copy left that checkout regression
unresolved. The review repair explicitly excludes the generated top-level
`artifacts/` directory from package discovery. Verify ordinary packaging checks
in the working checkout after preparation as well as the isolated build; the
regression test runs the actual preparation command and retains detection of
undeclared source packages, including nested packages named `artifacts`.
Both new regressions failed before the exclusion; all 24 packaging/launcher tests
now pass in 3.20 seconds, including normal discovery with existing snapshots.
Ruff 0.16.0 lint and formatting checks pass. The failure and passing logs remain
under the raw root as `packaging-review-before.log` and `packaging-review-after.log`.

The `d8f06ee` wheel build succeeded with all nine packages and all archived
production Python sources present; five canonical packaging tests passed in
2.09 seconds from the isolated copy. The [build receipt](results/verification/package_build.json)
records commands, environment, wheel hash and unchanged frozen snapshot hashes.
The Modal image also built successfully. Attempt `package_manifest`, app
[`ap-e5tn40CmWftp3a6wCcf0qy`](https://modal.com/apps/iskandr/main/ap-e5tn40CmWftp3a6wCcf0qy),
then failed during import of the SDK-relocated `/root/launch.py`: local checkout
discovery raised `IndexError` before the worker entered. The app was stopped;
[retry failure receipts](results/merged_measured/attempts/package_manifest/)
preserve its call ID, total launch elapsed time and raw log. No data was loaded.

The remote entry now explicitly uses Modal 1.1.4's serialized transport and loads
the worker from the frozen `/opt/presto` tree. The actual SDK round-trip passes
in a fresh isolated interpreter without importing the local launcher (one test,
0.48 seconds); all 28 focused checks pass in 4.29 seconds, and Ruff passes.
Attempt `serialized_entry` is now running, with the same input bytes and condition.

Earlier preparation and review history (the upload authorization above supersedes
the earlier transfer block):

- Status: observational instrumentation passed 75 focused/canonical checks in
  8.16 seconds, including unchanged full reports at chunk sizes 1/2/512, retained
  SQLite integrity, failed-gate restoration, duplicate-ID tie handling and
  exclusion of generated rows from diagnostic candidates. Ruff 0.16.0 passes.
  The Modal 1.1.4 declaration passed an offline API check with the specified
  resource tuple; shared Python is 3.12.6. No remote job has launched.
- Automatic approval review rejected the 1,464,177,529-byte upload pending
  explicit authorization for the ten files in `input_manifest.json` and the
  `iskandr` workspace's `presto-data` volume, under this experiment's prefix.
  No upload occurred. Source packaging was narrowed to executable code/config
  and the required B2M resource after local review found historical raw datasets
  in the first archive; that archive was never uploaded.
- Final verification including source-package exclusion passed 76 tests in
  7.60 seconds; the prior 75-test result is an overlapping earlier check.
- Continued author review reproduced a live-launcher mismatch that reached image
  construction despite a frozen archive, and a null sample ID converted to text
  `None`. Execution now checks its own archived bytes before any image work;
  absent IDs remain absent without changing census counts. Modal client/profile
  and the authenticated workspace are checked, and volume/app environment is
  explicitly `main`. The read-only lookup verified `iskandr`; no data transfer
  or compute was involved. Updated tests passed **85 cases in 9.85 seconds**
  (including the earlier cases), and Ruff passes. Use the new frozen snapshot;
  the earlier `0158c25` launcher intentionally fails the new self-identity check.
- Next step: collect and reconcile the live uncapped merged measured condition,
  then schedule the other source/augmentation conditions based on actual resource use.
- Open questions: actual per-column support, source-contamination impact,
  generated-parent provenance, rare assay incidence and per-column updates.
