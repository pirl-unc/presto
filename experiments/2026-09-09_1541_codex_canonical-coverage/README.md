# Canonical full-source output coverage

- Agent/model: Codex / GPT-6; date: 2026-09-09.
- Status: instrumentation verified; data-upload approval pending; no condition launched.
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
- Next step: freeze the clean source archive, upload immutable inputs, then
  launch and reconcile the uncapped merged measured condition.
- Open questions: actual per-column support, source-contamination impact,
  generated-parent provenance, rare assay incidence and per-column updates.
