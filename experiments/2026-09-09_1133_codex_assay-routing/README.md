# Full-source assay routing correction

- Date/agent/model: 2026-09-09; Codex / GPT-6.
- Status: before and after source scans completed; strict payload comparison next.
- Issue: [Presto #59](https://github.com/pirl-unc/presto/issues/59).
- Specification: [assay routing](../../tasks/assay_routing_spec.md).
- Prior evidence: [source inventory and flagged-study trace](../2026-09-09_1106_codex_output-coverage/).

## Contract

Compare the canonical merged loader before/after explicit assay routing using
the identical complete `data/merged_deduped.tsv`: 3,423,737 rows, SHA-256
`46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`.
No upstream cache rebuild, study exclusions or source-file edits. Observe all
classifications and every typed record constructed before cap sampling; retain
one record per modality using the actual head sampler to bound memory. The
audit reports pre-cap counts and grouped ordered payload hashes, and reconciles
them with the loader statistics. No MHC filtering or effective training count
is inferred from this source audit.

Conditions are `before` (the unchanged main adapter) and `after` (the #59
correction). Descriptor groups retain source, record type, value type, assay type,
method, response and scalar-presence metadata. Group payload hashes include all
typed-record fields; changed descriptor routing is explained at closure and
unaffected groups must retain exactly identical payloads and order.

## Reproduction and environment

Run `bash reproduce/launch.sh before` or `bash reproduce/launch.sh after` from
this checkout. Each condition freezes its invocation, source state, production
module snapshots, launcher and dependency versions before reading data. Existing
output directories are refused; supply a fresh `--output-dir` for a repeat.

Use the pinned isolated environment at
`artifacts/2026-09-09_1106_codex_output-coverage/.venv`: Python 3.12.6, torch
2.7.0, Hitlist 1.59.1, mhcseqs 2.5.12, mhcgnomes 3.41.0. Its complete package
and biological-parser source hashes are recorded by the linked inventory;
each new invocation records its own installed versions and that receipt's hash.
Local CPU; OMP/MKL threads one; no requested or observed GPU. Runtime includes
the full source hashes before and after, classification and record construction.

Raw console logs: `artifacts/2026-09-09_1133_codex_assay-routing/`. JSON/CSV
summaries and fingerprints belong in this experiment's `results/<condition>/`.
The input remains in shared `data/`; no full duplicate is required.

## Scientific interpretation

No pretraining, optimization, generated data, model selection or predictive
evaluation occurs. No validation/test split is used, so prediction dumps and
held-out metrics are intentionally absent. Existing assay-label transformations,
loss terms and weights remain unchanged for supported measurements. The
correction must prevent unsupported structural/qualitative measurements from
being relabeled as nM affinity or MS outcomes, with explicit source-ingest counts.
This does not establish supervision adequacy (#48/#50) or model quality (#53).

## Handoff

- Status: baseline complete at clean `1c5d1eacf94a6956f2e3cbc55869ffa07e94d396`
  in 100.868 seconds; source hashes unchanged.
- Next step: implement the explicit descriptor policy in the linked specification.
- Remaining: after-condition comparison, semantic regressions, review and CI.

## Baseline results

All 3,423,737 input rows reconcile: 156,765 invalid peptides and 3,266,972
classified rows in 927 descriptor groups. The actual loader constructs 249,292
binding, 106 kinetic, 12,259 stability, 2,630,813 elution, 207,987 T-cell and
166,285 TCR-evidence records before caps; 230 classified rows lack required
quantitative labels. Every observed append count matches the loader statistic.

The full inventory expands the flagged-study findings: 1,239 `3D structure`
scalars and 6,250 `qualitative binding` scalars enter nM affinity. Four
`association constant KA` measurements enter on-rate. There are 2,073,797
explicit MS-method presentation observations; the remainder of the current
elution bucket includes qualitative binding, missing quantitative values,
structures and 1,166 presentation observations with non-MS methods. Supported
concentration families total 241,803 rows. These source counts have not undergone
MHC filtering and do not establish final supervision or predictive quality.

`results/before/` contains all groups, original numeric examples and ordered
record hashes, complete loader stats and the status receipt. The baseline's
two audit tests verify observation beyond the retained cap and restoration of
production hooks after failure. The final correction must preserve payload
hashes for unaffected groups and explain every changed route.

## Corrected source scan

The after condition ran at clean `ce30bbe9b1ea80a1f29ecefd9234bf21e496ab24`, with
the identical file hash and all 927 descriptor groups retained. Pre-cap records
are now binding 241,803; kinetics 102; stability 12,259; elution 2,073,797;
T-cell 207,987; TCR evidence 166,285. There are 564,739 explicit omissions:
561,829 unsupported qualitative-binding measurements, 1,262 structures, four
equilibrium association constants, 1,166 non-MS presentation observations,
and 478 missing supported quantitative labels. Invalid-peptide and optional-
sequence sanitization counts are unchanged. All skip reasons reconcile with the
actual loader's aggregate omissions and every pre-cap append count.

Run `bash reproduce/compare.sh` after preserving the after receipt. This checks
every descriptor population, permits only the declared route transitions,
requires zero newly invented targets, and requires identical ordered payload
hashes for every unchanged group. Raw input and production module hashes were
stable across the complete after scan. No downstream MHC filtering, split,
gradient audit or predictive evaluation has been run.
