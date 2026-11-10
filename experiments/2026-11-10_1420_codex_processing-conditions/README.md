# Processing-condition preservation audit

Date: 2026-11-10. Agent/model: Codex / GPT-6.
Status: metadata audit and software verification complete; scientific condition-contrast fitting remains open in #155.

Presto's shared MS adapter dropped four explicitly curated condition fields from
model inputs. This change preserves them as source-qualified categorical
`Context.interventions` through the same cellular model for MHC-I and MHC-II.
It adds no architecture branch and makes no claim of improved prediction.

## Source and counts

Hitlist commit `43e226621434a483141764ed54faba725bbb3869`; the source hashes and
sizes are in `results/summary.json`. The audited curation contains 216 study
entries and 807 arms. 63 distinct arms contain at least one of the four fields.

| Curated field | Arms | Studies | MHC-I | MHC-II | I+II |
|---|---:|---:|---:|---:|---:|
| Infection | 46 | 20 | 31 | 8 | 7 |
| Stimulation | 13 | 8 | 5 | 1 | 7 |
| Antigen exposure | 4 | 4 | 2 | 1 | 1 |
| Genetic variants | 4 | 3 | 4 | 0 | 0 |

Counts overlap between fields. They are **curated arms, not peptide counts,
independent specimens, or admitted training examples**. Two infection arms,
one stimulation arm and two variant arms have partial annotations; they remain
outside the adapter's existing fully-annotated admission gate. The full arm
inventory, source values, class, status, evidence and reference are in
`results/source_arms.csv`. No peptide/outcome tables were loaded, and no export
join or candidate-MHC qualification was attempted here.

## Contract and behavior

The prior adapter retained knockout/knockdown/overexpression, transfection,
transduction, drugs and cytokines. The new policy
`unique_arm_explicit_tokens_v2` additionally retains `genetic_variant:`,
`infection:`, `stimulation:` and `antigen_exposure:` factors. It preserves
simultaneous tokens and source evidence. Only uniquely attributed, annotated
arms with admitted evidence qualify. Unknown remains unknown; source-permitted
explicit `none` produces absence only within its condition family and supplied
study vocabulary. Hitlist does not permit `none` for antigen exposure, so the
adapter rejects it.

An exposure label does not establish a measured active pathway, machinery
activity, delivery compartment, dose or duration. These fields do not populate
`processing_pathways` or `components`. Existing frozen datasets/checkpoints do
not change. New cohorts must select and qualify the expanded condition
vocabulary through standard training preflight; old weights cannot be described
as trained on these conditions.

## Verification, limitations and next experiment

The software regressions check preservation, provenance roundtrip, ambiguous or
partial arm exclusion, unknown versus absent, shared class-I/class-II cellular
sensitivity and gradient updates in both supported elution modes, and invariant
purified IC50/MS-detector predictions. Fixtures are artificial software inputs,
not biological data or performance evidence.

The initial broad focused run on the newer experimental checkout exposed an
unrelated existing `test_staged_elution` default-config expectation: it had not
accounted for the new default-off `tcell_direct_peptide` field. Publication is
based on PR131's current head `189fe276`, where that later experimental flag is
absent. The scoped data/test files are otherwise identical between that head
and discovery base `e554e7d3`. Final verification on the publication base passes **219 tests in 5.34 seconds**
across MS admission, Hitlist admission, cellular/ordinal context, preparation,
training, staged elution and feature coverage. Repository Ruff lint and format
checks pass (516 files). `results/focused-tests.xml` preserves the test report;
`results/verification.json` verifies the original adapter drops all five new
fixture factors and the changed adapter retains them. A second audit produced
byte-identical source-arm CSV and identical summary data except elapsed time.
No concurrent experiment code is included. The full repository test suite was
not run locally for this scoped change.

No training or pretraining experiment, synthetic labels, optimizer comparison,
loss weighting, GPU job, validation/test split, prediction dump, or predictive
metric applies to this metadata-only audit. Runtime: 0.548 seconds for the
initial local CPU audit. No TEST or locked confirmation data were accessed.

Next work is [Presto #155](https://github.com/pirl-unc/presto/issues/155): count
actual joined TRAIN observations and matched contrasts, freeze a condition-aware
versus condition-masked comparison, and qualify held-out gains and affinity
preservation. Existing upstream DM/DO and arm-identity gaps remain tracked in
Hitlist #853, #860, #816, #856 and #812; no duplicate issues were filed.

## Reproduction

Run `bash reproduce/launch.sh` from this directory, or invoke it by path from
any directory. It pins the source commit and rejects dirty input files. The
initial audit used the shared Python 3.12 environment and `code/audit.py` on
base `e554e7d3` with the planned adapter/test/docs changes uncommitted. The
launcher snapshot and invocation are under `reproduce/`. Set
`PRESTO_CONDITION_AUDIT_OUTPUT` to a fresh output directory for replay; the
launcher refuses overwrite. Summary counts and CSV should replay exactly;
elapsed wall time is deliberately not expected to match.

Publication: [PR156](https://github.com/pirl-unc/presto/pull/156), stacked on
draft PR131. Code commit `55d397b5`; audit/verification commit `b87868ea`.
Issue155 body was read back and matched the prepared source. Full GitHub CI
was running at publication; local verification above remains the completed check.
