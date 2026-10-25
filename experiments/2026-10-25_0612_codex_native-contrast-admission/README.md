# Native same-parent response: recipe admission

Registered before materializing real contrasts. Status: source recipe admission
passed; tested recipe release and biological fitting remain pending.
Agent/model: Codex / GPT-6. Local CPU only; no environment installation.

Hypothesis for the later fit: PSMD2 knockdown changes same-parent native peptide
signal ratios in a sequence-dependent way that transfers between parent dependency
components. This family first tests whether the fixed source contract supplies
enough eligible examples and whether every contrast reproduces the original data.

Data: PXD018722 Hi20S native HEK293T extracts, three numbered control and three
knockdown acquisitions. These are not established paired or independent specimens,
professional APCs, MHC eluates or calibrated concentrations. Source package 0519
and all original assets are reused by path/checksum; no repeated download.

The frozen detailed contract is `reproduce/plan.md`: canonical 8–25 residue
peptides, PEP <=0.01, no reverse/contaminant flags, only literal human nonengineered
origins at one complete parent sequence, all declared parents resolved to that
sequence, and all six raw sequence-level signals finite and positive. Each parent
needs two eligible peptides. Excluded and ambiguous records remain in the original
dependency graph and the all-peptide disposition table. Zeros are not negative
concentration labels. Modified-form support and feature identification types remain
annotations, not an unmodified-peptide claim.

Choose one reference per parent by a fixed sequence hash, without signal values.
One target per other peptide is the difference of condition means of within-run
log10 peptide/reference ratios. Preserve all six ratios and twelve contributors;
do not generate nine cross-arm pseudo-replicates. Shared multiplicative precursor
abundance and loading, and fixed peptide response, cancel under this observation
model; condition-dependent observation bias and clearance need not cancel.

Split original dependency components by the preregistered hash; minimum support
is 40/8/8 components and 80/16/16 contrasts for train/validation/test. Preserve all
outcomes if the gate fails; do not change seed, reference or eligibility. Weights
give each component equal mass, then each parent, then each contrast.

Conditions: (1) public registered recipe materialization and deterministic replay;
(2) an independently implemented eligibility/target/lineage/weight check using
original TSVs, full FASTA, the independently qualified 0424 origin list and the
0519 canonical parent claims/component IDs. Those claims/IDs already passed full
original-source reconciliation; this family does not rerun their whole graph
construction. It checks their exact retained membership and binding.

No predictor, training/pretraining, optimizer, synthetic biological data or loss
applies. Partitions and targets are created for prospective use, but no held-out
effect summary, model metric or prediction dump is produced. A 500 nM threshold
does not apply. Resource caps: 20 minutes/stage, 3 GiB process-tree RSS, 1 GiB disk
reserve plus 32 MiB startup margin. Do not launch concurrently with the full local
Hitlist suite. Numeric comparison tolerance: absolute 2e-12, relative 2e-13 for
independent log-ratio arithmetic; exact strings/IDs/sets and source quantities.

Run `reproduce/launch.sh build`, then `reproduce/launch.sh verify`. Invocation,
environment, Git state, changed Hitlist source overlay and input hashes are frozen
in `reproduce/`. Stages refuse overwrites and preserve failed attempts. Large
source/recipe packages remain under `data/`; summary and verification evidence
belong here. Both Hitlist source and recipe require their tested releases before
the separately registered biological fit. Passing admission does not establish
predictive quality or complete the APC availability goal.


## Results

The public recipe and independent original-value replay both pass. All 10,647
source peptides receive explicit decisions; 82 references and 228 contrasts remain,
with 10,337 exclusions. Exclusion reasons overlap and are not a disjoint funnel.
The retained parent sequences occupy 81 original dependency components. All-source
membership remains intact, including excluded bridges and ambiguities.

| Partition | Dependency components | Parent sequences | Contrasts | Gate |
|---|---:|---:|---:|---|
| Training | 55 | 56 | 158 | Pass |
| Validation | 8 | 8 | 38 | Pass, at component minimum |
| Test | 18 | 18 | 32 | Pass |

The verifier independently reproduces eligibility, reference/split selection,
original-value target arithmetic, all 2,736 measurement contributions and weights.
It checks all original FASTA records and 0424 origin evidence; canonical parent
claims and dependency construction are reused from the separately verified 0519
source contract. All modified-form and feature-type annotations reproduce the
original TSVs. No predictor was fitted, no held-out target-effect distribution was
summarized, and no predictive superiority or concentration calibration is claimed.

| Stage | Result | Seconds | Peak observed process-tree RSS |
|---|---|---:|---:|
| Public materialization and deterministic replay | Pass | 29.36 | 2,483.3 MiB |
| Independent original-value replay | Pass | 4.04 | 1,866.1 MiB |
| Full default local Hitlist suite | 3,544 pass | 184.61 pytest / 189.59 wall | 816.4 MiB process peak, not summed workers |

Format/lint and 36 focused tests also pass. An earlier full-suite invocation
stopped before pytest at the unchanged memory guard (2.33 GB free/speculative
versus 2.5 GB required). It is preserved separately. After capacity recovered,
the same guard admitted two workers at 5.01 GB; no threshold was reduced. The
subsequent real-source stages ran after the suite ended. All source-admission
resource caps passed; no failed real-data stage occurred.

Takeaway: the strict cohort supplies enough support for the prospectively planned
zero/length/sequence comparison, with only eight validation components. The
uncertainty and conditional-positive-observation limits matter. Finish the source
and recipe tested releases before the separate fitting registration. Exact versus
RL comparisons remain contingent on demonstrating useful biological signal under
the frozen criteria, and the full APC availability goal remains open.

Artifacts: `results/package-manifest.json`, `support.json`,
`independent-verification.json`, runtime/process receipts, logs and frozen source.
The large recipe package stays once at the `package` path in `reproduce/launch.json`;
its original input snapshots use independent copy-on-write files. No large raw
package or original data is added to Git. Git base/dirty states and the six-file
Hitlist executable overlay are explicit in the launch contract.


## Current-main publication verification

Concurrent Hitlist PR840 changed shared catalog/MAPP context code after admission.
The separately registered `reproduce/current-main-amendment.md` preserves this
sequence. Recipe [PR841](https://github.com/pirl-unc/hitlist/pull/841), final code
`90a7a62a5f1e052d87e797d86bac6d59cf3b0700`, now includes that main base.
The six executed recipe files are unchanged. The current tree passes format/lint
and all3,547default tests (179.98s pytest;184.42s wall). The public verifier accepts
the exact existing package unchanged in5.06s/1,702.9MiB observed process-tree RSS;
no target was recomputed for selection and no criterion changed. This checks
current-code compatibility in addition to the original independent replay.

Source release40167325306 succeeded and its exact artifacts verified against
clean e755be4 and the trusted Actions artifact. They remain unuploaded because
current main advanced at the same unreleased2.0.37version. New source release
40168710669 tests current maind2e0cb6. Recipe Tests40168839728 is running;
Docs40168839820 passes. Tested publication remains a prerequisite for fitting.
