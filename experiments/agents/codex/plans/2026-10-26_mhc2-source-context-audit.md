# MHC-II source/context audit — prospective contract

Agent/model: Codex / GPT-6; 2026-10-26. Parent objective and full scientific gates:
`tasks/mhc2_improvement_spec.md`. This metadata audit informs actual binding/MS/
route comparisons; it does not itself establish improved prediction.

## Questions and inputs

1. Count literal class-II binding and MS observations, distinct peptides and source
   studies in the pinned public v4 export. Preserve reported/corrected class,
   assay family, bound/unit, modifications and source attribution separately.
2. Describe supplied molecule restrictions and sample candidate sets without
   claiming that an allele string is a verified complete alpha/beta pair. Count
   parent/flank and cellular-condition field coverage; missing is unknown.
3. Inspect present canonical Hitlist curation for two useful class-II studies,
   Abelin2019/MAPTAC and Stražar2023, then add one intervention study only if its
   source evidence supports a distinct processing/condition contrast.
4. Compare known gaps to live GitHub owners before filing issues. Current issue
   state and source-version differences must be explicit. No duplicate ticket for
   the already-open Nanaware2024 admission or engineering-drift work.

Start with the existing read-only public bundle at
`/Users/iskander/.hitlist/corrected-corpus/public-downloads/v4/bundles/2026-10-16T07-20-43Z`.
Record size/hash/schema for each consumed file; do not claim it was regenerated
with today's software. Current Hitlist main was read from GitHub as
`13b91424295e01286db46e8e9818fcc2fe816834` (2026-10-26). Pin a separate read-only
source snapshot if needed; preserve the older dirty user checkout and shared env.

## Limits and acquisition

Local audit: one CPU/BLAS thread; 5 minutes and 3 GiB RSS per invocation; at most
1 GiB of existing export bytes, streamed in bounded batches. At most 100 MiB of
new source/metadata downloads. No raw MS acquisition download, package install,
cloud compute, training, generated labels, or predictive model selection.

Primary literature budget after this registration: eight tool/API operations,
counting queries, opens and HTTP fetches across interfaces, not eight per tool.
At most four primary papers, prioritizing the two named source studies. Record
each request/result, failed operations and any prospective budget amendment.
GitHub repository/issue/release metadata discovery is a separate non-literature
activity: retain the returned current states and source references. Initial work
already made two issue searches, three issue fetches and one current-head/release
inventory call; no primary-paper search had been made before registration.

## Outputs, verification and stop conditions

Create `experiments/2026-10-26_1803_codex_mhc2-source-context-audit/` with launch
contract, source snapshot, input receipt, summary JSON/tables and source-field
examples (identifiers only; do not export participant-level sensitive metadata).
Independently recheck class/assay totals using an alternative Arrow aggregation
and compare a deterministic selection of raw metadata rows. No train/validation/
test split or prediction dump applies: this is a source inventory, not an eval.
Do not silently use coverage as training eligibility or infer causal routes from
cell-line, class, gene expression or sample names. Distinguish explicit concurrent
factors from inferred biological interpretation.

If data exceed limits, stop and prospectively amend scope with an explicit reason.
If current data already address a gap, close that candidate without filing it.
Confirmed gaps need pinned rows/materials, exact source evidence, scope/unknown
semantics and testable acceptance criteria. Complete README and canonical registry
with useful/insufficient cohorts, issue links and the next experiment decision.

## Reporting correction before interpretation

The first inventory completed in 88.42 seconds with matching independent class/
assay totals. Inspecting its categorical output found a serializer defect:
`records()` treated scalar string keys as iterables and retained only their first
character. This affects `class_ii_categories` labels only; it does not affect tuple
assay keys or coverage counts. Preserve the original script/output and mark those
category tables invalid. A separate verification/correction command will reconstruct
all four categorical distributions directly with Arrow `value_counts`, independently
verify field-coverage and distinct-peptide/PMID totals, and write a distinct corrected
inventory. Do not use the damaged category labels in conclusions or upstream issues.
