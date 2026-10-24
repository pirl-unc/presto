# Canonical admission of native intracellular peptidomics perturbation data

## Why this is next

The cellular amplitude diagnostic0106 enforces its stop rule: imported digestion
sequence priors do not beat calibrated uniform cleavage. Stop tuning that bridge
on the same MAPP development parents. A modest geometry-based captured-signal
control does not validate a cellular peptide pool. The next work needs cellular
measurements with explicit processing/turnover perturbations and source lineage.

Hitlist issue788 already owns the native TMAB peptidomics source request. It was
read back open with no comments on2026-10-24; inspect current upstream state before
starting rather than assuming it stays unimplemented. This plan is not a second
source curator or a Presto-only training adapter.

## Phase 1: source retrieval and scientific admission audit

Create a new canonical Presto experiment directory before source-count or schema
qualification arithmetic. Record agent/model, current commits, dirty state,
commands/environment, original/public asset URLs, retrieval timestamps/status,
checksums, sizes, rights evidence and unavailable assets. Store original large
workbooks in shared data/artifact storage with explicit immutable references;
keep summaries and reproducibility code in the registered family.

Primary sources identified in788:
- Dasgupta et al.2014, DOI10.1371/journal.pone.0103604, especially S1/S2.
- Gelman et al.2013, DOI10.1371/journal.pone.0053263, especially S1/S2.
- Follow their cited original epoxomicin measurements to reconcile reused data;
  PMC3315381 is a discovery lead, not yet an audited asset contract.

Read full primary Methods, supplement legends and original tables. Do not treat
issue prose, plotting examples or article aggregate counts as verified workbook
schema. Public sources only; no author outreach or access request. An inaccessible
asset remains an explicit limitation, not permission to substitute inferred values.

Inventory every sheet, original row/cell, formula/cached value, identifier,
quantity/qualifier/unit, annotation and disposition. Independently cross-check the
workbook extraction with its actual container format (XML or BIFF as applicable),
not two reads through the same parsing function. Distinguish identified peptide
sequences from unidentified peaks and analytical labels from biological PTMs.

Audit these source-backed dimensions before any modeling:
- Exact study, cell line/species, material/extraction scope, harvest time,
  perturbation compound/vehicle/dose/duration, and any separately measured activity.
- Biological versus technical replicate, sample/aliquot/isotope channel/run,
  mixing design, label swaps, shared denominator/control and instrument/protocol.
- Exact ratios, source bounds, reporting/plotting caps, missing/undetected channels,
  overlapping peaks and invalid values. A threshold-looking number alone is not
  sufficient evidence for censoring; missing is not zero or a negative label.
- Charge states, multiple observations of one peptide form, source protein claims,
  sequence alternatives and unresolved source/reference versions.
- Original versus reused or reprocessed observations across publications; preserve
  conflicting copies and the author-described lineage before any split or pooling.

Chemical exposure remains chemical exposure. Do not infer gene KO, complete
machinery loss, catalytic subunit specificity or exact residual activity from a
drug name. Native extract measurements do not establish free cytosol, professional
APC identity, precursor abundance or absolute concentration.

Close the audit with original-value replay, homogeneous metadata mapping proposal,
coverage/disposition tables, explicit conflicts and a shared ingestion contract.
No training or predictive metrics in this phase; explain why no train/test split
exists. No source-value repairs driven by model fit. Update788 with concrete new
findings, not a duplicate issue or repetitive modeling-status comment.

## Phase 2: shared Hitlist implementation, after evidence qualification

Inspect current Hitlist main/release and existing source catalog/API before editing.
Local checkout inspected at8a10b5a is only a locator, not a frozen future base. Work
on an isolated branch/worktree under/private/tmp; preserve unrelated local work.
Read applicable AGENTS.md and lessons. Follow its version, formatting, lint and
full-test requirements; do not change the shared Python environment.

Use the existing EvidenceStore, source cells, sample/protocol, quantities,
contributors/derivations, source assertions and catalog/recipe interfaces. Preserve
all source rows and excluded/ambiguous dispositions. Add no competing metadata
registry. A general protocol or censoring limitation needs a concrete upstream
issue with acceptance criteria; check existing owners first.

Keep source ingestion separate from an eligible training recipe. A candidate
recipe may expose source-backed native-peptidome log-ratio contrasts, with exact
and censored observations distinct, peptide-form ambiguity handled explicitly,
shared-control/charge/reused-measurement dependence preserved and available parent
mappings recorded. The actual endpoint and counts must follow audited evidence,
not this prospective guess. No invented loss coefficients, source joins or model
features to paper over missing biology.

## Phase 3: next biological comparison, separately preregistered

Only after canonical source/recipe qualification, define a new bounded local
exact-versus-sampled comparison on the supported endpoint. Freeze target units,
qualifier loss, split/dependence rules, precursor/capture assumptions, context
features, baselines, update budget, hardware/cost limits and verification before
launch. Preserve unseen source/parent groups where actually supported; do not call
cross-paper reused observations independent studies. Record all allocated versus
frozen/untrained model components.

Use the source's real quantitative endpoint and meaningful no-change/common-shift
controls; avoid another synthetic benchmark standing in for biological skill.
Predict available peptide fragments including short/nonbinding products, with
pMHC selection downstream. A drug-response ratio fit still does not establish
absolute intracellular concentrations or the full APC machinery-conditioned goal.
Continue toward those requirements without narrowing success to admission alone.
