# Improve MHC-II prediction and pathway/cell-condition representation

User objective, 2026-10-26: improve MHC-II binding, MS-detected ligand prediction,
and distinct antigen-processing pathways/cell-condition representation; inspect
Hitlist data and file issues for confirmed under-annotation or missing evidence.
This is an active scientific goal, not a documentation-only or schema-only task.

## Full completion requirements

1. **Binding:** demonstrate reproducible MHC-II improvement over the frozen current
   modular model and simple controls on matched data/compute. Report exact and
   bounded affinity, ranking/discrimination, calibration, binder sensitivity and
   specificity, DR/DP/DQ and source/allele support. Keep the class-I retained-task
   comparison explicit. Existing inspected validation/test partitions are
   development evidence, not newly independent evaluation.
2. **Detected ligands:** fit/evaluate an actual MHC-II MS ligand endpoint with
   observed source labels, complete alpha/beta candidate sets and acquisition/
   sample context. Separate detection/ranking from affinity and free-peptide
   amounts. Preserve multi-allele ambiguity and missingness. No silent conversion
   of unobserved peptides into measured negatives. If a contrast set is generated,
   declare its distribution and limited estimand; independently evaluate genuine
   sample/study transfer and compare binding-only and sequence/context controls.
3. **Pathways/conditions:** implement and fit distinct, composable processing routes
   and cell-condition inputs where source contrasts support them. Endosomal/
   lysosomal processing, endogenous delivery/autophagy and other observed routes
   must not collapse to an MHC-class flag or a cell-line ID. Retain cell type,
   cytokines, machinery components, delivery, sample preparation and MS acquisition
   as separate concepts. Concurrent or unknown routes/states stay explicit.
   Preserve purified-binding invariance to irrelevant cellular fields and audit
   context/route ablations and held-out condition generalization.
4. **Hitlist:** inspect current canonical sources/exports, qualification recipes and
   raw source evidence. Quantify class-II binding, MS, pairing, parent/flank and
   pathway/condition coverage. File focused, reproducible issues for confirmed
   gaps not already owned, or add new evidence to existing owners when appropriate.
   Requests must identify actual affected source rows/materials and acceptance
   criteria; guessed biological labels are not curation.
5. **Verification/delivery:** preserve per-example predictions, source joins,
   checkpoints/coverage, resource/runtime receipts and independent metrics in
   registered experiment families. Run relevant software tests and final-head CI,
   publish reviewable changes and keep scientific failures visible. No complete
   claim until all three predictive/representation tracks have adequate evidence.

## Isolation and current work

Use `/private/tmp/presto-mhc2-20261026/presto`, a separate clone of published
PR131 headf86fce7046a884e37c2b6e96a660121175a59285. Preserve original dirty root and
modular worktree, shared environment, and the older immutable IC50/species runners.
The old IC50 trainer and both controllers were confirmed live through tool sessions
32202/12902/29950 at goal start. Do not restart them from a stale file or timeout.
New scientific work uses local CPU under the user's existing choice; no package
install or cloud launch is authorized by this plan. Serial resource-intensive fits
must respect the running job's allocation. Reading/auditing metadata is independent.

## Initial sequence and decision gates

- Verify current remote/local heads and live runs; inspect prior negative/positive
  evidence. Record the current class-II baseline without tuning its running recipe.
- Read the canonical experiment workflow and relevant primary literature, then
  inspect present model/trainer and Hitlist schema/source support. Distinguish
  earlier issue reports from current defects by reproducing against pinned bytes.
- Register a bounded class-II data/context audit before measuring coverage. Start
  with existing frozen corpora for matched baselines, then compare available newer
  canonical exports/specialized sources. Preserve source/library version separation.
- Use the audit to prioritize actual binding-data/representation fixes and a real
  MS cohort. Write experiment-specific recipes, matched controls, resource budgets,
  selection/test policies and quality/noninferiority thresholds before training.
- Inspect original Methods/metadata for source-backed route/condition distinctions.
  Design typed representation and endpoint masks from those facts; test failures
  before fixes. Do not add dormant heads or claim learned pathways from feature
  allocation or nonzero parameter changes alone.
- File confirmed Hitlist annotation/source issues with evidence links and no
  duplicate tickets. User explicitly authorizes issue filing; no author outreach,
  private-data access request or Slack/email message is implied.
- Train/evaluate, independently verify, analyze failures and iterate in new frozen
  families. Do not stop at inventory, tests, queued jobs or a single aggregate score.

## Scientific and execution invariants

Preserve exact raw assays/units/bounds and actual lineage; never equate KD proxies
with direct affinity, elution with production, search enzyme with preparation,
protein abundance with measured peptide amounts, or unknown condition with WT.
Keep supplied/override context separate from predictions. Parent/peptide/source/
aliquot dependencies govern leakage controls; nearby class-II registers are not
independent observations. Candidate MHC pairing requires explicit provenance.
Annotation priors are marked assumptions and cannot replace actual intervention
contrasts. Primary-data checks precede source claims. Existing closed holdouts stay
closed to tuning; model selection and calibration use their declared development
splits only. Comparative experiments conserve exposure/update budgets and report
extra compute. Preserve all failed attempts and deterministic reproduction paths.

## Progress classification and completion audit

This is the first turn of the new goal. Earlier conversation supplied status and
historical metrics, not completion of this objective. Live-session polling at
start is a verified wait for the existing runs; new authoritative evidence,
implemented changes and registered results constitute progress for this goal.
Keep a requirement-by-requirement evidence table with passed/failed/incomplete
states. A missing source for one route does not stop independent binding/MS work.
The goal remains active until the actual requested end state is demonstrated.
