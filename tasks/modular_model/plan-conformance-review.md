# PR #131 against frozen issue #130

Reviewed 2026-10-20. PR head: `6f935093448ff89ccdbf56d5aef321bbb8147c68`;
last Python change: `77d9856b15abd25fa1e6f963585cbc42e446c81e`.
Authority: [frozen specification](issue130.md), not the narrower implementation
handoff or the PR's own completion claims.

This is the preserved audit of the head above. Subsequent fixes, quantitative
extensions and the completed assay-only 32-row fitting diagnostic are recorded
in [completion progress](completion-progress.md); the gaps below describe the
reviewed historical code, not the current head. The full objective remains open.

## Verdict

The neural architecture follows the plan closely. The complete data-to-trained-
model contract does not. This is a substantial experimental software foundation,
with milestone A still partial and no completed real-data milestone B–E.
There is no new evidence that the nearly constant prediction problem is fixed.
The earlier wording "first software milestone delivered" overstated completeness.

Three review-only synthetic software checks reproduced missing guarantees. These
are counterexamples to software claims, not predictive experiments or biological
validation. No model implementation was changed during this review.

## Material findings

1. **High: the training guard can miss unused feature columns.**
   `train_step` treats each named tensor as trained when its aggregate gradient
   and update norms are nonzero ([training/modular.py](../../training/modular.py#L138)).
   `predict` checks those tensor names ([models/modular.py](../../models/modular.py#L651)).
   There is no corresponding admission/coverage check for every configured input
   slot or categorical state. A software fixture declared an acquisition feature
   missing from every training row. Preflight accepted it, its value/mask weight
   columns received zero gradients, the layer reported trained, and prediction
   accepted a newly supplied value. The plan requires unsupported feature effects
   to be omitted or exposed as unsupported, and unused entries to be identified
   explicitly (specification lines 266–269 and 907–927).

2. **High: the specified coverage closure gate is absent.**
   The runner writes `untrained_selected_parameters` and returns normally
   ([training/modular_run.py](../../training/modular_run.py#L384)). A bounded
   two-head software fixture selected a checkpoint after updating only one head;
   the other head remained listed as untrained. The output is correctly marked
   `not_qualified`, and ordinary inference rejects an entirely untouched requested
   head. Those safeguards help, but the required failed/incomplete closure state
   and justification policy are not implemented. Scientific promotion gates are
   also absent: no executable memorization, baseline, contraction or noninferiority
   decision is applied to a run.

3. **High: canonical source preparation is only partially implemented.**
   The new path starts from caller-curated JSONL, resolved pairs and frozen groups.
   It validates hashes, IDs and exact peptide/group split separation, but does not
   build a qualified cohort from the available source export
   ([data/modular_sources.py](../../data/modular_sources.py#L182)). There is no
   integrated source-lineage deduplication, contradiction report, raw/admitted/
   excluded/deferred census, or study/donor/receptor-cluster audit. A software
   fixture counted the same lineage/readout twice when its export IDs differed.
   Invalid rows abort rather than populate the planned exclusion report.
   Source policy strings are declarations, not automated evidence qualification.
   The PR does not establish that the newest Hitlist export is being used.

4. **Medium: important supported-data types still have no executable route.**
   MS, digest and cellular-response endpoints are binary only
   ([data/modular.py](../../data/modular.py#L30)). Quantitative presentation,
   intensity and direct digest-yield outcomes, including their censor limits,
   cannot be trained through this interface. These matter to the requested
   matched APC comparisons. Enzyme and source flanks exist; missed-cleavage rules,
   richer preparation conditions and reaction/stimulus features are not modeled.
   Quantities are rejected rather than silently binarized, which is correct, but
   their absence is real implementation work rather than merely missing data.

5. **Medium: support and subset continuation are narrower than the plan.**
   Checkpoint initialization supports compatible task supersets, not general
   supported subsetting/composition; previous tasks cannot be removed and inherited
   scalers must match ([training/modular.py](../../training/modular.py#L259),
   [training/modular_run.py](../../training/modular_run.py#L239)). Frozen-submodule
   training and exact resume are absent. `pretrained_frozen` is an accepted status,
   not a completed freezing workflow. Evaluation does not promote outputs to a
   validated status/domain. The Python API serves trained-but-unqualified results
   with visible metadata; there is no qualified-serving release gate.

6. **Medium: the planned debugging/evaluation evidence is incomplete.**
   Losses, basic correlations, RMSE, prediction spread, threshold metrics and
   class-specific exact regression are implemented. Missing: endpoint loss shares,
   task-specific shared-encoder gradient shares, per-allele/study analysis, identity
   calibration and supplied-versus-inferred comparisons, uncertainty/bootstrap
   comparisons, and automatic sequence-contrast checks. Register weights/entropy
   exist in forward diagnostics but are not retained in evaluation prediction
   dumps. Those dumps also omit the promised allele identities and residual field
   ([training/modular_evaluation.py](../../training/modular_evaluation.py#L101)).

## Requirement/status matrix

| Planned capability | PR status | What remains |
| --- | --- | --- |
| Small shared sequence encoder and direct assay heads | Implemented at the specified dimensions; no kinetic-factor/probe-mixture path | Real-data fitting and endpoint qualification |
| Whole-peptide class I; sliding nine-residue class II with PFRs | Implemented, including within-class pooling and singleton pruning | Real class-II fitting/held-out evidence; exported register diagnostics |
| Identity first, rich representations reused downstream | Class/MHC-species/TCR-species/lineage mechanisms and assay-gradient stop implemented | Curated identity catalog, actual pretraining, calibration/OOD and predicted-identity evaluation |
| Visible species defaults and independent overrides | Implemented; strict defaults-off policy available | Qualified reference data; no optional peptide-origin model |
| APC species reference machinery and concurrent losses | Resolver, caller-supplied profiles, state/activity fields and output masks implemented | Curated sourced/versioned profile catalog and cell-state refinements; state-level support; attributed control/defect/rescue cohorts |
| Purified-binding independence from APC state | Implemented and tested | No claim of learned cellular effects follows from this invariance |
| Bulk digestion and elution share one MS response component | Binary architecture and one-loss/bag semantics implemented | Quantitative endpoints, anchor data, transfer/retention tests, prospective feature admission and richer digest context |
| Full paired TCR–pMHC scorer and explicit candidate ranking | Five-sequence path, receptor conditioning before register pooling, functional/compatibility distinction implemented | Authorized TCRvdb schema/QC adapter and data; cluster/study leakage checks; trained scorer and retrieval evaluation |
| Coherent observations, units, censoring and missing labels | Implemented for the declared endpoint set; direct/proxy affinity separated | Complete source preparation and dispositions; missing outcome types |
| All trainable parts learn; unsupported effects stay unavailable | Endpoint allocation, optimizer membership and tensor-level gradient/update logging implemented | Feature/state-level evidence and enforceable closure/promotion gates |
| Work across data subsets | Identity-only, binding, bulk, elution and receptor routes share the new schema/loss/model | Broader checkpoint subsetting, frozen components, source selection and raw adapters |
| One public training/prediction workflow | Both training CLIs dispatch to one opt-in modular manifest runner; Python predictor exists | Existing default remains legacy; normal prediction CLI is not switched; new trainer/evaluator are separate implementations |
| Reproducible runs and evaluation | Source hashes, launch/source snapshots, selected/initial/last checkpoints, optimizer and prediction artifacts | Exact resume, richer source receipts and all prescribed comparison/qualification artifacts |

## Deferred science versus implementation omissions

The following experiments have **not run**: identity qualification, the real
32-example fit, exact-only versus censored IC50 conditions, matched mean/ridge
baselines, three-seed held-out evaluation, label permutation, added-head
noninferiority, matched APC comparisons, MS anchors/transfer, and TCR retrieval.
No qualified serving checkpoint is supplied. The existing valid training subset
can support the first fitting diagnostic; that work need not wait for TCRvdb or
the latest Hitlist export.

The TCRvdb author file remains an external data-access dependency. By contrast,
coverage gates, source preparation/dispositions, quantitative endpoints, reference
profiles and checkpoint continuation are code/data-curation work still owed by
the broader plan. These should not all be described as "scientific qualification."

Canonical-only peptides, no short/noncanonical class-II cores, and no speculative
peptide-origin classifier are intentional initial-spec limits. They are not
accidental deviations. Unsupported PTMs are also documented, but full peptidoform
support remains outside this first implementation.

## Recommended completion order

1. Close feature-support and run-closure gaps; implement the concrete preparation
   and lineage/disposition path for the initial identity plus IC50 subset.
2. Qualify identity pretraining and pass the registered 32-example gate. Stop and
   debug if it fails; do not treat nonconstant random outputs as progress.
3. Run the prescribed full IC50 conditions, matched baselines, class-specific
   evaluation and supplied/predicted-identity checks. Qualify a minimal checkpoint.
4. Qualify matched APC cohorts alongside that work. Add the endpoint actually
   measured, including a quantitative route when needed, then compare against
   identical context-withheld data/budgets. Add MS anchor/transfer and TCR stages
   with their own evidence gates and retained-task checks.

## Verification of this review

- Exact published-head GitHub PR and push lint/test and docs checks all succeeded.
- Previously recorded full regression: 2,418 passed, four skipped; final focused
  suite: 49 passed. These establish software regressions, not biological quality.
- Three review-only counterexample checks passed in 2.51 seconds. They confirm
  the gaps described above; they are deliberately not new product regression tests.
  Script: `/private/tmp/presto-plan-audit-checks/test_plan_audit.py`.
  Invocation used the existing shared Python with
  `PYTHONPATH=/private/tmp/presto-modular-imports`, OMP/OPENBLAS/MKL threads set to 1,
  and `python -m pytest -q -c /dev/null` with root/confcutdir restricted to that
  temporary check directory. Initial collection from `/private/tmp` hit an
  unrelated protected daemon directory; isolated collection then completed.
- No full-suite rerun was needed for this documentation-only audit. No model code,
  real training cohort, shared environment or original dirty checkout was changed.
