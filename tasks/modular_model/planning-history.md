# Consolidated simple-model issue — 2026-10-20

## Authorized deliverable

File one substantial Presto GitHub issue summarizing the audited state and a
detailed implementation/qualification plan. The model specification itself
must remain small. This task creates a planning issue, not a new model, training
run, dataset refresh, or change to existing issue status.

## Plan and decision

- [x] Read the audit and existing model-contract/coverage/underfitting issues.
- [x] Draft an issue with pinned evidence, one minimal architecture, explicit
      source-to-endpoint routing, all-module training checks and staged gates.
- [x] Review counts, losses, target units, missingness, data availability,
      held-out policy, and consistency with existing sequence/context boundaries.
- [x] Create the issue, read it back, verify the body, and record its URL.

Original publication choice (superseded below): independently encode peptide and the two resolved
MHC groove segments with one shared small sequence encoder; masked pooling,
concatenation plus known class, one fusion MLP and direct endpoint heads.
Start with IC50 alone. Admit additional measured endpoints through explicit
data-supported head construction, not a biological latent DAG. Keep MS/TCR
extensions separately gated by suitable data and their own measured outcomes.

Acceptance distinguishes a training memorization check from validation
improvement and from external generalization. Output variance alone is not
success. Any reduction of the broader #46 output inventory is an explicit
initial product scope, not a claim that those biological questions are answered.
Keep historical checkpoints and target contracts immutable. No experiment is
launched by filing this issue.

## Initial review and publication

Filed https://github.com/pirl-unc/presto/issues/130. The issue is open and its
published body matches `tasks/parsimonious_model_issue.md` exactly after newline
normalization. The receipt records its SHA-256. Training counts reconcile to
7,241, the fusion width is 386, censor-direction examples pass, and the body
contains no automatic issue-closing keywords. The approximately 3,600-word plan
preserves the historical audit, separates fitting from qualification and makes
MS/TCR extensions conditional on adequate measured evidence. No model code,
training job, existing issue status or historical artifact was changed.

## Correction: explicit MHC-II registers — 2026-10-20

The user correctly identified that removing explicit core handling left the
claimed MHC-II support underspecified. A general sequence model could learn
register dependence implicitly, but that is not evidence that this pooled
baseline does so. Simplify the latent graph while preserving this binding
problem in the first model.

Reviewed the original proposal, current implementation and primary NN-align
literature, and read the live issue before edits. The initial correction plan
retained three external sequences, explicit class-II cores, PFR features and
one whole-peptide loss. Its provisional supplied-class/parameter-free reduction
was superseded by the user's subsequent cascade/TCR requirements below before
publication.

No production model changes, training or new experiments are authorized by this
correction. Preserve the historical audit and the prior publication receipt hash.

## Superseding clarification: identity/context cascade and perturbation data

The user subsequently requested: class/species prediction first; explicit
overrides and default species cascades; APC-species machinery defaults with
component-level loss/KO overrides; a TCR-to-pMHC use case; and priority for cell
lines with known processing defects. These requirements supersede the pending
known-class-only and parameter-free-register-reduction choices above.

Revised plan:

- [x] Specify sequence-only shared encoding followed by small supervised
      identity heads. Keep raw predictions separate from resolved context.
- [x] Define an acyclic, provenance-preserving default/override resolver.
      User-authorized cross-role defaults are assumptions, never source facts
      or auxiliary training labels; antigen origin and selection MHC are not
      copied from the presenting molecule.
- [x] Add explicit class-I full-peptide/class-II sliding-register candidates,
      local PFR composition/length features, shared small fusion and one
      learned register pooling operation supervised by the peptide-level loss.
      Preserve candidates for contextual/TCR outputs instead of selecting an
      irreversible single register before the receptor is considered.
- [x] Scope molecular, cellular and TCR inputs by endpoint. Represent
      simultaneous per-component machinery states and loss/unknown explicitly.
      Use a single optional context fusion, rather than recreating a mechanistic
      graph with unsupervised scalar outputs.
- [x] Make matched perturbation/control cell-line cohorts the first cellular
      data milestone. Preserve quantitative outcomes, sample attribution,
      parental/clonal links, detection limits, source-protein context and
      biological exposure route; verify current Hitlist fields before new asks.
- [x] Specify five-sequence TCR/pMHC candidate scoring, class-II register
      handling and optional cellular conditioning. TCR-only queries rank a
      supplied candidate library; they do not uniquely generate an epitope.
- [x] Rewrite loss/training stages, head allocation and acceptance checks to
      match the new spec. Identity auxiliaries require actual varied identity
      labels; default machinery values provide no training evidence.
- [x] Review the entire issue for obsolete claims; publish and verify readback,
      recording the prior and updated body hashes.

The implementation remains deferred. No wet-lab procedure, new fit, public
accuracy claim, shared-environment change or mutation of historical artifacts.

### Revision verification

Updated issue #130 and verified complete published-body equality after newline
normalization. No intervening remote changes were overwritten. The receipt
preserves the prior body hash and records the new hash
`c89987b8cc4cd771c3ad721411c5549ab633be4fbd170c1029c3f588948049bd`.
Reviewed the 6,099-word issue for obsolete supplied-class-only/core-removal
claims; checked fusion dimensions (`429 + 2*S`), unchanged historical counts,
Markdown fences, GitHub size limit and absence of automatic closing references.
This was a planning-document revision, so no production-code test run or
experiment was needed or claimed.

## Final holistic revision and implementation authorization

The user now requires explicit non-MHC MS supervision for digestion/detectability,
a coherent architecture across all available sources and their subsets, and
implementation after the issue is precise and consistent. Preserve the simple
shared sequence backbone, explicit class-II registers, role-aware defaults,
component-level APC state and optional receptor branch.

- [ ] Specify separate preparation/digestion, cellular pMHC availability and
      shared MS-observation components, with one observed-outcome loss and
      explicit identifiability/support limits.
- [ ] Add a source/label/module matrix and subset-derived module allocation;
      permit identity-only, bulk-only, binding-only, cellular and receptor runs.
- [ ] Specify auxiliary TCR species/coreceptor-association supervision,
      representation reuse and measured coreceptor-state overrides.
- [ ] Freeze exact module interfaces, endpoint routing, target transforms,
      losses, missingness, calibration and checkpoint/support contracts.
- [ ] Review and publish the issue; verify body equality and preserve revision
      history. Write a self-contained implementation handoff.
- [ ] Isolate implementation from the existing PR and original dirty checkout.
- [ ] Implement the shared architecture, typed preparation/context resolver,
      subset-aware supervision/checkpoints and canonical trainer integration.
- [ ] Verify each subset, module gradients and meaningful invariance/routing;
      run relevant regression checks and publish a reviewable implementation.

Real-data qualification remains a separately registered, bounded experiment.
Do not turn a software smoke check into a claim of learned biological accuracy.
The author-controlled TCRvdb file is still unavailable; implement its normalized
contract without inventing that dataset or its exact raw schema.
