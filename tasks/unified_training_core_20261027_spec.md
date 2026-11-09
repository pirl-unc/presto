# One trainable Presto implementation and full peptide flank context

Codex / GPT-6, 2026-10-27. Base 9dd71348124faa49a4c6a60872385c86895d50b0.

## User decisions and objective

The user requires the same model/code across every kind of new Presto training,
not merely sharing MHC-I/MHC-II within an optional research model. The user chose
one observation-manifest interface and permits removal of old training flags/APIs,
while retaining old checkpoints and frozen code for inference/reproduction.
Three-residue flank composition has not been validated as sufficient for Presto.

## Audit findings

The default CLI still selects legacy Presto and a separate trainer. Synthetic,
MHC pretraining, focused probes, sweeps and distributional/ablation runners also
instantiate independent models or loops. The paired TCR wrapper owns additional
encoders. The previous shared-core work did not consolidate these paths.
The observation-based implementation already supports identity pretraining,
affinity/stability/kinetics, binary and quantitative bulk/elution/component
observations, paired receptors, joint training, transfer/freezing and resume.
It is the canonical implementation to retain and promote, not a third backend.

## Literature and representation decision

Primary sources: Nielsen et al., NetMHCIIpan 2008 (PMC2430535), and the 2015
binding-core study (PMC4637192) use up to three flank residues summarized by
composition. This is precedent, not evidence that the summary is sufficient for
all endpoints. Experimental flank studies (PMID12207335, PMID12097376) establish
endpoint-dependent contributions. No flank-width accuracy comparison is claimed.

Replace the fixed three-residue composition features for new training with
order-aware representations from the existing shared Transformer over the FULL
peptide. Keep the intrinsic encoded nine-residue core, and pool contextual token
states separately over the complete N and C flanks for each register. Retain
flank/peptide lengths and explicit empty-flank zeros. Encode the full peptide once,
not each growing flank for every register. MHC-I uses the whole peptide with empty
flanks. MHC-II retains every contiguous nine-residue candidate and variable-length
softmax. No new source-specific encoder, core finder or trained model is added.

## Implementation plan

1. Preserve original user worktree and all frozen experiment/runtime checkouts;
   implement in this isolated checkout. Record the correction in lessons.md.
2. Promote the observation model to the public Presto implementation and keep
   ModularPresto/ModularConfig only as aliases to that SAME class/config. Move
   historical default and modular-v2 implementations under explicit legacy
   inference support, with checkpoint-version dispatch. The single current trainer
   rejects historical models, and historical checkpoints cannot silently warm-start
   a changed feature representation. Exact historical predictions remain replayable.
3. Version the new full-flank model/checkpoint feature contract. Update feature
   coverage, freezing dependencies, checkpoint/resume source pins and diagnostics.
   Report register starts/weights and the full-context contract. Preserve endpoint
   isolation and scalar assay/domain meanings; do not turn elution into a binding
   or independent processing label.
4. Replace public training CLI variants with one manifest entry point; aliases
   route to the same run_manifest -> fit_index -> train_step -> Presto chain.
   Identity-only pretraining, binding-only, MHC-I/II, bulk/MS, TCR, joint, synthetic
   software fixtures, warm-start, freezing and continuation all use this path.
   Remove legacy source/training knobs from the active CLI. Keep source curation
   helpers where useful, but remove/deactivate their alternative training loops.
5. Audit all active script launchers, remote launch commands, sweep/probe APIs,
   factories and TCR entry points. Convert supported launchers to thin manifest
   adapters. Historical architecture bakeoffs/controls remain reproducible via
   their frozen Git/source pins, not additional selectable Presto backends. Old
   incompatible flags receive explicit migration errors, never silently ignored.
   Keep historical inference/tests separate from new training acceptance.
6. Add structural and executed tests proving one model class, one optimizer-step
   implementation and one checkpoint contract across active training entry points.
   Cover mixed classes/endpoints, pretraining/transfer/resume, full-flank sequence
   order and distal gradients, variable register counts, empty flanks, context
   isolation, historical checkpoint parity and rejection of historical training.
   Source/meaning changes must fail rather than pretend to preserve old semantics.
7. Update user documentation and migration/capability map. Clearly identify any
   retired experimental convenience or unsupported historical output; no silent
   claim of equivalence. Run relevant and broad tests, repository lint, docs/build
   checks where available without installing dependencies. Verify frozen files
   and user work remain intact, retain local commit/patch and reproducible checks.

## Acceptance and limits

There is one supported trainable Presto class and one manifest training path;
all new runs call it, including pretraining and TCR. Compatibility aliases do not
own separate weights or logic. Full peptide context replaces the three-residue
cutoff for new models. Historical models are inference/reproduction compatibility,
not alternative training options. A renamed default flag or a common window helper
inside two still-trainable architectures does not meet this requirement.

Use synthetic unit/integration fixtures only during this implementation task;
no new biological fit, tuning, compute deployment or qualification is authorized
by software verification. Existing failed DO results and scientific PR milestones
remain unchanged. An eventual predictive flank/transfer comparison must be a new
registered experiment, not repeated tuning of exposed heldouts. No remote push,
release or merge in this local consolidation.

## First verification checkpoint

The initial modular suite passed 238 cases and found six expected contract-update
failures: old 43-column flank assertions, four private `_encode` return-shape
assertions, and the now-required manifest CLI error. Update these assertions to
check the new full-context/internal token contract and required-manifest error;
retain all behavioral checks and production guards. No biological tests or
scientific outcomes are being altered.

## Broad verification repair plan

Attempt 01: 2,560 passed, 14 failed, six skipped. Two tests depended on running
from the repository root; make their checked-in reference path explicit. Nine
CLI assertions still expected removed flags; retain historical helper/config
checks and assert the active manifest parser rejects those flags. Two static
script audits must inspect their new legacy locations. One checkpoint assertion
accidentally changed a historical serialized class identifier; restore the
original identifier rather than rewrite checkpoint meaning. Preserve the attempt
XML, repair these causes, then rerun focused and full checks.

Move the unused historical PCGrad update wrapper into the legacy namespace so
only the common optimizer step remains in active training code. Retain generic
loss utilities. The archived argument helper is retained only for testing the
archived script configuration; it is not imported by active training.

The first repair run was interrupted after 36 passes while `mhcseqs` parsed
unrelated grooves in the 56,035-row catalog. The two DRA existence checks now
resolve the complete 360-row DRA subset copied from that same checked-in catalog;
all relevant records and actual resolution assertions are retained. No biological
source, default pairing or resolver implementation was changed.

Repair attempt 02 passed 401 checks; the sole failure was a new test expecting
RuntimeError while the retired wrapper deliberately raises ValueError. Correct
the assertion to the actual documented rejection. No model behavior was changed.
The compatibility replay exactly matches 24 saved DO predictions and all 168
frozen files. Docs and wheel/sdist builds pass; final full-suite verification
will run after this last test correction.

Final source audit: add `data/modular_ordinal.py` to resume/source-snapshot pins,
since ordinal context affects model inputs and was missing from the prior list.
Verify that changing each promoted model/entry-point/ordinal source fingerprint
rejects continuation before any optimizer update. This is a resume-integrity fix;
model predictions and the completed broad suite's computational code are unchanged.
