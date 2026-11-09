# Shared Presto core audit and integration

Codex / GPT-6, 2026-10-27. User requires MHC-II work to develop MHC-I and general
Presto capabilities through a common core, not task-specific sidecars.

## Findings before implementation

The completed DO:DM native family1250 uses immutable c0358852's canonical
Observation/MHCInput/QuantitativeDomain, ModularPresto, fit_index, checkpoint and
prediction path. It adds no DO-specific model or trainer. One encoder, pMHC
fusion, cellular context fusion and MS-response feature block serve both classes.
Class-specific register candidates remain appropriate biological structure:
whole-peptide class I and nine-residue class-II registers. Source-scoped ordinal
context and quantitative measurement domains are generic. Direct ridge/control
models are experimental comparators, not production capabilities.

The actual integration gap is input provenance: HumanMHCCatalog.resolve currently
labels every selected catalog restriction as reported and has no explicit path
for an authorized reference assumption. The DO experiment therefore implemented
catalog selection and assumption metadata locally. Put this behavior in the
existing shared catalog adapter, supporting both classes, rather than adding a
DO-specific adapter or model. Avoid duplicating the Observation constructor with
an unnecessary generic wrapper.

## Planned change

1. Preserve completed experiment snapshots and create an isolated clean checkout
   from c0358852. Do not modify the user worktree or any frozen scientific runtime.
2. Extend the existing catalog resolution API with an explicit optional reference
   assumption contract. Keep the current strict default and missing-partner errors.
   Caller must state selected reference restriction, actual source-reported
   restriction (possibly partial/unknown), and a nonempty rationale/reference.
   Resolve selected sequences through the same catalog validator. Validate source
   known alleles are not silently replaced. Preserve source identity separately
   from modeled alleles and mark assumptions in provenance. Support class I and II;
   no automatic DRA default, inferred genotype, source mutation or extra model.
3. Add meaningful cross-class software checks: source-vs-assumed identity and
   default strictness; one mixed-class native quantity model/trainer/checkpoint;
   shared encoder/pMHC/context/MS gradients from both classes; cellular context
   must not leak into purified binding or the shared detector. Reuse existing
   canonical test helpers where possible. All labels are synthetic software
   fixtures, not biological validation or a new predictive experiment.
4. Replay the completed DO molecular input through the new shared resolver using
   the pinned catalog rows, comparing exact modeled grooves and preserving
   unknown source alpha. This input-only integration audit must not rerun/tune the
   scientific experiment or rewrite its frozen observations/checkpoints.
5. Document architecture boundaries and an honest capability map: common software
   benefits versus unmeasured MHC-I predictive transfer. Keep source-specific
   parsing/curation in Hitlist and experiment configuration in experiment folders.
   No study IDs, source filenames or DO-specific branches in model/trainer code.
6. Run focused and full relevant modular tests, lint/format, review diff and record
   results in tasks/todo.md. Preserve the negative native DO result. No baseline
   promotion, original MS/censor gate change, remote publication, release or merge.

## Acceptance

Strict existing users behave identically. Explicit assumptions survive checkpoint
and inference provenance and never become reported typing. Both MHC classes and
bulk/cellular quantity routes demonstrably use the common modules/training path;
class-specific registers remain intact. No claim of MHC-I accuracy improvement
without a predictive transfer/regression comparison. Deliver a reviewable shared
core change and evidence, not a new experimental sidecar predictor.

## Verification adjustment after first focused run

The initial 86-case run passed 84 and found two fixture errors. Comparing
grad-enabled forward with no-grad prediction selects different Torch kernels;
compare both provenance variants through the same canonical predict method.
The new mixed-class fixture also exposed previously untrained class-II flank
residues in its synthetic test partition, correctly rejected by strict inference.
Keep the guard intact and give this integration fixture common flanking-residue
support across its distinct synthetic cores. This is a software-fixture repair,
not a change to scientific data, model behavior or a predictive result.

## Inherited repository lint repair

The full repository lint gate found one existing E501/format error in the
ordinal-context parity helper from c0358852. Preserve its exact executed bytes
as `tasks/mhc_shared_core_20261027_archive/parity.executed.py.txt`, format only
the working source, and prove AST identity. Retain all original verification
records and add a receipt mapping the original and formatted hashes; document
that historical hashes refer to executed bytes. The immutable runtime checkout
and completed scientific experiment remain unchanged.
