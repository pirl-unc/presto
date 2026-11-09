# Recover MHC-II MS evaluation after JSON allocation exceeded RSS

Codex / GPT-6,2026-10-27. Full objective unchanged. Previous goal turn: progress
(published2ed1bd8, verified cohort, two upstream issues, native/control launch).
Current authoritative native/logistic sessions78532/47037 terminated exit1.

## Evidence and root cause

Native101 completed all3epochs/2,592updates. The original action then exceeded
4GiB RSS while serializing its **training** evaluation cache: peak4,315,611,136bytes,
elapsed2,092.293 monotonic seconds. Only `evaluations/train-started.json` exists;
validation/test cache evaluation never started. The final checkpoint, selected
checkpoint, optimizer/RNG snapshot and full training history are intact.
The JSON encoder constructs a complete indented string and encoded byte buffer
in addition to the already resident prediction records. Hashing also creates
a full canonical JSON byte string.

Use a separate checkout at
`/private/tmp/presto-mhc2-evaluation-20261027/presto`, based on2ed1bd8. Preserve
the previous checkout, every failed receipt/log, all trained checkpoint bytes,
the150-step censor snapshot and older live IC50/species controllers.

## Minimal software fix

Change only canonical JSON persistence/hashing in `training/modular_resume.py`:
stream encoder chunks into the atomic temporary file and incremental SHA256
instead of materializing the full document. Preserve byte-for-byte formatting,
sort order, Unicode/float encoding, hash values, atomic replace/fsync semantics,
nonfinite-value rejection and existing resume contract behavior. Do not weaken
execution-contract matching or modify model/training/evaluation mathematics.

Before use, prove exact output/digest compatibility, previous-destination
preservation on serialization failure, and bounded serialization memory on a
large artificial repeated-record fixture. Run relevant resume/continuation/
training tests plus repository lint. Record software results in the2036family.
Any runtime measurements here qualify the serializer fix, not biological speed.

## Explicit scientific recovery

Keep the original fit/controller failure intact. Never patch a saved resume
contract or bypass its implementation checks. Native101 needs **evaluation only**:
verify final step/epoch, selected-versus-final tensor equality, trained parameter/
feature coverage, parent IC50 tensors/scaler, and saved checkpoint hashes.
Re-evaluate the fixed model with the unchanged inference code and streaming
persistence, preserving source-linked predictions and metrics. Report training
evaluation recomputation as recovery overhead; no optimizer step is added.
No test evaluation was started in the failed action.

For native102 and permuted101, start their originally specified fresh fits with
the sole runtime difference above. Freeze old/new runtime hash maps and show
all other source files identical. Original datasets/manifests/seeds/update budget,
fixed-final selection, resource floors and biological gates remain unchanged.
Use new experiment-local recovery/controller files and receipts, not edits to the
frozen original launcher. Queue the unchanged logistic comparison after these
actions, with a new wrapper that verifies the documented runtime transition.

The serializer fix is an evaluation/infrastructure recovery within the same
scientific family. Do not claim the original canonical action completed, or
compare different numerical training contracts as though identical. Preserve
the original state and clear recovery lineage in README/logs.

## Closure and remaining work

Implement independent prediction/metric replay and the prespecified ensemble/
component bootstrap. Report every arm and all failed/positive gates. Continue
binding recovery when resources permit and source-supported pathway/condition
work; this serializer fix does not finish any biological requirement.

- [ ] Preserve failure/checkpoint evidence in isolated checkout.
- [ ] Add failing memory/atomicity/byte-compatibility tests, implement minimal fix.
- [ ] Run focused/full required checks and freeze runtime transition.
- [ ] Recover native101 evaluation without new training.
- [ ] Finish native102/null and logistic controls, independently close comparisons.
