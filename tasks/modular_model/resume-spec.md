# Exact resume at durable training boundaries

This is the remaining continuation requirement of issue #130. It does not
qualify any biological endpoint or establish auxiliary benefit. Implement one
resume path in the existing modular trainer, with an explicit CLI switch;
pretrained transfer remains a different operation.

## State and identity

- Snapshot the current model, support and feature evidence, frozen dependencies,
  inherited scales and optimizer. Preserve ordered optimizer parameter names,
  not only positional optimizer state. Save the selected checkpoint inside the
  resume bundle so a later, uncommitted selection cannot replace it.
- Preserve epoch/order/cursor, the independent replay order/cursor, both local
  sampler RNGs, Torch CPU/CUDA RNGs, Python and NumPy RNGs, partial epoch losses
  and exposures, accumulated coverage, global steps, patience, best score and
  history. Restore RNGs after constructing model and optimizer.
- Pin the manifest/source hashes, resolved configuration/scales, implementation
  hashes, Python/Torch/NumPy versions, device and relevant deterministic/thread
  settings. Reject a changed execution contract before optimizing or reading
  evaluation labels. Device migration is transfer, not exact resume.
- Add a positive `checkpoint_every_steps` manifest option, default 100. Write
  atomically at initialization, that interval, each completed epoch/selection,
  and the transition to final evaluation. Resume starts at the last durable
  update; work after that point is replayed. Do not promise preservation of
  uncheckpointed RAM state or bitwise equivalence on unverified GPU kernels.

## Optimization and finalization

- Keep the existing sampling, loss, validation selection and stopping rules.
  Refactor loop bookkeeping without changing uninterrupted updates. A partially
  consumed epoch resumes from its saved cursor, including replay consumption.
- A durable phase transition to evaluation prevents further optimization even
  if final reporting fails. Persist each split's evaluation as one atomic cache
  containing metrics and records, tied to the selected checkpoint and run
  contract; public prediction dumps can be regenerated from that cache.
- Mark test evaluation started before consuming it. If a crash leaves this
  marker without its complete cache, fail closed rather than silently score
  the test set again. Existing complete caches are reused and verified.
- Persist completed summary/coverage/artifact hashes. A completed resume returns
  that result without training or evaluation. A failed coverage gate remains
  failed; resume cannot turn a reported failure into a new selection exercise.
- Reject an ordinary launch into a used output directory. `resume=True` requires
  a valid durable bundle. Never overwrite initial evidence on resume.

## Registered CLI

- Add `--modular-resume`, requiring explicit `--run-dir`, for both canonical
  entry points. Reject it for legacy models and in combination with preflight.
- Reuse the existing registered experiment and its frozen source manifest and
  source snapshot. Accept only the original pinned manifest or the frozen one;
  do not reinitialize registry files or rewrite original reproduction commands.
- Record resumed attempts separately. Make registry closure idempotent so a
  reporting retry does not duplicate a completed experiment. Preserve previous
  failure evidence.

## Verification

- Compare uninterrupted and interrupted/resumed deterministic CPU runs with
  dropout, identity replay and a non-epoch checkpoint: exact model/optimizer
  state, support/coverage, history, selected epoch and prediction records.
- Verify frozen dependencies/scalers survive resume and selected weights are
  recovered from the durable bundle, not an ahead-of-snapshot file.
- Change data/config/source/runtime contracts and optimizer membership to prove
  rejection before any update. Reject missing/invalid snapshots and accidental
  overwrite attempts.
- Interrupt final reporting to prove completed split caches are reused; reject
  incomplete test consumption; repeat completed closure without any rescoring.
- Exercise the registered CLI in a temporary software-fixture registry. Run
  focused modular/CLI tests, pinned Ruff, strict docs and appropriate regression.
  Artificial contract tests are not biological experiments or quality evidence.
