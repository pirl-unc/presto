# Lessons

## 2026-02-24

- When the user resolves design conflicts with explicit canonical decisions, convert those decisions into one single canonical spec before making code edits.
- Do not keep parallel contradictory wording ("implemented" vs "future/TODO") across docs; enforce one status statement across architecture, training, and index docs.
- For missingness, avoid conflating unknown content with missing modality. Keep dedicated tokens/flags for missing values and separate unknown-token handling.
- Never silently tokenize unresolved structured identifiers (like allele names) as sequence content; fail fast with an actionable resolution path.
- For training data, unresolved MHC alleles must be treated as data-quality errors (with explicit reports/allowlists), not silently fed as token strings.
- Even in non-strict/debug inference modes, unresolved identifiers should degrade to empty/missing sequence signals, not raw identifier tokenization.
- If the user asks for strict sequence semantics, tokenizer defaults must fail fast on unfamiliar characters; compatibility `<UNK>` behavior should be opt-in only.
- Avoid hard clipping before smooth calibration layers in probabilistic heads; this can collapse distinct weak-signal cases into identical outputs and hide model behavior.
- When the user asks for internal probabilistic handling, keep the external interface simple unless they explicitly request user-facing probability controls.
- If class uncertainty is already modeled by inferred `pI/pII` downstream, avoid duplicating class as an early token-level embedding path; it adds coupling and can mask where class signal is entering.
- Respect user vocabulary preferences as canonical naming conventions: if the user defines a term (for example `10x`), normalize user-facing args/help/log IDs/tests to that term and avoid legacy synonyms (`tenx`) in outputs.
- For Python-only identifiers where `10x_*` is awkward, prefer explicit prefixes like `sc10x_*` instead of flipped forms like `x10_*`; the latter is ambiguous and easy to misread.
- Prefer positive boolean controls in public APIs (`synthetic_negatives`) and keep any legacy negative-form flags/kwargs hidden or compatibility-only.
- For synthetic-negative modes, encode perturbation target and method directly in names (`peptide_scramble`, `mhc_random`) and document `random` (de novo generation) vs `scramble` (permutation) explicitly.
- When canonical naming is decided, remove legacy aliases and bridges promptly; update all callers/tests/configs in one pass instead of leaving compatibility code behind.
- When a training-control matrix (like synthetic negatives) matters for interpretation, keep a compact table in `README.md` with defaults and mode semantics so users can verify enablement without digging through code.
- When users are confused by many coupled controls, collapse user-facing knobs to a minimal set and derive deterministic downstream behavior internally; keep biologically independent controls (like processing negatives) explicit.
- When discussing `--max-*` training caps, always pair the explanation with concrete corpus-size numbers and explicitly state that GPU memory scales with batch tensors, not total dataset rows.
- Do not disable probe tracking (for example `--no-track-probe-affinity`) unless the user explicitly asks for it; defaults should preserve requested control monitoring like `SLLQHLIGL` checkpoints over epochs.
- When the user asks for fallback biological data sourcing (e.g., allele/gene lists + UniProt scraping), make provenance first-class: include source URL, query string, accession, and derivation rule per sequence so mis-mapped proteins can be audited later.

## 2026-03-02

- When filtering MHC sequences for augmentation/training, check ALL non-canonical characters (not just `X`). The MHC index may contain `?`, `*`, or other non-AA characters in sequences. Always validate against the full `MHC_SEQUENCE_ALLOWED_AA` set, not a hardcoded subset.
- Every code path that produces sequences for the tokenizer must go through the same quality filter. The MHC augmentation path (`_generate_mhc_only_samples`) initially bypassed the training data quality check and allowed `?`-containing sequences to crash the tokenizer at batch 1054.

## 2026-03-04

- If the user specifies an information-flow constraint (for example "recognition depends on foreignness + peptide only"), enforce it end-to-end: latent DAG dependencies, extra-token routing, and any shared-state injections (core-relative encodings, global conditioning) must all obey the same boundary.
- When exposing side-information overrides, treat them as explicit latent-variable fixes (override hooks) rather than silent feature leaks into unrelated pathways.
- When the user asks for explicit architectural semantics (for example whether `cd8/cd4` branches are distinct), answer with concrete code-path evidence and then codify any requested behavior in tests so the contract stays explicit.

## 2026-03-05

- When vocabulary changes remove token classes (for example B/Z/O/U), update every input-validation path in the same patch (predictor heuristics, CLI validators, loader filters), not just tokenizer vocab.
- If design docs describe conceptual boundary tokens but runtime actually uses segment IDs only, align documentation to execution semantics immediately and add tests around the semantic behavior (segment routing/isolation) rather than token names.
- For full-data merge/unification steps, treat performance instrumentation as a first-class requirement: add per-stage timing stats + tqdm progress in the same patch as algorithmic optimizations so bottlenecks are obvious without ad-hoc profiling.
- For heavyweight merge outputs, default to minimal required artifacts and make duplicated convenience exports opt-in; avoid writing multi-GB duplicate files unless explicitly requested.
- When a field is named `*_allele_set`, enforce allele-like token constraints at ingestion (exclude class labels/serotypes/mutants) and codify with tests; never allow semantic drift into mixed free-text labels.
- Do not infer cell-level allele sets from global cell-type unions across papers (e.g., all `B cell` rows). Use experiment-specific keys (at least PMID + cell context) and drop rows lacking verifiable allele evidence when strict semantics are requested.
- For long-running merge debugging, avoid output modes that hide stage progress (`--quiet`) or flood logs (raw tqdm redraws). Prefer visible stage timings and disable tqdm redraw overhead (`TQDM_DISABLE=1`) when capturing logs.

## 2026-03-06

- When training looks flat and the user suspects sampler imbalance, inspect actual per-batch task composition and support weights directly before blaming architecture. Global corpus counts are not enough.
- When adding biologic sequence validation, match the threshold to the project's accepted representation first. If groove-only MHC fragments are allowed, do not hard-code a full-chain length floor.
- Before choosing a biologic length cutoff, inspect the empirical length distribution. Use the lowest threshold that removes obvious garbage without throwing away acceptable partial domains.
- When the user asks to remove a modality's sequences, do not assume they want to remove all supervision derived from that modality. Distinguish "remove as model input" from "keep as pMHC-level output/label" and plan the data/model boundary explicitly.
- Keep three concepts distinct in this codebase: repertoire-level `recognition` latent, downstream `immunogenicity`, and TCR-database evidence / TCR-specific matching. Do not collapse them into one "recognition" concept in plans or APIs.
- When a new upstream architecture plan lands before a downstream refactor is implemented, re-stage the whole change around shared interfaces first. Do not execute two separate rewrites of `loaders` / `collate` / `model.forward()` if one coordinated contract change can cover both.
- For biologic parsing heuristics, do not stop at human and mouse reference alleles. Before declaring the parser sound, audit the actual species/gene distribution in the local corpus and explicitly inspect non-model groups the user flags, such as birds and fish.
- When auditing class-II groove extraction through a generic dispatcher, always pass or derive the chain explicitly. A default-to-alpha convenience path can make valid beta-chain records look like parser failures in ad hoc audits.
- Keep fine-grained biological identity for parsing/indexing separate from coarse model buckets. Do not collapse fish/bird/non-model species too early just because the network currently uses a smaller classification taxonomy.
- When a third-party biologic parser is the canonical source of truth, harden the integration against packaging quirks instead of silently falling back to handwritten heuristics. Normalize import behavior first, then make failures explicit.
- For MHC chain inference, prefer allele-name/gene inference from the canonical parser before weaker sequence-only heuristics. Sequence heuristics are a recovery path, not the primary source of truth.
- When the user wants allele names at protein resolution, enforce a single `mhcgnomes`-based two-field normalization contract across resolver maps, defaults, and index aliases. Do not leave mixed-resolution names in canonical dictionaries just because they happen to exist in source FASTA headers.
- Canonical biologic lookup dictionaries should be keyed by stable allele namespaces (`HLA`, `SLA`, `Mamu`, etc.), with free-text species names treated as aliases rather than the primary key.
- Do not infer nomenclature semantics from a dirty mixed-completeness index audit. If two-field allele collisions appear, first separate full-length-vs-fragment records from true same-length amino-acid disagreements before concluding that the naming scheme itself is inconsistent.
- For MHC two-field representatives in a groove-based model, do not stop at raw sequence equality. Prefer a hierarchy of: nested longest record, conservative overlap assembly, then groove-equivalent exemplar; only leave the alias ambiguous when the candidate groove content actually disagrees or no structurally valid representative can be built.

## 2026-03-07

- When the user says training should run on Modal, do not keep iterating on laptop-bound diagnostics out of convenience. Move focused probes and subset ablations onto Modal as first-class entrypoints.
- Before interpreting a probe peptide as a fitting failure, verify whether it has direct quantitative supervision in the merged corpus. A peptide with only elution or T-cell evidence is a generalization probe, not a supervised binding target.
- Do not trust probe diagnostics from head-capped training slices on the merged corpus. The TSV ordering can wipe out allele-specific supervision for exactly the allele you are trying to compare. Use reservoir sampling for any probe/representation sanity run unless the user explicitly wants first-N behavior.
- Even reservoir row sampling can destroy the multi-allele peptide-family structure needed for allele-discrimination diagnostics. When a short binding canary is supposed to test same-peptide allele ranking, audit the sampled slice for actual shared-peptide families and bootstrap them explicitly if needed.
- Profile presets must never silently override explicit CLI flags. If presets are applied after parsing, track which destinations the user actually set and preserve them.
- Keep `mhcgnomes` strict on canonical class/species inference, but helper paths used only for auxiliary labels (for example coarse gene extraction) must degrade gracefully on coarse shorthands instead of crashing the whole training job.
- When a regularizer is supposed to improve mechanistic focus (for example MHC-attention sparsity), test it on the smallest discriminative toy before trusting it in training. A prior that looks biologically plausible can still destroy the exact signal you need the model to learn.
- When the user asks for diversity-preserving batching, enforce it explicitly at batch construction time across the requested biological axes (at least allele, MHC class, and species). Inverse-frequency weighting alone is not a sufficient guarantee.
- If a gradient-flow audit is requested, trace every saturating nonlinearity on the active prediction path (`tanh`, hard `clamp`, bounded calibrations) before changing losses. Do not assume a mostly-GELU network is free of saturation bottlenecks.
- When a tiny canary disagrees with known biology but a task-focused larger subset recovers the correct sign, prefer `scale-first` and task-focused data before adding another new loss term. Use the smallest new objective set that the larger relevant subset still cannot solve.
- For focused synthetic-augmentation diagnostics, keep validation real-only. Synthetic negatives belong in the training split; putting them in validation contaminates the measurement you are trying to interpret.
- In allele-panel binding diagnostics, same-peptide / different-allele ranking can amplify corpus-level allele priors instead of peptide specificity. Check whether it helps the fit-supported probes before assuming it is universally beneficial.
- Do not over-interpret tiny numeric ordering differences as meaningful biological “sign” errors. When allele predictions differ by only a few nM on a multi-log-scale affinity target, report them as effectively tied unless the gap is materially large.
- When comparing a complex model against a simpler baseline, match the label contract exactly before drawing conclusions. For binding-affinity diagnostics, a baseline trained on exact `IC50` rows is not comparable to a model run that still mixes censored `>` rows.
- When the user asks to judge a specific assay output, make that output the primary supervised target in the focused experiment. Do not use an internal probe head as the headline metric unless the user explicitly wants a latent-only diagnostic.
- Any new MHC-only pretraining or auxiliary data path must reuse the same sequence-character validation contract as the main tokenizer path before batching. Do not assume indexed groove columns are already token-safe; validate extracted halves before sample construction.
- Keep a durable experiment ledger in-repo once architecture/data sweeps start. Do not rely on thread context or scattered `summary.json` files to remember which design/data contract produced which result.
- Any artifact poller that re-fetches Modal outputs must overwrite local copies explicitly (`modal volume get --force`), otherwise the leaderboard can freeze on stale first-epoch snapshots and mis-rank designs.

- When scaling a focused binding diagnostic to a broader assay family, prefer full task-relevant data with batch-time balancing over row caps; caps are only a debugging speed tool, not a training contract.
- When a code path supports both explicit allele panels and `train_all_alleles`, test the empty-panel branch explicitly; otherwise peptide-family splitting can silently collapse validation to a fallback peptide and invalidate broad-run metrics.
- Distinguish three synthetic-negative concepts explicitly: changing anchor positions, making anchors biologically implausible, and changing MHC context. Do not casually describe all three as “anchor-aware”; users will rightly call out that these are different assumptions.
- When a scalar already influences a prediction through one canonical route, do not inject it again through multiple residual/bias heads without benchmarking against known-good runs. Redundant score-to-assay couplings can destabilize synthetic-negative training even when real-data-only runs still look acceptable.
- When synthetic negatives are important to interpretation, treat three things as first-class experimental knobs: regeneration cadence, per-batch real:synth composition, and mode mix. A fixed precomputed synthetic pool can look like “data augmentation” while really acting like a static biasing dataset.
- Audit synthetic MHC perturbation modes separately from peptide-only perturbations. Even if the training path runs, `mhc_scramble` / `mhc_random` may differ sharply in groove-parser fallback behavior and must be benchmarked in isolation before mixing them back together.
- For synthetic-negative ablations, do not trust validation loss alone. Keep a small fixed biologic probe panel and reject any mode that improves val loss while collapsing known orderings.
- Treat groove-parser fallback rate as a gating metric for synthetic MHC corruption modes. If a mode regularly produces `no_cys_pairs`, `no_alpha2_pair`, or variable-length groove halves, it is not a safe default training negative.
- When an experiment family reveals that the current benchmark base has regressed relative to an earlier stronger contract, stop stacking new ablations on the weaker base. Restore the strongest known baseline first, prove parity, and only then layer new factors on top.
- When reproducing a historical baseline, match the dataset-selection flags exactly. Do not combine `--train-all-alleles` with an explicit probe allele panel and then call it parity; verify row counts against the original run before launching follow-on ablations.
# Lessons

- When iteration speed is part of the goal, prioritize data/setup/runtime bottlenecks before widening architecture sweeps. For the focused affinity runner, `num_workers=0`, `pin_memory=false`, repeated row filtering, and collate-time tokenization are more likely wall-clock bottlenecks than small model-parameter deltas.
- When benchmarking runtime variants, keep the semantic training contract fixed and measure setup time, per-epoch wall-clock, and data-wait/compute breakdown explicitly. Otherwise a “faster” run can just be silently changing the task or the amount of work done.
- On the ~44k mixed-assay multi-allele contract, do not assume dataloader tuning is the main lever. The 16-variant runtime sweep showed `train_forward_loss_s` and `train_backward_s` dominate wall-clock, while `train_data_wait_s` stayed much smaller. Optimize model-side compute and Python work in the training step before chasing more worker processes.
- When detached Modal runs fail to publish checkpoint artifacts, fall back to structured app logs if the training loop already emits JSON summaries. Treat `modal app logs` as a first-class collector path, not only the checkpoint volume.
# Runtime benchmark harness lesson (2026-03-10)
- When adding benchmark-only CLI/runtime knobs, smoke-test the full entrypoint that uses them before launching a sweep. Unit tests on helper functions were not enough; the runtime sweep failed because `_build_epoch_train_state(...)` did not accept the new loader args even though lower-level loader tests passed.
- Do not treat a static no-augmentation training loop as the canonical runtime benchmark if the intended production path always includes dynamic augmentation or pair mining. A static-path runtime sweep is only a lower-bound diagnostic and must be labeled as such.

- When a runtime benchmark is meant to inform the eventual production trainer, do not optimize or benchmark a static no-augmentation fast path as canonical. Preserve the intended dynamic regime (pair mining / synthetic refresh), and push performance work into fixed dataset metadata, index-based pairing, and asynchronous epoch-state generation instead.

- When the user wants one canonical model improved by experiments, do not keep recommending specialized fast-path models or separate affinity-only execution as the architectural direction. Optimize the shared training/dataflow path instead, and treat specialized paths only as temporary diagnostics if explicitly requested.
