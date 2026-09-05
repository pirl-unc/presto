# Lessons

## 2026-09-03

- Never catch a broad runtime exception in a test and translate it to
  `pytest.skip`: that turns broken APIs, stale caches, schema drift, and import
  shadowing into apparent environmental absence. Skip only a specifically
  identified missing optional prerequisite; once it imports, let the original
  failure fail loudly, and keep an AST meta-test preventing the broad pattern
  from returning.
- Put pytest options in the configuration file pytest actually loads. This repo
  has `pytest.ini`, so a `[tool.pytest.ini_options]` block in `pyproject.toml` is
  ignored; assert the effective option through `pytestconfig` rather than
  merely checking configuration text.
- When a focused fix needs its own PR while the main worktree is dirty, create a
  clean worktree and branch directly from freshly fetched `origin/main`. Do not
  mix, stash, or otherwise disturb the existing changes.

## 2026-09-02

- Keep three versions separate for editable dependencies: the live source/runtime
  version (`package.__version__`), the install-time `.dist-info` version reported by
  `pip` / `importlib.metadata`, and the version that produced a cached artifact. Do
  not assign the current runtime version to an older artifact unless its provenance
  metadata stamps that version; otherwise label the producer version as inferred and
  state the evidence.

## 2026-08-26

- State an ablation as a factorial, not "with vs without". Adding a corpus usually adds
  data, a loss, and a new supervision channel at once; a single contrast cannot
  attribute the result to any of them. Name the arms and say what each isolates, and
  include a shuffled-label arm so "extra gradient signal" is ruled out separately from
  the mechanism.
- Pair every outcome metric with a mechanism check. "Does elution AUPRC improve" and
  "did the detectability latent actually become identifiable" are different questions,
  and the second one is measurable directly.
- New supervision belongs in the canonical trainer, not a fourth script. If the
  canonical path cannot yet measure the thing being added, that measurement gap is part
  of the work — surface it in the plan rather than quietly adding another trainer.
- When an identification argument rests on comparisons (same protein, different enzyme;
  same enzyme, observed vs not), the batch sampler has to guarantee those comparisons
  co-occur. Random sampling confounds them with whatever varies across groups.
- Check what a library already provides before writing the spec for it. Two upstream
  asks in this session were already implemented; one was filed as an issue before
  checking and had to be publicly corrected. Read the actual output columns first.
- A held-out prediction dump must apply the same target transform the loss applies.
  Dumping raw targets (nM) against transformed predictions compares two different
  spaces, which makes RMSE, MAE and Pearson meaningless and the CSV impossible to read.
  Worth fixing on its own merits — but note the follow-on lesson below.
- When the same failure shape appears a third time, stop patching instances and fix the
  class. Four tests here flaked for one reason — numeric assertions over unseeded
  randomness — and under xdist each run blamed a different test, which read as an
  infrastructure problem. One autouse `torch.manual_seed` fixture in `conftest.py`
  removed all of them. Note what it does not do: seeding makes a weak assertion
  reproducible, not correct.
- Check whether a proposed explanation is even capable of producing the symptom before
  acting on it. I hypothesised that the mismatched target space explained an observed
  negative binding Spearman and fixed it expecting the sign to flip. Spearman is
  invariant to monotone transforms, so it could never have been the cause; the value
  moved -0.0606 -> -0.0765 and stayed negative. The correct reading was the boring one
  all along: one epoch of a 32-dim model over 136 measurements.
- `float("nan")` does not raise, so `try: float(x) except` is not a null check. Test
  `value != value` explicitly, or NaN targets enter training and poison a loss silently.
- Verify a claimed input is actually populated end to end before trusting any result
  that depends on it. Presto declared `nflank`/`cflank` inputs for months while the
  canonical data path supplied neither, and every recorded baseline listed them.
- Never launch Presto with `~/code` on `sys.path` (either as cwd or via `PYTHONPATH`).
  That directory contains a `mhcseqs/` repo folder with no `__init__.py`, which Python
  treats as an empty namespace package and which shadows the real installed `mhcseqs`.
  The failure is silent and looks like a data problem: `mhcseqs.__file__` becomes `None`,
  every allele fails to resolve, and the trainer reports
  `MHC sequence coverage: resolved=0/11622` then drops 100% of MHC rows in
  resolved-only mode. Both packages are installed editable, so run from any neutral
  directory with no `PYTHONPATH` and both import correctly.
- When a test fails, first ask whether the test or the code is wrong. Two failures here
  were incorrect assertions about the design (a sanctioned context channel, and a
  context term that legitimately shifts) — asserting the wrong invariant would have
  masked the real contract.

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
- When the user explicitly asks to continue with a broader, more varied data contract (inequalities, multiple numeric assay families), do not steer the next experiment batch back to a narrower exact-only comparator unless that narrower run is strictly needed to debug a blocker. Keep the sweep aligned with the requested contract.

- When parallel experiment sweeps are happening, keep one canonical `experiments/` registry with per-experiment directories and a unified log. Do not let results live only in `tasks/experiment_log.md` or scattered `modal_runs/` tables.
- Do not maintain two canonical experiment registries in parallel unless there is a proven automation need. If a human-readable markdown log already carries the decision record, prefer generating machine-readable summaries later over hand-maintaining a second JSONL index that can drift.
- A finished experiment is not complete when the runs stop; it is complete only after metrics are extracted, the experiment directory contains the summary tables/plots/artifact links, and `experiments/experiment_log.md` has a contextualized writeup with dataset, training, assay mapping, conditions, and conclusions.
- For experiment reproducibility, freeze the exact launch invocation and relevant environment overrides inside the experiment directory itself (`reproduce/launch.sh`, metadata JSON, launcher snapshot). Do not rely on mutable launcher defaults or chat context to explain how a run was started.

## 2026-03-12 - Experiments need recomputable held-out metrics
- If an experiment only saves aggregate losses and probe metrics, it is impossible to answer later questions like exact-IC50 rank correlation or <=500 nM accuracy.
- Future experiments must save enough per-example validation/test predictions and labels to recompute downstream metrics without rerunning the training job.
- Experiment docs must state the held-out split policy clearly and include both validation and test metrics, or explicitly justify why a test split was not used.
- Treat the per-example prediction dump plus the held-out metric table as required closure artifacts for predictive evaluation sweeps, not optional nice-to-haves.

- 2026-03-13: In output-head/target-encoding experiments, do not freeze the target encoding to the previously winning model-specific setting. Keep backbone/data/schedule fixed and vary the encoding itself so the comparison is interpretable.
- 2026-03-13: Modal CLI bool parameters use flag-style syntax: include `--flag` for True, omit it for False. Do NOT pass `--flag true` or `--flag false` — Modal treats the value as an unexpected positional argument.
- 2026-03-13: When the user says to stop depending on a shared experiment runner and make a benchmark self-contained, do not keep patching the shared path. Freeze a local package/launcher and have Modal execute that exact package.
- 2026-03-13: If a Modal image excludes `experiments/**` or other local paths from the base repo upload, a self-contained experiment package will not exist in-container even if the wrapper path is correct. Explicitly add the experiment-local code directory to the image when the runtime depends on it.

## 2026-03-14 - Experiment closure must pull ALL raw artifacts

- When closing out a Modal experiment, pull ALL raw output files (summary.json, probes.jsonl, metrics.jsonl, step_log.jsonl) into the experiment directory — not just summary.json. The experiment directory must be self-contained: all summaries, tables, and plots must be reproducible from local data alone, without re-fetching from the Modal volume.
- Do not consider an experiment "analyzed" if you only extracted aggregate metrics (Spearman/AUROC) from summary.json. Per-allele probe predictions, per-epoch training curves, correlation metrics, and binary classification metrics (<=500nM) must all be extracted and included.
- For multi-head experiments, extract metrics from ALL output heads, not just the primary prediction path.
- If an experiment run finishes, do the closure work immediately before reporting status upstream. "Runs are done" is not a meaningful completion state on its own; completion requires local artifact harvest, summary tables, README updates, and `experiments/experiment_log.md` updates in the same work cycle.

## 2026-03-16 - Keep stable-model docs separate from experiment writeups

- Do not turn per-experiment `README.md` files into an ever-growing scoreboard for the current best model family. Keep each experiment README focused on that experiment's contract, result, and artifact pointers.
- When the user wants a durable "model to beat" record across datasets/metrics, create or update a separate summary document under `experiments/` rather than stuffing that comparison into every new experiment README.
- When experiment-local launchers are canonicalized, use the shortest stable convention (`code/launch.py`) rather than verbose tool-specific names like `launch_modal.py`. Keep shared machinery unique in `scripts/`, and keep per-experiment entrypoints local and consistently named.

## 2026-03-16 - Architectural input invariants must be enforced, not merely described

- When the user declares that a feature family is forbidden as model input, do not leave it as one option among many mode flags. Remove it from defaults, reject it in public entrypoints, and state the invariant consistently in code, docs, and tests.
- For Presto affinity modeling specifically, assay identity belongs on the output/supervision side only. Assay labels may choose which head is supervised, but assay selector/context tensors must not be available as predictive inputs in the main path.
- When the user broadens an architectural rule from one task family to all assay families, update the canonical contract immediately instead of leaving a narrower document name or scope in place. The repo should describe one assay-input policy, not separate local exceptions by default.
- When claiming that a path is "sequence-only," verify the actual training and evaluation call sites end-to-end. Config flags and docstrings are not enough if the runner still passes `binding_context` into the model.
- When comparing "best honest" models across benchmark families, audit both the input contract and the output contract before promoting a result. A path can avoid the forbidden input leak and still be non-comparable because it collapses multiple assay families into one assay-conditioned output.
- Do not over-interpret probe peptides as supervised affinity anchors without checking the merged dataset first. A probe can exist in the corpus only as presentation / structural metadata and still look like a meaningful affinity sanity check if you skip the support audit.
- When the user says hardware-dependent implementation differences are not acceptable, do not keep a split default with one backend on a workaround and others on native behavior. Promote the shared implementation to the default and require explicit opt-out for backend-specific benchmarking.
- Never pop a stash you did not create. While splitting a branch into PRs I used stash/pop around a `commit --amend` to test whether the committed tree was self-consistent, and popped a pre-existing stash from another branch (`isk/curriculum-and-class2-plans`), dropping three unrelated experiment files into the tree and leaving `experiments/experiment_log.md` conflicted. Nothing was lost -- a conflicted pop keeps the entry, verified before touching anything -- but recovery was awkward because the worktree-discard commands are blocked, correctly, since from the tool's view I was about to throw away uncommitted work. Run `git stash list` *before* any stash operation: a stash that predates the session belongs to someone else. To check a commit's self-consistency, inspect it directly with `git show <sha>:<path>` or use a throwaway clone -- do not stash. To recover, write the intended content back via `git show HEAD:<path> > <path>`, which discards nothing already committed or stashed. When two safety rules fire in a row, re-read the situation instead of looking for a third way around.
- Fetch and check the remote base before branching, and before trusting a local test run. I built PR 2 off a stale local `main` (`e05f062`) while the remote had already merged PR 1 (`4276acf`), so the two test files PR 1 added were invisible locally. Every local run reported a confident 1164/1164 against a tree CI collected as 1178, and the seven failures it found were in exactly the drift-pinning test I had just made stale by renaming a field. A green local suite proves nothing when the working tree is missing files the base branch has. `git fetch && git log main..origin/main` costs nothing; run it at the start of any branch-and-PR sequence and again before claiming a suite passes. Related: when a file the summary says exists turns up missing, suspect a stale base before concluding the file was never written.
- "CI does not lint it" is not "the config is dead". I deleted a `per-file-ignores` stanza for `experiments/` reasoning that `lint.sh` never passed that path, and stripped the only protection reaching every *other* ruff caller: editors, LSP, pre-commit, and a bare `ruff check .`. Verified after the fact — 546 findings appeared and `ruff format .` offered to rewrite 211 files including frozen experiment snapshots, the exact falsification the comment I wrote in the same commit said it prevented. Before deleting config as dead, enumerate who reads it, not just who I happen to be running. And exclusion belongs in the config with `force-exclude = true`, since an explicitly-named path bypasses a plain exclude — which is precisely what format-on-save does.
- A scale mismatch does not raise, so nothing catches it but magnitude. The assay-panel loss regressed a normalized-log10 head against a raw-nanomolar target: finite loss, training ran to completion, all 1,417 tests passed, and the model learned nothing — validation moved 0.012% over ten epochs while one term sat at ~142,900 and swamped every gradient. Normalizing it took the same run to a 36% reduction. When a model trains but does not learn, print the per-task loss magnitudes before touching architecture; a term orders of magnitude off its neighbours is the whole answer.
- Do not "tidy" a fixture whose value looks wrong. A test row declared `seq_len: "89"` against 77 residues; I made it self-consistent while reflowing long lines and deleted the only in-repo example of the `seq_len_mismatch` condition the code validates. Deliberately-wrong data is a fixture, not a typo — check what consumes it before making it agree with itself.
- State the invariant that is true, not the one that is tidy. I wrote a guard asserting benchmark argv tuples alternate flag/value; 27 of 75 failed because `store_true` flags legitimately carry no value. The real invariant — no two consecutive values — is weaker, actually holds, and still catches the shift it was written for. A guard built on a premise the data contradicts gets deleted, not fixed.

## 2026-09-04

- Do not infer production masking behavior from a tiny synthetic frame. Audit the
  actual corpus at both the source-row and expanded training-record levels, with
  stable row identifiers and explicit before/after transition counts. A four-row
  example can exercise code paths without establishing their prevalence or
  semantics.
- A controlled experiment is not ready merely because its flags look aligned.
  Compare invariant preflight fingerprints against the reference family before
  launch, including row counts, coverage, resolution, masking rates, and pinned
  parser versions. Any unexplained mismatch means the supposedly single-variable
  comparison is not yet controlled.
- When frame-level and record-level audits make contradictory claims about the
  same signal, stop the launch and reconcile lineage through filtering, joins,
  grouping, and record expansion. A green test suite or matching aggregate
  coverage cannot establish semantic correctness while that conservation check
  fails.
- Delete constants that encode disproven invariants instead of renaming or
  numerically adjusting them. Preserve the measured counterexample near the
  surviving logic so the false assumption is not independently re-derived.
- Pin biologic sequence parsers as part of the data contract and verify their
  extracted boundaries on the actual allele set. A dependency version that moves
  mature-chain starts changes model inputs even when the training code is
  otherwise identical.
- Never normalize a nullable sequence with `str(value or "")`. Floating NaN is
  truthy and stringifies to `"nan"`; after upper-casing, `"NAN"` is composed
  entirely of valid amino-acid codes and silently becomes fake biological
  sequence. Detect pandas/NumPy nulls before string conversion and test every
  sequence-normalization entrypoint with `None`, `float("nan")`, `pd.NA`, and
  empty strings.
- A required-lineage gate must begin by checking the required identity key. Do
  not put the rest of the validation under `if identity`: that makes missing
  identity bypass the gate it is supposed to enforce.
- Every new CLI destination supported by YAML/JSON configuration must be added
  to the canonical defaults registry in the same change and tested through the
  config merge, including repeatable lists, nullable values, and booleans.
- Keep source provenance immutable across resolution joins. For multi-allele
  observations, store the complete reported allele set separately from the
  resolved model-facing subset; filtering one must never rewrite the other.
- A common funnel schema needs an adapter for every loader. Reading Hitlist-only
  keys from merged-TSV statistics silently produces a pretty but incomplete
  audit on the default path; test required stages and drop reasons per source.
- Keep the curation seed and split/model seed visibly separate in rerun commands.
  When validating split stability across model seeds, pin `data_seed` once and
  vary only `seed`; otherwise reservoir membership changes and a curation drift
  can be mistaken for a split effect. Assert the dataset-level supervision hash
  before accepting any per-split comparison.
- A package catalog may mix named alleles, accessions, partial grooves, and
  invalid sequences even when single-allele lookup is reliable. Default
  augmentation must select canonical named alleles with complete class-correct
  groove inputs and report rejection counts; never sample the raw catalog
  inventory directly.
- Do not guard a canonical-resolver retry with `if fallback_path`. When
  `mhcseqs` is primary and the CSV is optional, the resolver must be called for
  missing training, augmentation, and diagnostic alleles even when the fallback
  path is absent; the resolver itself decides whether a supplement is needed.

## 2026-09-05

- Define source-derived mapping observations from the union of their explicit
  mapping signals: a mapping category identifies mapped and unmapped Hitlist
  rows, while a positive candidate count independently proves mapping occurred.
  Test both signals instead of assuming either field always implies the other.
- A named data source must select one exact loader path. Do not let the
  presence of an unrelated file silently turn a source-specific run into a
  hybrid dataset; any hybrid contract must be explicit and separately named.
- Load immutable curated records once when auditing several split seeds. Reuse
  that dataset for deterministic resplitting instead of retaining or rebuilding
  multiple full-corpus copies at the same time.
- A frozen experiment script must add the repository root itself to `sys.path`,
  never its parent. A sibling checkout with the same package name can otherwise
  be imported as an empty namespace package and poison every later test in that
  Python process.
- A self-describing checkpoint must preserve every constructor option that
  changes parameter structure, including latent topology. Held-out evaluation
  must fail if the selected checkpoint cannot be reconstructed; silently
  scoring the final in-memory epoch makes the artifact internally inconsistent.
- A capped in-loop validation pass is an optimization control, not the final
  evaluation contract. Rebuild an uncapped held-out loader for the selected
  checkpoint, compute full loss terms, and emit every validation/test example;
  never fall back to the last in-memory model or the capped iterator.
- End-to-end verification must exercise each named data source while competing
  files are present. Otherwise an opportunistic loader branch can look healthy
  in isolation while silently creating a hybrid corpus in production.
- A lineage test that expects an allele to be resolved must supply the exact
  sequence input itself. Letting an installed process-wide MHC registry satisfy
  the assertion makes local success depend on optional machine state and lets
  clean CI exercise a different semantic case.
- Install optional-package stubs before importing the script under test. A
  monkeypatch applied after module execution cannot isolate a top-level import
  and creates a local-only pass when that optional package happens to exist.
