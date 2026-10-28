# Lessons

## 2026-10-16

- Copy optimizer and scheduler names from the pinned trainer when writing a
  prospective experiment plan. A base learning rate is not a constant schedule,
  and AdamW should not be shortened to Adam. Preserve any original published
  plan and timestamp corrections; never imply that prose changes alter the
  already frozen executable recipe.

- Before carrying a prospective contract across software releases, inspect every
  embedded artifact fingerprint. Producer-version receipts can change while
  observations and membership stay identical. Declare the expected metadata
  delta explicitly and require canonical recomputation; never silently accept a
  drifted source contract or confuse receipt identity with biological membership.

- Before launching a real-data fit, execute every active target transformation on
  the frozen raw labels and compare physical values, censor boundaries and units.
  Coverage counts and preserved source columns do not prove that effective targets
  equal the original measurements. Baseline target space must be declared before
  predictions; a discovered mismatch requires a preserved stop and a new contract.

## 2026-10-14

- When the user requests review, iteration and deployment, keep ownership through
  passing final-head checks, merge, clean-main build and verified publication.
  A draft PR or running CI is an intermediate checkpoint, not completion.
  Check the actual package namespace and release destination before publishing.

## 2026-09-07

- Freeze source and test files during a full regression run. Tests using
  inspect.getsource resolve live file lines against already-imported code
  offsets; edits or formatting mid-run can make them inspect a different
  function. Finish edits first, then run the final suite on an immutable commit.

## 2026-09-06

- Sequence-only encoding is not context-free biological prediction. Keep
  metadata out of residue encodings while allowing explicitly scoped cellular
  and repertoire context into downstream biological components.
- Never conflate APC MHC with the organism/selection-system MHC conditioning
  the T-cell repertoire. Likewise keep APC species, MHC molecular species,
  individual TCR species, repertoire-system species and antigen species
  independent; missing fields do not authorize copying another role's value.
- The user's repertoire-context requirement supersedes the previous proposed
  peptide-plus-foreignness-only recognition restriction. Historical lessons
  using that restriction as an example must not override the clarified design.
- APC processing knockouts/interference and APC cytokines are permitted
  biological inputs. Scope them by affected component/cell, retain concurrent
  interventions, and distinguish unknown status from an explicit control.
  Do not classify a field as forbidden solely because its source column says
  stimulation, culture, or assay context.

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

- Derive resolved MHC lineage from the chain segment actually exported for the
  allele, not from a locus-family shortcut. Class-II beta loci include DRB,
  DQB, and DPB; exact resolver metadata is the authoritative alpha/beta signal,
  and tests must cover beta-only `groove2` exports plus a partner-only negative
  boundary.
- Normalize every loader-side destructive filter into the common funnel at the
  adapter boundary, including counters nested under diagnostic groupings. A
  source statistic is not auditable merely because it survives somewhere in
  loader JSON: assert the canonical `drop_reasons` category in both persisted
  JSON and flattened CSV.
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

## 2026-09-08

- Trace units from the loader record through dataset construction, collation
  and output heads before declaring a mismatch. A multiplication by 60 can
  convert hours to minutes; it does not establish a minutes-to-seconds contract.
  Existing source-unit declarations and semantic regression tests take priority
  over an inferred unit from one transform.

## 2026-09-09

- A passing isolated build does not resolve a workflow that breaks subsequent
  tests in the user's checkout. Verify discovery before and after the real
  preparation command. Explicitly exclude generated artifact roots while keeping
  undeclared source packages, including similarly named nested packages, visible.

## 2026-09-10

- An `AttributeError` on a sibling package's object is a diagnosis to finish,
  not a "dependency drift" note to hand back. Resolve the symbol: check the
  installed distribution's version and path, look for a local checkout of that
  sibling, and compare against the declared floor of whichever package calls
  the method. Reporting an unrelated test failure without naming its cause is
  the lazy stopping point the root-cause rule exists to prevent.
- In the shared virtual environment every sibling repo is installed editable
  from `/Users/iskander/code/<repo>`. A sibling that is instead a plain PyPI
  build is the anomaly and will silently sit behind the working-tree code that
  imports it. Check install mode before blaming the caller.
- `pip check` does not catch a violated floor when the requirement sits behind
  an optional extra, and an editable install's recorded metadata can lag its
  working tree. Read the caller's `pyproject.toml` rather than trusting a clean
  `pip check`.

## 2026-09-13

- Check a cited path exactly as written before calling it missing. Twice a
  parity script was reported absent after looking one directory too shallow; the
  file existed at the dated path the note gave.
- `ruff` printing "No Python files found under the given path(s)" before "All
  checks passed!" means nothing was checked. Presto excludes `experiments/` from
  lint on purpose, so scripts added there are unlinted by policy; say so rather
  than reporting them as lint-clean.
- Prefer scoping a cache to a read-only operation over detecting every mutation.
  A SQLite connection's `total_changes` looked like a complete invalidation token,
  but DDL table swaps, temp tables shadowing a table, backups or deserialization
  into the connection, and writes from another connection all bypass it. Caching
  only inside `report()` makes staleness impossible elsewhere by construction.
- A rollback inside a library method must undo only that method's work. Use a
  savepoint rather than `connection.rollback()`, which also discards the caller's
  earlier uncommitted writes.
- A mutation that survives the whole suite is a finding even when current code is
  correct. Pair each new reuse or invalidation test with the mutant it must kill,
  and run the new tests against the pre-fix code to prove they fail there.

## 2026-09-14

- A reproducibility artifact is not included merely because it exists in the
  working tree. Before claiming that an experiment bundle is tracked, inspect
  `git status --untracked-files=all` and the staged file list for every linked
  script and generated summary; verify the links from a checkout-visible path.

## 2026-10-15

- The user explicitly deferred MHCflurry and authorized a Hitlist/Presto PR
  series. Respect that scope. Use datacache for shared immutable input artifacts,
  retain citable sources and distinct cell/MHC/condition/assay fields, and deliver
  actual hashes, label policies, partitions, coverage and leakage reports. Software
  tests alone do not qualify a corpus or establish biological independence.

- Never put a GitHub closing keyword directly before an issue reference, even in
  a negated sentence. GitHub closed #115 from a negated phrase in #121. Use
  "leaves #115 open", and inspect closing events after merges.
- When adapting an audit recipe, scope replacements to one syntax location and
  inspect every changed call. Exercise snapshot/publication paths before freezing
  a real-source run. Preserve failed executed recipes and logs as separate attempts.

## 2026-10-16 — source qualification priority

Prioritize source lineage and donor/specimen independence before more fitting when
the user identifies those as blockers. Interpret missing context against the assay:
a purified-MHC binding reaction does not require a cellular processing assignment.
Keep applicability reporting separate from model features and the measured label.
# 2026-10-20

- “Validated VDJdb” can refer to independent functional re-testing (TCRvdb),
  not a VDJdb confidence-score threshold. Resolve the cited source before
  defining the training cohort; preserve non-validation's assay scope.
- Distinguish installed package freshness from frozen data-export freshness.
  A newer software version does not imply a newer training bundle, and a
  narrowly varying predictor that fails on training needs an optimization
  diagnostic before collecting more rows.
- For a requested research dataset, inspect author and downstream public
  repositories before declaring acquisition blocked. A small derived subset
  or structure archive does not establish access to the complete functional
  labels and QC table.
- Distinguish shared representations from their scalar readout heads when
  describing training. Downstream presentation gradients can train a processing
  latent while leaving the returned processing score at initialization.
- Audit actual loss aliases, generic target reuse and active-task normalization
  before calling a model an IC50 fit. Source endpoint preservation is not proof
  of independent endpoint supervision; a successful forward pass does not
  establish that every returned output was trained.

## 2026-10-20 — modular model requirements

- Keep explicit class-II registers and within-peptide flanks; one whole-peptide measurement gives one loss after aggregation.
- Species defaults are user-authorized inference assumptions with explicit overrides, never molecular supervision or source facts. Preserve APC/MHC/TCR/repertoire/antigen roles.
- Retain concurrent component losses, unknown/control distinctions and matched cellular outcomes. Keep purified binding invariant to APC state.
- Separate digestion from shared MS response; a combined observation fit does not establish independently calibrated detectability.
- Compile only modules reachable from a declared subset; preserve labels, missingness, evidence tiers and coverage in one canonical path.
- Audit a PR against the frozen issue, not a narrower implementation handoff.
  Separate executable architecture, source preparation, enforced training gates
  and demonstrated predictive quality before declaring a milestone delivered.
- A nonzero gradient for a named weight tensor does not establish support for
  every input column or category. Audit feature/state exposure and guard unseen
  effects; an untrained-parameter report is not an enforced closure gate.
- The full requested pretraining/context/MS/TCR architecture remains the objective.
  An IC50 diagnostic isolates a failure mode; it must not replace the requested
  system or silently remove explicitly requested auxiliary supervision.
- Define "identity" as supervised sequence annotations whenever discussing the
  model; ambiguous "identity plus IC50" shorthand obscures both information flow
  and the full objective. Auxiliary classification success does not establish
  assay learning. Use a supplied-metadata assay-only control to test transfer,
  and distinguish detached classifier inputs from freezing their shared encoder.
- Build focused pytest commands from actual `rg --files tests` results or a
  verified file glob. A plausible test filename can stop collection entirely;
  an invocation reporting no tests supplies no verification evidence.
- Inspect actual archive array names, shapes and dtypes before calling an adapter
  source-ready. A synthetic fixture with newer `*_groups` arrays did not establish
  that the pinned Weirdo archive contains them. Missing grouping remains unknown
  until an explicit reference audit reconstructs it; never invent supplied buckets.
- An experiment launcher imported by normal tests must not import optional
  baseline libraries globally. Keep real fitting imports inside that execution
  path and verify report-helper imports with the optional library deliberately
  unavailable in a fresh process. The broad shared environment can hide a clean
  CI collection failure; do not solve that by skipping the new tests.

## 2026-10-21 — processing products before MHC selection

- A request to sample antigen-processing trajectories includes the broader
  peptide-product pool, including sub-ligand fragments and nonbinders. Do not
  narrow the target to successfully presented pMHCs. Preserve nonbinding product
  identities and quantities, distinguish them from degradation/removal and
  missing observations, and expose the ligand repertoire as a downstream
  conditional distribution. Rewards, data contracts and evaluation must retain
  the broader support rather than assigning every nonbinder zero value.

## 2026-10-22 — homogeneous study curation

- When admitting a new biological dataset, capture what distinguishes its studies,
  samples, measured populations and observation scopes in the shared Hitlist
  metadata system. Per-study prose or a one-off adapter is not sufficient.
- Check the existing upstream roadmap before inventing a parallel registry.
  Preserve scope, evidence, unknowns and exact versus surrogate material joins;
  separate source ingestion from endpoint-specific training eligibility.

## 2026-10-22 — local training preference

- When the user rules out Modal, use local execution and establish actual hardware
  support before considering another provider. A sandbox reporting MPS unavailable
  does not establish absent hardware; verify with a permitted local GPU calculation.
- Distinguish a local feasibility run from converged predictive performance and
  biological qualification; preserve the broader availability objective.

## 2026-10-22 — file upstream data gaps as they appear

- When a biological experiment hits missing data, metadata, pairing or representation limits, file a concrete Hitlist issue with source evidence, the blocked modeling use and acceptance criteria. Search existing issues first; link or extend an existing owner instead of creating duplicates. Do not leave actionable gaps only in Presto experiment prose.

## 2026-10-23 — constant-predictor metrics

- Centering exactly equal predictions can leave floating-point residuals that produce spurious correlation ranks. Preserve raw artifacts, explicitly identify mathematical constant controls, and report their correlation as undefined; never interpret roundoff as predictive spread. Keep macro distribution error and pooled peptide correlation weighting distinct.

## 2026-10-23 — concurrent upstream release state

- Before editing an issue or reporting a dependency as open, inspect its actual current state and latest upstream release. A body can contain stale open-work language even after another change closed it. Preserve legitimate closure and reconcile draft data adapters with the new canonical interface; do not relabel pinned experimental inputs as a newer release.

## 2026-10-23 — analytic ties in distribution controls

- Mathematical ties can occur within a nonconstant predictor. Before interpreting rank metrics, preserve the raw values and evaluate equivalent spans through their exact analytic class (time, length and boundary type for a uniform cut-rate control). Do not widen numerical tolerances or let summation roundoff turn those ties into ranks.


## 2026-10-23 — per-peptide temporal references

- When a temporal endpoint uses the earliest common positive reference, retain the reference time per peptide. Group sampled plug-in diagnostics by target and reference time; do not assume one reference per condition. Verify this separately from the gradient estimator and preserve failed diagnostic attempts.

## 2026-10-23 — metric weighting labels

- Check the actual reduction for each diagnostic before labeling it weighted. Sharing a dataset or retaining weight columns does not imply every summary uses those weights. Report pooled counts and equal-parent reductions separately; preserve numerical artifacts when correcting prose.


## 2026-10-23 — preserve metric reductions in acceptance gates

- Mean absolute error and absolute mean signed error are different gates. Store
  both with explicit reduction names, test a cancellation counterexample, and
  compare gate implementation with the registered mathematical definition before
  launch. If a discrepancy is found afterward, preserve original outputs and
  decisions, enforce the original contract in independent closure, and disclose
  changed decisions without changing seeds, thresholds or data.

## 2026-10-23 — biological outcomes before numerical qualification

- When asked about biological prediction, lead with actual development/test
  scores against biological controls. Numerical gates, parameter updates and
  optimizer replay do not establish predictive improvement. After a bounded
  numerical/gradient prerequisite passes, run the matched biological comparison
  before extending sampler or solver machinery. Keep nuisance-optimization gaps,
  biological generalization and absolute concentration calibration distinct.

## 2026-10-24 — reference inventories versus model inputs

- Before calling a nullable reference field an upstream blocker, inspect its
  explicit status and the rows actually reachable from the modeling cohort.
  Unmatched inventory entries can legitimately omit full sequences while every
  observed occurrence remains supported. Keep source uncertainty intact and fix
  consumer assumptions; do not file an upstream bug without a reachable failure.


## 2026-10-24 — Gate every declared representation before biological fitting

- Aggregate prediction agreement can hide larger errors in low-probability paths. Execute every registered path-level numerical gate before launching biological fits; do not substitute response-level equality. If a later check fails, stop the live fit, preserve its artifacts, diagnose producer versus verifier arithmetic, and freeze a numerical recovery before rerunning unchanged scientific conditions.
- For ragged kinetic paths, avoid global prefix accumulation across unrelated origins. Use local sums with bounded padding and test long multi-origin populations; preserve original tolerances. Make reference arithmetic explicitly float64 rather than relying on integer-tensor promotion.
- Reused qualification code must implement the registered reduction and denominator floor exactly, even when that floor happens to be inactive in the observed probes.

## 2026-10-24 — numeric evidence embedded in metadata values

- A quantitative source audit must inspect values as well as field names. mzIdentML can encode numeric intensity in spectrum-title/name strings even when no CV term is named intensity. Preserve raw values, verify duplicate representations, establish their measurement grain and aggregation, and correct public claims before calling an identification export quantity-free. A schema-name check alone cannot prove absence of numeric observations.

## 2026-10-24 — Independent verifier scope

- Keep training weights in a distinct binding from per-split evaluation weights.
  A verifier can accidentally reconstruct a later model using the last evaluated
  split, producing a false training-integrity failure. Preserve the failed verifier,
  inspect the production fitter separately, and require numerical replay after a
  verifier-only correction; never alter a frozen model to satisfy a broken checker.

- Every multi-command publication shell must begin with `set -e`; verification
  failure must prevent commit/push. CSV's valid default CRLF can trip Git whitespace
  checks. Publish an LF derivative with parsed-cell equality and preserve original
  bytes, instead of changing frozen numerical results or ignoring the check.

## 2026-10-25 — warm starts and historical outcome exposure

- An initializer fitted on the full original training cohort leaks information
  into a new cross-validation split of that cohort. Use data-free fixed starts or
  refit strictly inside the fold; an inspected-development label does not excuse
  invalid comparisons. Check warm-start provenance before registering the runner.
- A later fit's exclusion of test parents does not prove the partition was never
  inspected. Read earlier admission/review exposure notes before claiming a fresh
  test; descriptive response summaries count as prior outcome exposure even when
  no predictor was evaluated.

- An isolated experiment syntax check does not cover other newly published Python
  files. Run repository lint/format over the tracked checkout before publication,
  including task helpers. Preserve exact executed source when formatting archived
  helpers, and prove AST equality rather than silently rewriting provenance.

## 2026-10-24 — Independent regression replay and rank ties

- Agreement of predictions to a numerical tolerance does not imply agreement of raw Spearman to that tolerance. Verify coefficients/predictions independently, recompute published metrics from the exact archived values, and separately audit rank discrepancies on mathematically identical inputs. Preserve raw values and failed checks; explicitly record any revised verification contract rather than silently loosening a tolerance or rounding biological results.

## 2026-10-24 — Production boundary coverage

- A reference-algebra identity is not a test of the production implementation at that boundary. Invoke the actual function for every named boundary/monotonicity gate before fitting. If closure finds missing coverage, preserve original checks and fits, register a post-fit amendment, and disclose its timing rather than retroactively claiming the prospective gate ran.

## 2026-10-24 — shared disk headroom during source verification

- Memory-safe jobs can still exhaust a shared disk reserve. Large source replay
  creates temporary copies that compete with tests' provenance scratch space.
  Check free-space headroom for simultaneous temporary outputs before launching;
  serialize source materialization and the full suite on a near-full volume.
  Preserve disk-guard failures, never lower the reserve to make checks pass, and
  rerun only after authoritative capacity has recovered.

- Before handing a storage blocker back to the user, audit project-owned duplicate
  datasets, per-run split copies, assembled-download fragments and ignored caches.
  Directory size is not reclaimable space on APFS; measure capacity after cleanup.
  Preserve unique evidence while eliminating verified redundant storage. Keep
  immutable inputs once per content hash and avoid recopied source tables per fit.
- APFS cloning can assign different macOS provenance metadata. Preserve the
  original and skip that candidate; do not remove or rewrite security metadata to
  force a storage optimization. Record successful operations so recovery never
  repeats deletion or replaces an already processed file blindly.


## 2026-10-24 — source headers versus data layout

- Audit field types and structural invariants against actual worksheet bodies before
  labeling a discrepancy biological. An inserted unlabeled column or missing ID
  column can invalidate an otherwise exact literal-cell replay. Preserve original
  coordinates, register the interpretation, and verify it independently against
  source spans and alternate views. Never manufacture missing IDs from row order.

- Gene-name columns can contain date-formatted Excel serials. Preserve literal
  serial/style and rendered date, verify the representation, and flag the gene
  annotation as unusable; never infer the intended gene symbol from that date.
- A200response is not a successful data retrieval. Require the completed receipt,
  expected payload type and checksums before parsing; distinguish a resource-floor
  refusal from authentication, client challenges and absent data. Fetch commands
  must return a failing exit code when retrieval fails. Reuse successful recorded
  API routes, not historical failed defaults retained in an old launcher.

- Scientific artifact manifests must explicitly exclude tool caches. Before
  pushing, verify every required published manifest entry against the committed
  Git tree, not only the local filesystem. Retain required ignored experiment
  logs through an exact reviewed whitelist; never force-add entire raw directories.


## 2026-10-25 — Preserve named release authentication

- Twine's explicit repository URL bypasses named `.pypirc` credentials. When an
  existing named release configuration is available, verify its resolved endpoint
  is the intended official registry, then use that repository name. Do not expose
  credential values, rewrite user credentials or ask for a token before checking
  the supported configuration path. Preserve failed upload receipts; independently
  compare published distribution bytes before declaring a tested release complete.

## 2026-10-25 — One network budget across discovery tools

- Count search queries, documentation opens and retries against a single prospective discovery budget, including tools outside the guarded fetcher. A local fetch receipt cannot establish time/byte compliance for separate browser calls. Freeze the scope before switching tools; disclose deviations and stop rather than reset the budget.

## 2026-10-26 — PR completion includes the agreed scientific gates

- When a research PR explicitly retains scientific completion criteria, do not
  replace them with software-only merge readiness. The user confirmed that #131
  must remain draft until its scientific milestones pass. Preserve failed gates,
  external-data blockers and the full accepted scope while making independent
  progress; passing CI does not authorize biological promotion or early merge.

## 2026-10-26 — Follow scientific work through its terminal evidence

- When asked to finish research work, do not stop after queuing experiments and
  reporting progress while independent authorized work remains. Follow terminal
  runs through verification and closure, address failures, and use waiting time
  for the next qualified source or integration requirement. An external blocker
  for one assay does not block the rest of the accepted scope.


## 2026-10-27 — MHC-II development must strengthen the common core

- The user requires reusable machinery across MHC-I, MHC-II and other Presto
  endpoints. Keep shared behavior in the existing input/model/trainer path;
  adapters translate source evidence and experiment code configures comparisons.
  Do not promote diagnostic controls into task-specific production predictors.
- Verify shared gradients, mixed-class training/checkpoint behavior and endpoint
  isolation. Shared architecture does not establish MHC-I predictive improvement;
  that requires a measured transfer/regression comparison.


## 2026-10-27 — Sharing inside an optional model is not repository consolidation

- When the user asks for a common core, inventory ALL default factories, CLIs,
  synthetic/pretraining scripts, receptor wrappers, sweeps and remote launchers.
  Sharing classes within an optional model does not satisfy one training path
  while the default and older scripts still train other architectures.
- The user explicitly chose one observation-manifest interface. Retain old
  checkpoint inference and frozen reproduction, but do not preserve competing
  training APIs by adding another backend. Test executed entry-point identity.
- A flank encoding borrowed from a published baseline is not validated for all
  Presto tasks. Preserve ordered full context when feasible; distinguish richer
  input representation from demonstrated accuracy improvement.

## 2026-10-27 — Architecture changes need measured comparisons

- Do not stop at software verification when the user's purpose is better prediction.
  Once benchmarking is requested, execute a frozen accuracy/runtime comparison and
  show its evidence, rather than merely repeating that gains are unmeasured.
- Sharing parameters does not prove cross-class transfer. Test both directions,
  identify untrained class-specific paths, and keep zero-shot diagnostics distinct
  from target-class fine-tuning and supported serving.
- When the user asks to watch comparisons, create a live side-by-side view before
  fitting. Stream training/validation progress; reserve test evaluation for the
  prespecified selected checkpoint.

## 2026-10-27 — Cross-class transfer needs trained candidate evaluation

- Sharing an encoder is insufficient when register selection or flank projections
  are class-specific and untrained in source-only fits. Prefer a common candidate
  representation and supervised readout before testing cross-class generalization.
  Record remaining distribution shift; architectural sharing is not accuracy.

## 2026-10-27 — A live dashboard is not a result

- When opening an empty experiment dashboard, explicitly say that training has not
  started and name the current preparation stage. Surface failed admission and
  recovery, not a vague preparing label. Do not let optional presentation or new
  architecture ideas indefinitely delay the already agreed predictive comparison.

## 2026-10-27 — Use authorized compute and avoid arbitrary startup blockers

- The user explicitly reauthorized Modal, overriding the earlier local-only
  preference. Use requested H100! and record hardware, rather than delaying
  prediction runs behind invented laptop RAM caps. Keep active useful work running,
  show actual metrics promptly, and distinguish CPU references from GPU comparisons.

## 2026-10-27 — shared-model scale-up

- Record interpreter and package versions by running the actual executable; do not
  infer them from an environment name or an older reproduction receipt.
- Verify producer conventions for source aliases and zero/one-based row ordinals
  before declaring a provenance conflict. Cross-check immutable raw fingerprints
  and original assay identities, and preserve frozen original observation files.
- For data-size comparisons, match validation opportunities by update count. Keep
  partially consumed data epochs and pending validation durable across interruption.
- Tensor movement plus a gradient elsewhere is not coordinate-level training
  evidence. Preserve same-coordinate masks, fixed padding, structural redundancy
  and unknown historical coverage separately.
- A training-only frozen package may omit public inference imports. Before a remote
  serving replay, export the complete package from the same pinned commit, prove
  the frozen subset is byte-identical, and import the actual public predictor.
  Preserve failed verification attempts without changing fitted artifacts or
  numerical tolerances.
- When expanding labels at a fixed update budget, quantify endpoint/class exposure
  before attributing gains or losses to transfer. Pair capacity controls and distinguish
  unique outcomes from repeated optimizer labels; report source attrition and allele
  concentration instead of presenting one dataset-size number without context.

## 2026-10-28 — Live progress must survive slow artifact downloads

- Keep worker-status polling independent of archive download/extraction. A blocked
  collector can make healthy GPU jobs look stalled for minutes. Show remote fit
  completion, local artifact collection and source poll age separately.
- Recover observability without killing the original Modal controller or relaunching
  fits. Use one atomic writer per status file and preserve the validation/test gate.
- A subprocess's completed status can precede its parent's archive commit and
  final receipt. Artifact consumers and terminal pollers must require that receipt,
  not only the status string; preserve failed transport attempts and revalidate
  caches when restarting a reader.
