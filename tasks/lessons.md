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

- Do not trust probe diagnostics from head-capped training slices on the merged corpus. The TSV ordering can wipe out allele-specific supervision for exactly the allele you are trying to compare. Use reservoir sampling for any probe/representation sanity run unless the user explicitly wants first-N behavior.
- Profile presets must never silently override explicit CLI flags. If presets are applied after parsing, track which destinations the user actually set and preserve them.
- Keep `mhcgnomes` strict on canonical class/species inference, but helper paths used only for auxiliary labels (for example coarse gene extraction) must degrade gracefully on coarse shorthands instead of crashing the whole training job.
- When a regularizer is supposed to improve mechanistic focus (for example MHC-attention sparsity), test it on the smallest discriminative toy before trusting it in training. A prior that looks biologically plausible can still destroy the exact signal you need the model to learn.
