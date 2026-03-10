# Receptor Removal Plan

## Objective

Remove receptor sequences from canonical Presto end to end:

- no TCR/BCR sequences in canonical training data
- no receptor tokens in `PrestoSample` / `PrestoBatch`
- no TCR-conditioned forward path or pMHC:TCR matcher in the canonical model
- no receptor-specific CLI or predictor APIs
- no legacy receptor-pair or receptor-chain training tasks in the canonical stack

At the same time, keep the parts that still fit the pMHC model:

- peptide, flanks, MHC sequences, and side-information overrides
- processing, binding, presentation, elution, immunogenicity, and T-cell response labels
- T-cell assay-context modeling
- pMHC-only MIL contrastive regularization from the recent refactor
- pMHC-only outputs indicating whether a cognate TCR has been observed in curated receptor datasets

## Recommendation

Canonical Presto should be a pMHC plus immune-response model, not a receptor-conditioned model.

Reasoning:

1. Binding and presentation are pMHC-intrinsic. Receptor sequence is downstream of those mechanisms, not a causal parent.
2. TCR/BCR recognition is clonotype-specific and repertoire-specific. That is a retrieval/matching problem over receptors, not a natural extension of the current latent DAG.
3. The current code already reflects this philosophically in the docs: `docs/tcr_spec.md` says TCR-conditioned matching is a future feature and not canonical.
4. The runtime code contradicts the docs by still dragging receptor sequences into unified training, collation, predictor APIs, and legacy task registries.

If receptor modeling comes back later, it should be a separate experimental module that consumes a frozen or exported `pmhc_vec`, not a branch inside canonical Presto.

User-required refinement:

- keep a pMHC-only output representing curated cognate-TCR evidence from sources like VDJdb and McPAS
- optionally expose method-segmented evidence outputs when the source carries assay metadata
- remove receptor sequences as inputs while preserving these labels as downstream supervision

Recommended naming:

- overall: `tcr_evidence_logit`, `tcr_evidence_prob`
- optional panel: `tcr_evidence_method_logits`, `tcr_evidence_method_probs`

The semantic contract is:

- not "this exact TCR binds"
- not "this pMHC is universally immunogenic"
- instead: "this pMHC has curated evidence of at least one cognate TCR"

## Current State Audit

### Canonical-path receptor leakage

- `models/presto.py`
  - still has `enable_tcr`
  - still accepts `tcr_a_tok` and `tcr_b_tok` in `forward()`
  - still instantiates `TCRpMHCMatcher`, `ChainAttributeClassifier`, and `CellTypeClassifier`
  - still exposes `encode_tcr()` and `predict_chain_attributes()`
- `data/loaders.py`
  - `PrestoDataset` still accepts `vdjdb_records`, `sc10x_records`, and compatibility `tcr_records`
  - still creates `PrestoSample` objects carrying `tcr_a`, `tcr_b`, `chain_type`, and `phenotype`
  - still emits `assay_group="tcr_pmhc"` and `assay_group="chain_aux"`
- `data/collate.py`
  - `PrestoSample` still includes `tcr_a` and `tcr_b`
  - `PrestoBatch` still includes `tcr_a_tok`, `tcr_b_tok`, `chain_species_label`, `chain_type_label`, `chain_phenotype_label`, and masks
  - collator still tokenizes receptor sequences
- `scripts/train_iedb.py`
  - still loads `vdjdb_records` and `sc10x_records`
  - still exposes `--vdjdb-file`, `--10x-file`, `--max-vdjdb`, and `--max-10x`
  - still forwards `tcr_a_tok` / `tcr_b_tok`
- `inference/predictor.py`
  - still exposes `predict_recognition()` even though it raises `NotImplementedError`
  - still exposes `classify_chain()` and `embed_tcr()`
- `cli/main.py`
  - still exposes TCR-oriented train and predict surfaces
- `cli/evaluate.py`
  - still computes TCR:pMHC retrieval metrics from `tcr_vec`
- `training/tasks.py`
  - still defines `ReceptorChainTypeTask`
  - still defines `TCRPairingTask`
  - still defines `TCRpMHCMatchingTask`
- `training/trainer.py`
  - still forwards optional TCR tokens through the model

### Missing metadata for the refined output

- `VDJdbRecord` in `data/loaders.py` currently keeps peptide, allele, gene, and CDR3 fields, but not VDJdb evidence metadata such as:
  - `method.identification`
  - `method.verification`
  - `method.singlecell`
  - `method.sequencing`
  - `vdjdb.score`
- `_parse_vdjdb_content()` in `data/cross_source_dedup.py` also drops those fields, even though `UnifiedRecord` already has generic `assay_type` and `assay_method` slots.
- Therefore, method-segmented `tcr_evidence` outputs require a schema change before model changes.

### Important non-goals

- Do not remove pMHC-only MIL contrastive regularization in `scripts/train_synthetic.py` and `scripts/train_iedb.py`.
  - That contrastive path compares original versus genotype-substituted MHC bags.
  - It is unrelated to the TCR matcher.
- Do not remove T-cell response labels or T-cell assay context fields.
  - These remain valid as population-level outcome supervision without explicit receptor sequences.
- Do not remove the fact that a pMHC has cognate-TCR evidence in receptor databases.
  - Convert that fact into a pMHC-only supervised output.
  - Remove only the receptor sequence dependency.
- Do not conflate receptor-sequence removal with deleting all B-cell data.
  - `BCellRecord` parsing exists, but explicit BCR sequences are not currently part of the canonical Presto dataset path.
  - The live BCR-adjacent receptor surface is mainly the `sc10x` chain auxiliary path covering `IGH` / `IGK` / `IGL`.

## Phase Plan

### Phase 0: Baseline and Safety Rails

Purpose: lock a pMHC-only reference point before deleting interfaces.

Changes:

- Run a short no-AMP local training sanity on the current model with receptor inputs absent.
- Record:
  - dataset composition by `assay_group`
  - one mini-batch forward/backward
  - one short train trace
  - expected output keys

Why first:

- receptor code removal is broad enough that we need a clean before/after comparison
- current CUDA bf16 cleanup is still open, so the baseline should be CPU or `--no-use-amp`

Verification:

- finite forward/backward on canonical pMHC batches
- stable output-key inventory for pMHC-only tasks

### Phase 1: Convert Receptor Sources Into pMHC-Only Evidence Supervision

Purpose: stop receptor sequences at the training entrypoints while preserving pMHC-level receptor evidence.

Files:

- `scripts/train_iedb.py`
- `scripts/sanity_check_modal.py`
- `cli/main.py`
- `data/cross_source_dedup.py`
- `data/loaders.py`
- `docs/training_spec.md`

Changes:

1. Replace `vdjdb_records` with a new pMHC-only evidence record family, for example:
   - `TcrEvidenceRecord`
   - fields: `peptide`, `mhc_allele` or `mhc_a/mhc_b`, `mhc_class`, `species`, `antigen_species`, `source`, `evidence_label`, optional method metadata, optional confidence metadata
2. Extend `load_vdjdb()` to parse and preserve evidence metadata rather than only CDR3/gene fields.
3. Extend McPAS or comparable source parsers similarly where possible.
4. Remove 10x from canonical training entirely; it is pure receptor-sequence auxiliary supervision and has no pMHC-only meaning.
5. In merged-corpus logic:
   - stop treating `record_type="tcr"` as a sequence-conditioned training bucket
   - instead emit a pMHC-only assay bucket such as `tcr_evidence`
6. Update dataset summaries and config so canonical training reports `tcr_evidence`, not `tcr_pmhc` or `chain_aux`.

Implementation note:

- the safest path is to keep raw parsers for one cycle, but route their canonical output through pMHC-only evidence records.
- 10x should be removed outright from canonical training because it has no pMHC-only supervision to preserve.

Verification:

- canonical trainer no longer ingests receptor sequences from `vdjdb` or `10x`
- canonical trainer still sees pMHC-only `tcr_evidence` supervision from VDJdb/McPAS-like sources
- merged-source audit shows `tcr_evidence`, not `tcr_pmhc`

### Phase 2: Remove Receptor Fields From Dataset and Batch Schema

Purpose: simplify the canonical sample/batch contract.

Files:

- `data/collate.py`
- `data/loaders.py`
- tests touching `PrestoSample` / `PrestoBatch`

Changes:

1. Remove `tcr_a` and `tcr_b` from `PrestoSample`.
2. Remove `tcr_a_tok` and `tcr_b_tok` from `PrestoBatch`.
3. Remove receptor chain auxiliary tensors:
   - `chain_species_label`
   - `chain_type_label`
   - `chain_phenotype_label`
   - corresponding masks
4. Remove receptor tokenization from the collator.
5. Remove `vdjdb_records`, `sc10x_records`, and compatibility `tcr_records` from `PrestoDataset`.
6. Keep `TCellRecord` support, but ignore receptor sequence fields on ingest and preserve only:
   - peptide
   - MHC
   - response label
   - assay context
   - pathway MIL metadata
7. Add pMHC-only `tcr_evidence` targets and masks to the sample/batch contract.
8. If method segmentation is kept, add masked multi-label or panel targets for normalized evidence-method bins.

Verification:

- canonical `PrestoBatch` contains only pMHC, assay context, and task targets
- no batch slicing helper still references `tcr_*` or `chain_*`
- `PrestoBatch` can still carry `tcr_evidence_*` labels because those are pMHC-level outputs, not receptor inputs

### Phase 3: Remove the TCR Tower From the Model and Inference Surface

Purpose: make the model API match the canonical design.

Files:

- `models/presto.py`
- `models/tcr.py`
- `models/heads.py`
- `inference/predictor.py`
- `cli/main.py`
- `cli/evaluate.py`

Changes:

1. In `models/presto.py`:
   - remove `enable_tcr`
   - remove `tcr_a_tok` / `tcr_b_tok` from `forward()`
   - remove `TCRpMHCMatcher`
   - remove `ChainAttributeClassifier`
   - remove `CellTypeClassifier`
   - remove `encode_tcr()`
   - remove `predict_chain_attributes()`
   - stop emitting `tcr_vec`, `match_logit`, `match_prob`, `chain_*_logits`, and cell-type outputs
   - add pMHC-only `tcr_evidence` readout head(s)
2. In `models/heads.py`:
   - remove optional TCR branches if they only serve the dead matcher/retrieval path
   - keep repertoire-style pMHC-only recognition heads if still used by canonical immunogenicity/T-cell outputs
   - add or reuse a pMHC-only head for `tcr_evidence`
3. In `inference/predictor.py`:
   - remove `predict_recognition()`
   - remove `classify_chain()`
   - remove `embed_tcr()`
   - delete now-dead result dataclasses
4. In `cli/main.py`:
   - remove `predict recognition`
   - remove TCR sequence flags from user-facing prediction commands
5. In `cli/evaluate.py`:
   - remove TCR:pMHC retrieval evaluation

Decision point:

- `models/tcr.py` should either be deleted or moved under an `experimental/` namespace.
- I recommend a quarantine move only if there is a real plan to reuse it soon; otherwise hard delete is cleaner.

Verification:

- `Presto.forward()` no longer accepts receptor tokens
- predictor and CLI no longer advertise receptor operations
- pMHC-only inference still works
- `Presto.forward()` still emits `tcr_evidence_prob` as a pMHC-only output

### Phase 4: Remove Legacy Receptor Tasks and Replace Them With pMHC-Only TCR-Evidence Outputs

Purpose: eliminate leftover training infrastructure that can silently reintroduce receptor semantics.

Files:

- `training/tasks.py`
- `training/trainer.py`
- any loss/registry helpers referenced only by receptor tasks

Changes:

1. Remove:
   - `ReceptorChainTypeTask`
   - `TCRPairingTask`
   - `TCRpMHCMatchingTask`
2. Remove any registry entries or task-name plumbing for:
   - `receptor_chain_type`
   - `tcr_pairing`
   - `tcr_pmhc`
   - TCR-specific `contrastive` handling
3. In `training/trainer.py`:
   - stop passing `tcr_a_tok` / `tcr_b_tok`
   - remove stale generic `contrastive` task wiring if it refers to the receptor matcher
4. Keep only pMHC-aligned tasks.
5. Add a new canonical task such as `tcr_evidence`:
   - overall binary or PU-style output: curated cognate-TCR evidence exists for this pMHC
   - optional method-panel outputs when metadata is present
6. Prefer positive-unlabeled or ranking-style supervision over naive BCE on "not in VDJdb".
   - absence from VDJdb/McPAS is not a true biological negative
   - synthetic negatives should be treated as contrastive/ranking examples, not ground-truth absence

Important distinction:

- keep `presentation_mil_contrastive`
- remove TCR matcher contrastive / InfoNCE infrastructure

Verification:

- no canonical loss path consumes `match_logit`
- task registries enumerate only pMHC-aligned tasks
- `tcr_evidence` is trained from pMHC-only labels, not receptor tokens

### Phase 5: Docs, Tests, and Canonical Messaging Cleanup

Purpose: remove contradictory statements and lock the new contract.

Files:

- `docs/design.md`
- `docs/tcr_spec.md`
- `docs/training_spec.md`
- `TODO.md`
- test files across `tests/`

Changes:

1. Update docs to state one canonical truth:
   - receptor-conditioned modeling is not part of canonical Presto
   - receptor-derived database evidence can still supervise pMHC-only outputs
   - if future receptor work exists, it is separate experimental work
2. Remove or rewrite tests that depend on:
   - `tcr_a_tok`
   - `tcr_b_tok`
   - `match_logit`
   - `embed_tcr`
   - chain classification
   - `tcr_pmhc` task generation
3. Add regression tests for the new contract:
   - no receptor tensors in `PrestoBatch`
   - no receptor args in `Presto.forward()`
   - unified trainer ignores receptor-only sources
   - predictor exports only pMHC-oriented methods

Verification:

- docs and runtime match
- no stale TCR/BCR APIs survive in tests or help text

## Proposed Implementation Order

1. Phase 0 baseline
2. Phase 1 training-entry removal
3. Phase 2 batch-schema removal
4. Phase 3 model/inference removal
5. Phase 4 legacy task cleanup
6. Phase 5 docs/tests cleanup
7. short pMHC-only training sanity run

This order minimizes breakage:

- entrypoints stop creating receptor data first
- schema changes happen before model API deletion
- docs/tests are updated after the runtime contract is stable

## Verification Checklist

- `rg` over the codebase finds no canonical references to:
  - `tcr_a`
  - `tcr_b`
  - `tcr_a_tok`
  - `tcr_b_tok`
  - `enable_tcr`
  - `match_logit`
  - `predict_recognition`
  - `embed_tcr`
  - `classify_chain`
  - `vdjdb_records`
  - `sc10x_records`
  - `tcr_pmhc`
- dataset composition no longer includes:
  - `tcr_pmhc`
  - `chain_aux`
- dataset composition may include:
  - `tcr_evidence`
- a collated canonical batch contains:
  - peptide / flank / MHC tensors
  - task targets and masks
  - T-cell assay context
  - no receptor tensors
- local pMHC-only training sanity stays finite
- no-AMP one-epoch diagnostic remains at least as healthy as the Phase 0 baseline

## Scope Boundaries

### Keep

- T-cell response labels from IEDB/CEDAR-like assay data
- T-cell assay context modeling
- immunogenicity outputs
- pMHC embeddings such as `pmhc_vec`
- pMHC-only MIL contrastive and sparsity losses
- pMHC-only TCR-evidence outputs derived from curated receptor databases

### Remove

- explicit TCR/BCR sequence inputs
- receptor chain classification
- TCR alpha/beta pairing tasks
- TCR:pMHC contrastive / matcher tower
- TCR retrieval evaluation
- 10x receptor supervision from canonical training

### Transform

- VDJdb / McPAS-style receptor data:
  - from sequence-conditioned matching supervision
  - to pMHC-only `tcr_evidence` supervision with optional method segmentation

### Defer

- whether archival raw receptor parsers stay in-tree under `experimental/`
- whether a future standalone receptor model should consume frozen `pmhc_vec`

## Final Recommendation

Proceed with removal.

This is not just cleanup; it resolves a conceptual mismatch:

- the canonical model is trying to learn population-level pMHC biology
- the receptor path injects sparse, heterogeneous, clonotype-specific signals that do not belong in the same training contract

The right end state is a smaller, sharper model and trainer:

- pMHC plus immune response in canonical Presto
- pMHC-only `tcr_evidence` outputs from curated receptor datasets
- receptor-specific modeling, if revived later, as a separate downstream system
