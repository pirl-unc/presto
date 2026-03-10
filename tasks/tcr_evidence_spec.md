# pMHC-Only TCR Evidence Spec

## Goal

Remove TCR/BCR sequences from canonical Presto inputs while preserving a pMHC-only output that captures curated evidence that a pMHC has at least one cognate TCR in receptor databases.

This document defines the exact output contract, data mappings, supported bins, and required code changes.

## Canonical Outputs

### Required

1. `tcr_evidence_logit`
2. `tcr_evidence_prob`
3. `tcr_evidence_method_logits`
4. `tcr_evidence_method_probs`

### Method panel bins

`tcr_evidence_method_*` is a multi-label panel over:

1. `multimer_binding`
2. `target_cell_functional`
3. `functional_readout`

Interpretation:

- `tcr_evidence_prob`
  - "This pMHC has curated evidence of at least one cognate TCR."
- `tcr_evidence_method_probs[k]`
  - "This pMHC has curated cognate-TCR evidence observed via assay family `k`."

These outputs are about observed evidence, not biological impossibility.

They do **not** mean:

- a specific receptor sequence is known at inference time
- all cognate TCRs bind
- absence from a database implies a negative label

## Relationship To Existing Recognition And Immunogenicity

This output family does **not** replace the existing canonical `recognition` latent.

Canonical semantics remain:

- `recognition`
  - repertoire-level propensity for some typical TCR repertoire to recognize the peptide
  - driven by peptide properties plus `foreignness`
  - upstream of `immunogenicity`
- `immunogenicity`
  - downstream response propensity that depends on pMHC interaction plus `recognition`
- `tcr_evidence`
  - database-derived evidence that at least one cognate TCR has been observed for this pMHC
  - auxiliary downstream output
  - not part of the latent DAG

In the current code this distinction already exists:

- `recognition` is a peptide-only cross-attention latent with `foreignness` as an upstream dependency in [models/presto.py](/Users/iskander/code/presto/models/presto.py)
- `immunogenicity_vec` is computed from `interaction_vec` and `recognition_vec`, not from `match_logit`
- the old TCR matcher branch is optional legacy scaffolding and should not define canonical `recognition`

Implementation rule:

- `tcr_evidence` may read from `pmhc_vec` and optionally from `recognition_vec` / `immunogenicity_vec`
- but it must not feed back into `recognition` or `immunogenicity`
- there should be no reverse dependency in the latent DAG

## Optional Auxiliary Outputs

These are supported by the data and may be useful later, but they are not required for the first removal pass.

1. `tcr_evidence_singlecell_logit` / `prob`
2. `tcr_evidence_score_ge_2_logit` / `prob`

Interpretation:

- `singlecell`
  - receptor evidence includes VDJdb `singlecell=yes`
- `score_ge_2`
  - receptor evidence includes VDJdb score `>= 2`

I do **not** recommend exposing `verified_present` in the first pass.
The field exists, but it is sparse and semantically heterogeneous.

## Label Taxonomy

### Overall label: `tcr_evidence`

Positive if the normalized pMHC appears in any curated receptor source used for canonical supervision:

- VDJdb
- McPAS-TCR

Negative:

- never use raw "not present in DB" as a true negative
- treat non-matching pMHCs as unlabeled

### Method panel: `tcr_evidence_method`

Only VDJdb currently supports a method-panel target with enough semantic resolution.

Method labels are derived from the VDJdb `method.identification` field as a **set**, not a single class.

Mapping from `method.identification` text to normalized bins:

- `multimer_binding`
  - if text contains any of:
    - `tetramer`
    - `dextramer`
    - `multimer`
    - `pentamer`
- `target_cell_functional`
  - if text contains any of:
    - `antigen-loaded-targets`
    - `antigen-expressing-targets`
    - `antigen coated targets`
    - `antigen-coated-targets`
- `functional_readout`
  - if text contains any of:
    - `cd137 expression`
    - `ifng`
    - `capture assay`
    - `elispot`
    - `ics`
    - `intracellular cytokine`
    - `cytokine`

Additional parsed bins that should **not** be modeled initially:

- `culture_stimulation`
  - `cultured-t-cells`, `peptide-stimulation`, `antigen-stimulation`, `enrichment`, `limiting-dilution-cloning`
- `display_selection`
  - `phage display`
- `other`
- `unknown`

Why those are rejected:

- `culture_stimulation` is only 74 unique pMHCs and is not a clean specificity assay family
- `display_selection` explodes in raw rows but collapses to 1 unique pMHC, so it is biologically narrow and dataset-specific
- `other` and `unknown` are too heterogeneous to be stable supervised targets

## Data Support

Counts below are from local files on 2026-03-06:

- `data/vdjdb/vdjdb.txt`
- `data/mcpas/McPAS-TCR.csv`

### Overall source support

Raw rows:

- VDJdb: `226,494`
- McPAS-TCR: `20,227`

Unique positive keys:

- VDJdb coarse key `(peptide, primary_mhc)`: `1,962`
- McPAS coarse key `(peptide, mhc)`: `427`
- coarse union across sources: `2,329`
- coarse source overlap: `60`

For VDJdb method bins, exact keys should use `(peptide, mhc_a, mhc_b)`:

- VDJdb exact pMHC keys: `1,987`

McPAS currently has only one MHC column, so it supports overall evidence labels but not the exact VDJdb-style method panel.

### VDJdb exact pMHC support by selected assay-family bins

Unique exact pMHC counts:

- `multimer_binding`: `1,217`
- `target_cell_functional`: `409`
- `functional_readout`: `291`

Raw row counts:

- `multimer_binding`: `148,285`
- `target_cell_functional`: `25,118`
- `functional_readout`: `955`

### VDJdb exact pMHC support by rejected bins

Unique exact pMHC counts:

- `culture_stimulation`: `74`
- `display_selection`: `1`
- `other`: `74`
- `unknown`: `213`

Raw row counts:

- `culture_stimulation`: `7,036`
- `display_selection`: `59,376`
- `other`: `479`
- `unknown`: `1,005`

`display_selection` was re-checked directly against raw VDJdb rows.
The `59,376` phage-display rows all map to the same exact pMHC:

- peptide: `SLLMWITQV`
- `mhc_a`: `HLA-A*02:01`
- `mhc_b`: `B2M`
- reference: `PMID:40498839`

So the `1 exact unique pMHC` count is real, not a counting bug.

### Multi-label overlap

Selected assay-family bin count per exact pMHC:

- exactly 1 selected family: `1,529`
- exactly 2 selected families: `188`
- exactly 3 selected families: `4`

Top label combinations:

- `('multimer_binding',)`: `1,181`
- `('target_cell_functional',)`: `217`
- `('functional_readout', 'target_cell_functional')`: `156`
- `('functional_readout',)`: `131`
- `('multimer_binding', 'target_cell_functional')`: `32`
- `('functional_readout', 'multimer_binding', 'target_cell_functional')`: `4`

Conclusion:

- the method panel must be **multi-label**, not multiclass

### Class distribution caveat

VDJdb exact pMHC support by class:

- `MHCI`: `1,794`
- `MHCII`: `193`

Selected method bins by class:

- `multimer_binding`
  - `MHCI`: `1,172`
  - `MHCII`: `45`
- `target_cell_functional`
  - `MHCI`: `359`
  - `MHCII`: `50`
- `functional_readout`
  - `MHCI`: `291`
  - `MHCII`: `0`

Conclusion:

- the method panel is strongly class-I biased
- `functional_readout` is effectively a class-I auxiliary target in the current data

### Optional auxiliary support

VDJdb exact pMHC counts:

- `singlecell_yes`: `1,142`
- `score_ge_1`: `633`
- `score_ge_2`: `506`
- `score_ge_3`: `251`
- `verified_present`: `317`

McPAS unique pMHC by `NGS`:

- `no`: `327`
- `na`: `118`
- `yes`: `45`

Conclusion:

- `singlecell_yes` and `score_ge_2` are usable auxiliary targets
- McPAS `NGS` support is too small and semantically weak for a canonical output

## Exact Training Semantics

### `tcr_evidence`

Target type:

- positive-unlabeled pMHC target

Positive pool:

- any pMHC present in VDJdb or McPAS after canonical normalization

Unlabeled pool:

- all other pMHCs from binding, elution, processing, and T-cell response data not present in the receptor-evidence set

Recommended loss:

1. Positive anchor term:
   - BCE on labeled positives only
2. Pairwise ranking term:
   - positive pMHC score should exceed matched unlabeled pMHC score by a margin

Recommended form:

```python
L_pos = BCEWithLogits(tcr_evidence_logit[pos], 1)
L_rank = relu(margin - (score_pos - score_unl)).mean()
L = L_pos + lambda_rank * L_rank
```

Recommended defaults:

- `margin = 0.5`
- `lambda_rank = 0.5`

Hard-negative selection:

- same MHC class
- prefer same allele or same locus when available
- similar peptide length
- high predicted presentation/immunogenicity from the current model
- not present in the receptor-evidence positive set

Do **not** train this head with plain BCE against "not in VDJdb/McPAS = 0".

### `tcr_evidence_method`

Target type:

- VDJdb-only masked multi-label target

Positive examples:

- VDJdb-positive pMHCs

Label derivation:

- for a given exact pMHC key, union all normalized assay-family bins observed across VDJdb rows for that pMHC

Masking:

- train only on pMHCs with VDJdb provenance
- mask out all McPAS-only positives and general unlabeled pMHCs

Recommended loss:

- masked multi-label BCE

Recommended form:

```python
L_method = masked_bce_with_logits(
    tcr_evidence_method_logits,
    method_targets,
    method_mask,
)
```

Why BCE is acceptable here:

- these are not global biological truths
- they are explicit "observed evidence mode in VDJdb" labels
- within the VDJdb-conditioned subset, unobserved bins can be treated as 0 for that descriptive auxiliary task

### Optional auxiliary heads

If added:

- `tcr_evidence_singlecell`
  - VDJdb-only masked BCE
- `tcr_evidence_score_ge_2`
  - VDJdb-only masked BCE

## Exact Data/Code Changes Required

### 1. New record type

Add a pMHC-only record class in [data/loaders.py](/Users/iskander/code/presto/data/loaders.py):

- `TcrEvidenceRecord`

Fields:

- `peptide`
- `mhc_a` or `mhc_allele`
- `mhc_b`
- `mhc_class`
- `species`
- `antigen_species`
- `source`
- `evidence_label: float`
- `evidence_methods: set[str] | list[str]`
- optional:
  - `singlecell_label`
  - `score_ge_2_label`
  - `score_raw`

### 2. Loader changes

[data/loaders.py](/Users/iskander/code/presto/data/loaders.py)

- extend `VDJdbRecord` or replace it on the canonical path
- parse and preserve:
  - `method`
  - `web.method.seq`
  - `vdjdb.score`
- derive:
  - overall evidence positive
  - multi-label method bins
  - optional singlecell / score auxiliaries

### 3. Cross-source merge changes

[data/cross_source_dedup.py](/Users/iskander/code/presto/data/cross_source_dedup.py)

- preserve VDJdb method metadata instead of dropping it
- emit canonical assay bucket:
  - `tcr_evidence`
- stop using:
  - `tcr_pmhc`

Recommended field usage in `UnifiedRecord`:

- `assay_type`
  - normalized method-family bins joined or stored one-per-row before aggregation
- `assay_method`
  - original source method string

### 4. Dataset / collate changes

[data/collate.py](/Users/iskander/code/presto/data/collate.py)

Add pMHC-only targets:

- `tcr_evidence_label`
- `tcr_evidence_mask`
- `tcr_evidence_method_target`
- `tcr_evidence_method_mask`

Optional:

- `tcr_evidence_singlecell_label`
- `tcr_evidence_score_ge_2_label`

Remove receptor sequence fields entirely:

- `tcr_a`
- `tcr_b`
- `tcr_a_tok`
- `tcr_b_tok`
- `chain_species_*`
- `chain_type_*`
- `chain_phenotype_*`

### 5. Model changes

[models/presto.py](/Users/iskander/code/presto/models/presto.py)

Add pMHC-only heads:

- `tcr_evidence_head`
- `tcr_evidence_method_head`

Recommended input:

- `pmhc_vec`

Optional richer input:

- `concat(pmhc_vec, recognition_vec, immunogenicity_vec)`

Constraint:

- this is a downstream auxiliary head only
- `recognition` remains the canonical upstream latent for `immunogenicity`
- do not route `tcr_evidence` back into the immunogenicity or recognition path

Conditioning rule:

- do **not** use assay/culture/APC/stimulation embeddings as inference-time inputs for these heads
- unlike the IEDB T-cell assay head, `tcr_evidence` is not predicting an outcome under a user-specified assay condition
- the method panel is part of the target definition, not an input context

Therefore:

- `tcr_evidence_head`
  - unconditional pMHC-only readout
- `tcr_evidence_method_head`
  - unconditional multi-label readout from pMHC representation

What is allowed:

- source-aware masking during training
- source-specific calibration bias terms used only during training/analysis if needed
- label embeddings inside the output head implementation if they are used as learned output queries, not as per-example conditioning inputs

What is not allowed:

- passing VDJdb assay method or McPAS metadata as context embeddings into the model at inference time
- conditioning `tcr_evidence` predictions on assay/culture metadata the user does not provide

Do not use any receptor tokens or TCR encoder states.

### 6. Training changes

[scripts/train_iedb.py](/Users/iskander/code/presto/scripts/train_iedb.py)

- replace `vdjdb_records` with `tcr_evidence_records`
- remove `sc10x_records` from canonical training
- add support-weighted loss terms for:
  - `tcr_evidence`
  - `tcr_evidence_method`
- implement matched unlabeled ranking batches for `tcr_evidence`

### 7. Remove sequence-conditioned receptor infra

Delete from canonical path:

- TCR input args in `Presto.forward()`
- `enable_tcr`
- `TCRpMHCMatcher`
- `encode_tcr()`
- `predict_chain_attributes()`
- 10x chain aux ingestion
- `tcr_pairing`
- `tcr_pmhc`
- receptor retrieval evaluation

## Exact Recommendation

Implement:

1. overall pMHC-only `tcr_evidence`
2. VDJdb-only multi-label `tcr_evidence_method` over:
   - `multimer_binding`
   - `target_cell_functional`
   - `functional_readout`

Defer:

- `culture_stimulation`
- `display_selection`
- McPAS method segmentation
- `verified_present`

Optional second wave:

1. `tcr_evidence_singlecell`
2. `tcr_evidence_score_ge_2`

## BCR Note

No analogous pMHC:BCR evidence output should be preserved.

BCR/TCR sequences should both be removed as inputs, but only TCR databases produce a coherent pMHC-level evidence target for canonical Presto.
