# Assay Modeling Contract

This document is normative for canonical Presto across assay families.
The filename is historical; the scope is repo-wide.

## Canonical Invariant

Presto must never consume **measurement-apparatus** metadata as predictive
input. It may consume **biological state**.

The distinction is causal, not syntactic. Both are categorical per-example
metadata; what separates them is whether the thing is part of the system being
measured or part of the instrument measuring it.

**Biological state — allowed as input.** Properties of the cells or the
molecules, which causally determine what is presented and which you know when
you pose the question:

- `peptide`, `nflank`, `cflank`, `mhc_a`, `mhc_b`
- host species
- antigen-presenting cell type (tumour line vs dendritic cell vs other) --
  this sets the expression level of every antigen-processing component, which
  is the mechanism the model is trying to represent
- antigen-processing machinery perturbations (TAP, ERAP, B2M, tapasin
  knockouts and inhibitors)
- cytokine and stimulation state (IFN-gamma, type I IFN, TLR)

These are inputs for the same reason the MHC allele is: a TAP-null cell
genuinely presents a different repertoire, and "what does this tumour present"
is a question whose answer depends on the tumour. Asking the model to predict
across them instead would discard information the questioner actually has.

**Measurement apparatus — forbidden as input.** Properties of how the
observation was made, which you do not know at prediction time and do not want
reflected in the answer:

- assay type, method, prep, geometry, readout
- instrument or platform id
- peptide format (pulsed peptide vs expressed construct)
- assay duration bucket
- culture context, insofar as it describes the assay rather than the cells
- any assay id / assay selector / assay context tensor

Feeding these means the model cannot answer "is this peptide presented"
without first being told how you intend to look, and it lets assay-specific
bias masquerade as biology.

**The test to apply to a new field:** if two rows differ only in this field,
are they two different biological situations, or the same situation observed
two ways? The first is an input; the second is an output track.

## Output Contract

Presto should predict all supported assay outputs in parallel from shared latent representations.

Allowed:
- shared pMHC latents
- assay-specific output heads
- shared head parameters or learned output-side assay/task structure
- learned assay/property embeddings that live on the output side rather than in the per-example input
- loss routing that uses assay labels to choose which output target is supervised

Forbidden:
- feeding assay identity back into the model as an input feature for the same example

In short:
- assay labels may choose supervision targets
- assay/task descriptors may parameterize output heads
- assay labels may not condition the predictive input path

This rule applies to:
- binding affinity / kinetics / stability
- presentation / elution / MS
- T-cell assays
- future assay families unless a stricter canonical replacement document supersedes it

## Optional Class / Species Overrides

MHC class and species should be inferred from `mhc_a` / `mhc_b` by default.

If explicit user overrides remain useful, the preferred mechanism is:
- treat them as constrained priors on the existing MHC-derived logits/probabilities
- keep them optional
- avoid introducing separate free-form side-input embeddings into the assay prediction path

That is better than adding a new categorical feature stream, because it preserves the rule that the model's content inputs are the sequences themselves.

## Current Repo Status

The main Presto affinity codepath is expected to enforce this sequence-only contract directly.

Some older experiment code and the current T-cell assay head still describe or use assay-context conditioning. Those paths should be treated as historical or pending refactor work, not as the canonical direction for Presto.

For clarity:
- affinity is already enforced as sequence-only in the main path
- T-cell assay conditioning is still a legacy implementation that violates this broader policy
- elution/MS should follow the same outputs-only assay rule; if assay/platform structure is modeled later, it must remain output-side rather than input-conditioned


## Compliance status (2026-08-31)

Inputs and outputs are split on the causal test above, not on whether a field
is categorical.

**Biological state — inputs, verified to reach the prediction:**

| field | vocabulary | corpus coverage |
|---|---|---|
| peptide, flanks, MHC alpha/beta | sequence | — |
| host species | — | — |
| `cell_lineage` | 10 lineages by processing phenotype | 80.6% of elution rows |
| `sample_origin` | 3 (unknown, primary, cell_line) | 100% |
| `disease_state` | 5 (unknown, healthy, tumor_adjacent, cancer, diseased) | 100% |
| `apm_perturbation` | 7 (TAP, ERAP, B2M, PLC, class II loading) | 5.69% of class I rows |
| `processing_stimulus` | 7 (IFN-gamma, type I IFN, TNF, TLR, activation, unspecified cytokine) | 3.86% |

These three provenance axes replace a single flat `apc_cell_class` label. That
label was wrong in three ways: it was derived from the worst-covered field
(58.3%, against 100% for the two booleans), it conflated orthogonal axes, and
it could not express the space that actually occurs — solid tissue, solid and
haematological cancers, donor blood, PBMC, sorted immune cells, and cell lines
derived from any of those. That space is a product, not an enum: a primary AML
blast and an AML cell line share a lineage and a disease state but differ in
origin, and that is the difference most likely to matter.

`cell_lineage` groups by antigen-processing phenotype rather than tissue of
origin, because that is the mechanism: professional APCs and lymphoblastoid
lines carry constitutive immunoproteasome, high TAP and high MHC-I, while most
solid-tumour lines carry little immunoproteasome unless induced and some have
lost TAP or B2M outright. `sample_origin` earns its own axis because
immortalized lines drift and routinely lose immunoproteasome subunits, TAP or
B2M. An unrecognized value maps to `unknown`, never to a guess.

**Measurement apparatus — output tracks, verified inert as inputs:**

| family | tracks |
|---|---|
| binding observables | `KD`, `IC50`, `EC50`, `Tm`, `t_half`, `koff`, `kon` |
| binding assay descriptors | `binding_assay_panel_*` -- 11 types, 6 preps, 5 geometries, 4 readouts |
| T-cell assay conditions | `tcell_panel_logits` -- method, readout, APC type, culture, stim, format |
| TCR | `tcr_evidence_method` |
| MHC identity | `mhc_class`, `mhc_species`, `mhc_a_fine_type`, `mhc_b_fine_type` |

`binding_context` and `tcell_context` are inert: the former is ignored and its
input-side embeddings are deleted, the latter's seven arguments are gone from
`TCellAssayHead.forward`. `tests/test_many_output_contract.py` asserts both as
observable invariance, and asserts the converse for biological state -- that
changing the APC class or APM state *does* move the prediction, since
asserting invariance there would be asserting the bug.

**Counterfactual tracks.** `excision_panel_apm` and `excision_panel_stimulus`
predict what a peptide would look like under each cellular condition. These
coexist with conditioning rather than replacing it: the scalar prediction uses
the observed state because it is known and causal, and the panel answers "what
would this look like in a TAP-null cell", which is what makes the machinery
interpretable.

**Still absent.** TCR sequences never reach the model; `TcrEvidenceRecord`
contributes only a binary "some receptor was found". `ElutionRecord.tissue` is
still dropped.
