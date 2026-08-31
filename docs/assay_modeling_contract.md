# Assay Modeling Contract

This document is normative for canonical Presto across assay families.
The filename is historical; the scope is repo-wide.

## Canonical Invariant

Presto must never consume assay-selector metadata as predictive input.

The canonical sequence-side inputs are:
- `nflank`
- `peptide`
- `cflank`
- `mhc_a`
- `mhc_b`

There is intentionally no per-example assay-selector input such as:
- assay type
- assay method
- assay prep
- assay geometry
- assay readout
- instrument/platform id
- APC type
- culture context
- stimulation context
- peptide format
- assay duration bucket
- assay id / assay selector / assay context tensor

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

Presto is a many-output model in the sense a DNA sequence model is: the trunk
reads peptide and MHC, and every assay configuration and condition is an
*output track*, never an input feature. Track identity selects which output the
loss reads; it never selects what the model computes.

**Compliant — predicted jointly, label routes supervision only:**

| family | tracks |
|---|---|
| binding observables | `KD`, `IC50`, `EC50`, `Tm`, `t_half`, `koff`, `kon` |
| binding assay descriptors | `assay_type` (11), `assay_prep` (6), `assay_geometry` (5), `assay_readout` (4) via `binding_assay_panel_*` |
| T-cell conditions | `apc_type`, `assay_method`, `assay_readout`, `culture_context`, `peptide_format`, `stim_context` via `tcell_panel_logits` |
| TCR | `tcr_evidence_method` |
| MHC | `mhc_class`, `mhc_species`, `mhc_a_fine_type`, `mhc_b_fine_type` |

**Removed input paths.** The binding assay context is gone structurally, not
merely zeroed: `factorized_context_dim` is 0 and the four input-side
embeddings no longer exist, so nothing can refill the slot. `binding_context`
is still accepted as an argument and ignored;
`tests/test_many_output_contract.py` asserts the prediction does not move when
it is supplied.

The T-cell context is no longer supplied by any training or evaluation path,
so every trained model is a context-free predictor.

**Known remaining deviation.** `TCellAssayHead` still *accepts* the seven
forbidden keys and its prediction still moves when given them. Only the
callers were changed. Closing this means deleting those arguments from the
head, which alters its input dimensions and needs a checkpoint migration. The
gap is pinned by
`TestTCellIsContextInvariant::test_head_remains_conditionable_and_that_is_the_open_half`,
which fails when the work is done.

**Note on cellular state.** `provenance` still carries
`processing_stimulus_idx` and `apm_perturbation_idx` into the processing
latent. Under a strict reading of the forbidden list -- which names
"stimulation context" -- these should also be output tracks: predict elution
under each cellular condition rather than conditioning on the observed one.
That change interacts with the gap-2 fix, which deliberately routed cellular
state inward to give the in-vivo excision parameters gradient, so it needs its
own design pass rather than a mechanical edit.

