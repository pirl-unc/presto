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
