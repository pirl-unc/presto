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
reads peptide and MHC, and every assay configuration and cellular condition is
an *output track*. Track identity selects which output the loss reads; it never
selects what the model computes.

**Verified as observable invariance**, not by reading the source --
`tests/test_many_output_contract.py` asserts the prediction does not move when
each forbidden input is supplied:

| forbidden input | status |
|---|---|
| `binding_context` (assay type, prep, geometry, readout) | argument ignored; input-side embeddings deleted |
| `tcell_context` (7 keys) | arguments removed from `TCellAssayHead.forward` |
| cellular state (APM perturbation, stimulation context) | swept, not indexed; no input embedding exists |

**Output tracks:**

| family | tracks |
|---|---|
| binding observables | `KD`, `IC50`, `EC50`, `Tm`, `t_half`, `koff`, `kon` |
| binding assay descriptors | `binding_assay_panel_*` — 11 types, 6 preps, 5 geometries, 4 readouts |
| T-cell conditions | `tcell_panel_logits` — method, readout, APC type, culture, stim, format |
| cellular state | `excision_panel_apm` (7), `excision_panel_stimulus` (6) |
| TCR | `tcr_evidence_method` |
| MHC | `mhc_class`, `mhc_species`, `mhc_a_fine_type`, `mhc_b_fine_type` |

**What is still an input, and why.** `peptide_source_idx` and
`enzymatic_digest_idx` select which branch computes the peptide termini --
in-vivo proteasomal versus in-vitro protease. That is structural routing, not a
condition to predict: a shotgun row and an MHC elution row are different
experiments, not the same experiment under different settings. `species` (the
host organism) is likewise a biological input. `mhc_species` and
`species_of_origin` are *predicted* and are not fed.

**Not represented at all.** `ElutionRecord.cell_type` and `tissue` never reach
`PrestoSample`, so the source cell line -- 172 distinct lines, and the design's
intended proxy for expression profile -- is neither an input nor an output
track. TCR sequences are likewise absent: `TcrEvidenceRecord` contributes only
a binary "some receptor was found" label. Both are gaps, not deviations.

**Gap 2 under the new design.** The in-vivo excision profiles still receive
gradient, which was gap 2's entire content, but by a different route: the head
sweeps `invivo_profile_c/n`, `stimulus_profile_c` and `invivo_bias` across all
conditions and the observed condition selects the supervised column. Pinned by
`tests/test_provenance_fork.py::TestInVivoGradient`.
