# Shared APC context for presentation

The canonical `Presto` model supports a shared APC state encoder through
`state_mode: harmonized` and a positive `state_bottleneck` dimension. It uses the
existing `Context` schema for both MHC classes:

- `components`: categorical machinery state and optional measured activity or RNA
  abundance, with declared units and provenance;
- `cell_type`: supplied, inferred or overridden APC identity;
- `cytokines`: present/absent/unknown exposure, optionally dose and duration;
- `interventions`: namespaced gene and chemical factors, including
  `knockout:B2M`, `knockdown:ERAP1`, `transduction:CIITA`,
  `overexpression:ERAP2`, `genetic_variant:TAPBPR:M29`, and `drug:decitabine`.

An intervention's `present` means that the intervention was applied. It never
means the target protein is present. KO, rescue and variant assertions remain
separate concurrent factors; unspecified expression changes are not numeric
fold changes. Unknown assertions encode no evidence and are distinct from an
explicitly absent intervention. Reference RNA/activity carries a separate
indicator from directly supplied measurements. TPM uses log1p scaling.

The same encoder produces a latent state after a class-specific direct-pathway
mask. B2M/TAP and class-I HLA factors route to I; CIITA and class-II HLA/loading
factors route to II. Unclassified exposures may affect both. These are architectural
scope priors, not measured effect sizes, and do not describe every indirect or
cross-presentation effect. The state conditions a peptide/flank supply score and
an MHC-dependent, per-register selection score. It does not enter the intrinsic
binding output. The old flat APC inputs are removed from the staged presentation
processing head in this mode, so that route cannot bypass the biological masks.

A minimal model fragment is:

```json
{
  "elution_staging": "affinity_downstream",
  "elution_affinity_endpoint": "ic50",
  "state_mode": "harmonized",
  "state_bottleneck": 8,
  "interventions": ["knockout:B2M", "transduction:CIITA", "drug:decitabine"],
  "cytokines": ["IFNG"],
  "cell_types": ["Myeloblast", "Glial cell"]
}
```

Allocate vocabularies from training evidence. Current checkpoints record feature
exposure and real parameter updates; prediction support checks reject active
features that were never trained. Selecting the same vocabulary does not confer
support for an unseen intervention state or justify a causal claim.

## Source admission

`data.modular_hitlist_ms` is the common I/II adapter. It admits uniquely resolved
condition metadata and keeps unassigned or ambiguous arms unknown. Explicit
`none` asserts absence only within the same condition family; a blank is not a
control. Condition/control references are retained as provenance, not model inputs.

`sample_candidate_set` can recover a missing MHC candidate set from a uniquely
joined, exact-condition sample record. The caller must establish uniqueness and
provide the source reference. All typed candidates are retained; DP/DQ chains
are never paired speculatively. DR beta-only sources require an explicit DRA
reference assumption at `ms_observation` admission.

Strict admission remains the default. For older caches, an explicit
`legacy_source_reference` allows stable MS row/source IDs plus study-scoped,
hash-pinned export lineage. It does not fabricate an indexed contributor ID.
`unspecified_modification_policy="sequence_only_unknown_ptm"` allows sequence-only
training with unknown modification status; known modified peptidoforms remain
excluded. This is unsuitable for learning drug-adduct specificity. Both policies
remain recorded on each observation.

The trainer defaults to requiring contrasted context states. Explicit
`context_support_policy: predictive` permits observed, one-sided categorical
context as a predictor without claiming an isolated intervention effect. It still
requires a known training state, and does not synthesize missing controls. Numeric
and component-state contrast checks retain their stricter rules.

## Transfer and validation

An affinity-preserving continuation can use the existing pretrained `freeze`
dependency targets. `allow_apc_expansion: true` allows harmonized inputs to be added
to a parent with no allocated APC context. The new supply and selection readouts
start at zero, preserving the parent's initial presentation function. The option
refuses a parent with existing APC weights that would otherwise be discarded.
Existing structured/free state checkpoints remain readable.

Use the canonical manifest trainer and checkpoint format. Compare against an
otherwise matched state-absent continuation and condition-shuffled control, save
per-example predictions, and report class/source-stratified and within-pair metrics.
Conditional peptide identification cannot establish total presentation abundance:
large B2M/TAP effects on yield need calibrated measurements or qualified detection
panels. Paired protein-level bootstrap uncertainty describes this development
cohort; it does not establish transfer to a new study.

Current upstream curation requests are tracked in
[Hitlist #905](https://github.com/pirl-unc/hitlist/issues/905). The implementation
uses currently resolvable evidence and does not depend on those requests closing.
