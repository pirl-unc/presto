# Shared presentation preparation and training

Both MHC classes use the native `data.modular_hitlist_ms.ms_observation` adapter.
Ordinary presentation observations do not require a resolved intervention arm:
unavailable APC state stays unknown. Source-specific curation still determines
which outcomes, peptide modifications and typed MHC candidate sets are admissible.
Single reported/curated row cell identities are retained independently of treatment
arm resolution. Missing, placeholder or multivalued cell identities remain unknown;
an unresolved arm never supplies intervention or cytokine assertions.

`data.presentation` provides exact reference occurrence mapping, protein-family
unions, source-junction ambiguity handling, same-length possible-source backgrounds,
and exact/shared-8mer/Hamming-1 sequence protection. Mapping considers every exact
reference occurrence without a hit cap. A possible source protein is not a measured
protein assignment. Unanimous junctions retain their known flanks; disagreements
are masked without discarding the positive outcome. Noncanonical reference flanks
are unknown. A generated background is an unobserved comparison, not a measured
nonbinder. The caller must pin reference versions, source identities and partitions,
exclude known observed and protected peptides, and independently audit final splits.

The canonical manifest trainer supports these opt-in controls:

- `balance_by` accepts `study`, using explicit `provenance.study_id` or source PMID.
  Protein `group_id` remains available for disjoint partitions; it is not relabeled
  as a study to achieve sampling balance. Missing study provenance fails.
- `sampling_mode: weighted_steps` draws only the remaining finite update budget
  from explicit `stratum_weights`, cycling shuffled examples within each stratum.
  It requires `max_steps` and explicit weights, and rejects a changing sampling
  schedule. The complete chosen order and RNG state retain exact resume behavior.
  A bounded update budget does not imply a complete traversal of every example.
- `selection_endpoint: elution_observation` with
  `selection_reduction: class_study_mean` selects the arithmetic mean of study
  losses within each class, then the arithmetic mean across configured classes.
  Every configured class and every evaluated selected-endpoint record is required;
  ambiguous class assignment, missing study identity or nonfinite losses fail.
  Binding losses remain reported separately and do not numerically dominate this
  presentation selection criterion.

Historical defaults keep their original weighted-epoch/selection semantics and
training traces. Declare source sizes, sampling frequencies and selection weights
separately; they describe different aspects of training. Report per-class metrics
and paired controls, including affinity retention, before promoting a checkpoint.
A functioning shared APC input path does not itself demonstrate learned intervention
effects or predictive gains.

## Unrepresented APC categories

Checkpoint vocabulary is derived from TRAIN only. The default public predictor remains
strict. For evaluation or deployment with new reported categories, explicitly use
`predict(rows, unsupported_apc_policy="unknown")`. The central
`data.modular_context.project_apc_context` retains supported claims and projects
unrepresented categorical claims to unknown; omitted source assertions appear in
`unrepresented_apc_context` provenance and prediction output. An unsupported override
does not fall back to the superseded supplied identity. Supported dose/unit and ordinal
reference validation stays strict. Apply the projection to original inputs separately
for each checkpoint; projection does not learn a vocabulary or imply an absent exposure.
# Optional pathway-conditioned source processing

`model.elution_pathway_conditioning: true` enables a shared processing MLP whose
hidden activation is conditioned on class I or class II. A zero-initialized
`2 → d_model` map adds the pathway input before the shared GELU. The source
encoder, flank representation, processing MLP and output remain shared. This
adds256 weights at width128 and does not give the intrinsic affinity decoder an
APC or pathway-processing input.

The option requires staged elution and is off by default. Default checkpoint
serialization and predictions are unchanged. Adding it to a warm start requires
`pretrained.allow_pathway_expansion: true`; removal of an existing trained
pathway module is rejected. Each class input needs actual gradient/update
evidence for qualified inference. The new map starts at zero, preserving parent
outputs and common-parameter initialization; all modules remain trainable.

This is an experimental capacity option, not a demonstrated predictive gain.
The original expanded-data experiments keep their frozen source and selection
contract. Its controlled validation is specified in
`experiments/agents/codex/plans/2026-11-17_shared-pathway-processing.md`.
