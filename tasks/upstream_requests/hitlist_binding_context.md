Presto needs source-reviewed binding-assay context to stratify heterogeneous IC50 measurements and avoid conflating IC50 with KD. Current `generate_training_table` preserves endpoint, units, inequality, quantitative value, method, assay/reference IDs and contributors. Those are sufficient to investigate the failed bounded fit; this request does not assert that missing context caused its almost constant predictions (pirl-unc/presto#117).

Please audit and expose structured assay-context fields where supported by the primary source: assay preparation/format, temperature and pH, incubation/equilibration conditions, competitor/tracer identity and concentration, tested concentration range and measurement limits, replicate/aggregation identity and uncertainty. Preserve reported values and units, source locator/citation and applicability; distinguish unknown, not applicable and explicitly measured absence. No imputation from a paper title or generic protocol. Separate reported IC50 from any derived KD and record a derivation only when its assumptions/inputs are evidenced.

Acceptance:
- Publish a coverage audit by endpoint/method/study, identifying what IEDB/CEDAR already expose versus what requires supplement curation. Start with the IC50 studies in the existing corrected human cohort; no new arbitrary sampling.
- Add a versioned sidecar keyed by evidence_row_id/provenance_id (or equivalent reviewed schema) and expose it through frozen exports without multiplying observation weights.
- Preserve original raw values/units/qualifiers, replicate grouping and source contributor links; conflicting source conditions remain explicit.
- Provide source-linked examples, round-trip/projection tests, counts of unknown/applicable fields and changed-row audit against human-cohort-corrected-v1.
- Keep metadata suitable for assay-stratified supervision/evaluation; identifiers are not predictive features.

Related: #678, #680, #706. This extends assay context and replicate provenance rather than duplicating donor/specimen-independence qualification. Current inspection: main 7d84da34 / Hitlist 1.82.11; latest published corrected human bundle has manifest b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372.
