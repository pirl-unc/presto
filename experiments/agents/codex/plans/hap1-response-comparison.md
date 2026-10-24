# HAP1 response comparison: biological signal before another sampler extension

Status: next implementation plan, not a launched experiment. The source package
is independently checked; Hitlist PR806 and its full validation/release remain
pending. The previous PXD057588/native/MAPP test outcomes are closed to tuning.

## Question and scope

Does the new quantitative panel contain reproducible, peptide-dependent machinery
response beyond condition effects and protein abundance? First establish that
signal with simple controls and a shared source-linked recipe. Then compare exact
and sampled training under an identical defensible observation objective.

The target is MHC-eluted peptide response, conditional on the source observation
process. It is not free intracellular concentration. HAP1 is not a professional
APC. Most interventions affect transport/loading/selection; do not encode every
KO as a proteasome cleavage-rate change. A positive benchmark cannot independently
identify all latent processing, selection and detector components.

## Canonical recipe requirements

- Implement the recipe in Hitlist's existing registered EvidenceStore recipe
  interface. Presto consumes the resulting package; no raw-source importer or
  source curation in the model launcher. Pin package manifest88cad10d7bc2a638305cb0d37ad96ac2826c3badaf7b3e0aad7928820df5c4b8,
  source contracts and consumer commit, including the nullable-join correction.
- Preserve one source biological-replicate peptide cell as the observation. Do
  not multiply it into all possible control pairs. Quantities already average
  technical runs; retain their contributors and denominator policy. Additional
  DDA library biological labels are not additional DIA quantity observations.
- A response is log10 positive KO peptide intensity minus the mean log10 positive
  WT intensity for that peptide, requiring at least two measured WT biological
  replicates. Keep original numerator and all denominator cell IDs. Missing or
  zero values remain separate dispositions, without generated zero/bound labels.
- B2M is a selection/background control, not a free-peptide negative. Preserve its
  measured quantities and use the publication's explicitly recorded background
  exclusion as a separate flag; do not silently erase its source rows. Freeze the
  exact primary background-admission policy before fitting.
- Build molecular dependency components before eligibility filtering using every
  source peptide identity, reported parent claim and exact candidate sequence
  origin. Keep all alternatives and collapse identical parent sequences only for
  model input, retaining source contributors. Reserve a deterministic component
  test split before any predictive evaluation. These splits do not establish
  independent studies, clones, specimens or homology families.
- For a precursor-controlled population, require adequate measured positive WT
  and KO protein quantities for every candidate parent/group. Use source group
  identities and indistinguishable-protein claims, not a selected best parent.
  Protein abundance is a group-level surrogate; equal origin weighting remains
  an assumption. Do not fill supplement NA cells with positive deposited values.
  A separate deposited-report contract would need an explicit source-layer policy.
- Export admitted responses, all excluded/deferred source cells, original values,
  contexts, conditions, components, candidate origins, precursor contributors and
  counts. Independently verify response arithmetic and conservation before fitting.

## Initial model comparison

Use local CPU and fixed final checkpoints. Register the exact cohort, grouping,
features, transformations, losses, penalties and steps in a new timestamped
experiment before the first optimizer update. Initial controls are no change,
condition-only, calibrated precursor plus condition, and a small direct sequence
response model. Include length and terminal/adjacent-residue features with
training-only transforms; avoid peptide identity parameters. Keep known machinery
labels as supplied context, without claiming transfer to unseen knockouts.

Use grouped inner CV on the training partition to assess whether nonconstant
outputs generalize; leave the reserved outer test unscored during these diagnostic
baselines. Preserve both predictive spread and error relative to condition/input
controls. More training fit alone does not justify extra biological mechanisms.

After this source and baseline qualification, specify the physical availability /
transport / selection observation model. Reuse the existing exact processing
primitives where applicable; do not mislabel a direct response regressor as that
model. Compare sampled and exact updates on the same objective and data only when
the physical model shows useful development signal or a discrete observation
objective warrants sampling. The full APC availability goal remains unchanged.
