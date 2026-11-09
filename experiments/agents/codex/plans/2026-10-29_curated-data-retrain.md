# Retrain the shared model on source-reviewed data improvements

2026-10-29; Codex/GPT-6. Family `2026-10-29_1956_codex_curated-data-retrain`.
User explicitly requested continued data improvements and retraining. Existing
authorization covers source/data upload and at most two concurrent H100! fits.

## Frozen intent before implementation

Use the current shared-v5 wide ten-output architecture and common observation
manifest/trainer. Start fresh paired seeds 42/43. Preserve the parent temporal
family's whole-study validation/test membership, measured values and generated
evaluation comparisons byte for byte. These holdouts have now been inspected;
comparisons are repeated fixed-holdout development evidence, not a new blind test.
Do not select changes from those test results.

### Data changes

1. Use the verified source-scoped curation output from
   `2026-10-29_1653_codex_measurement-curation-recovery`: 506 inferred inclusive
   bounds (482 ceilings/24 floors), 618 unresolved TRAIN observations withheld.
   Keep raw values, original qualifiers, source fingerprints and inference status.
   No universal or automatically transferred threshold. Original reported method
   and literal quantity remain distinct; unresolved protocol conflicts are flagged.
2. Independently verify the recovered Stražar source evidence and common-adapter
   output: 14,187 reviewed observations, 289 excluded by the current nine-residue
   MHC-II minimum, 348 held-out peptide exclusions, 13,550 eligible TRAIN positives.
   Check actual IDs, measurements, source forms and partition overlap, not counts
   alone. Check every recovered sequence against the parent's positive blacklist.
3. Supply one generated unobserved comparison per new TRAIN positive using the
   existing sampler, seed, pinned proteome, length/context matching and positive/
   heldout exclusions. These are not measured nonbinders. Existing backgrounds and
   heldouts remain immutable. If a newly known positive conflicts with an existing
   background, stop and record a cohort amendment before fitting.
4. Preserve old training observations apart from the declared qualifier/removal
   delta; append the recovered study and its comparisons. New downloaded archives
   are not automatically admitted: finish source/assay/PTM/overlap review first.
   Acquire more evidence independently while the frozen fit runs; later additions
   require another frozen cohort rather than mutating active inputs.

### Training and selection

Same architecture 192/3/6/768, dropout .1, batch128, AdamW3e-4/decay1e-4,
constant learning rate, no identity pretraining and no additional losses.
Retain 60% elution /20% literal IC50 /15% other physical /5% qualitative sampling,
with square-root endpoint/class allocation inside each family. Recompute common
weighted quotas to cover every new TRAIN row at least once; record exact new
update count and label exposures before launch. This is the same complete-epoch
policy, not an exactly matched number of optimizer updates.

Two H100! workers maximum, with actual hardware/memory logged. Validate every
1,000 updates, durable checkpoint every500, select minimum equal-endpoint
validation loss, seed as tie-break. Report selected versus completed exposure.
No test-based selection. Preserve all outputs and failed attempts.

### Execution and gates

- [ ] Resolve upstream Hitlist release conflict, retain source-qualified wheel
  hashes and verify common peptidoform export. The tested premerge wheel remains
  a valid reproducible source for the already prepared recovery; do not silently
  substitute a release with different bytes.
- [ ] Verify the curation/recovery artifacts and prepare/hash the combined cohort
  on CPU; freeze membership, quotas, scalers and exact runtime snapshot.
- [ ] Run focused source/admission tests and the canonical full training preflight.
- [ ] Launch paired H100! fits and expose real step/validation progress promptly.
- [ ] Independently recompute final per-output/class/era regression and ligand
  metrics from per-example validation/test dumps, including Spearman/Pearson/RMSE,
  exact/bounded loss, meaningful500nM classification and calibration.
- [ ] Compare with parent seeds side by side; source-reported nondetections and
  generated same-protein/proteome comparisons stay separate. Include failure
  strata, gradient/update/feature coverage and register diagnostics.
- [ ] Close family README, reproducibility bundle and canonical experiment log;
  retain null/negative results and source coverage limits. Promote only with
  supported validation benefit, never merely because training finished.

## Upstream adjustment

Hitlist main advanced from2.1.0 to2.1.2 while PR872 checks ran. Its2.1.1 bump
conflicts; merge main into the feature branch and bump to2.1.3, rerun checks,
then merge and run the release gate on the actual merged commit. A redundant
release workflow accidentally dispatched on old main was identified and canceled;
it does not validate this patch. Preserve the existing recovery's2.1.1 wheel pin.

## Cohort amendment before retry / any optimizer update

Independent CPU audit found 292 reviewed positive sequences absent from the old
positive blacklist (192 among eligible recovered sequences). They intersect 35
generated TRAIN rows and three generated TEST rows; all three TEST rows share
one sequence. One otherwise eligible recovered observation therefore needs an
additional heldout embargo. Preserve the failed preparation and its source under
`attempts/blacklist-assumption/`; retry as `source-review-v2`.

- Expand the sampler blacklist with all 11,100 source-verified positive sequences,
  including those outside current architectural admission.
- Remove exactly the independently identified 35 generated TRAIN comparisons;
  preserve all other old nonphysical TRAIN row bytes/order. These sequences are
  known ligands, not necessarily demonstrated binders for each comparator's MHC.
- Retain 13,549 recovered TRAIN positives and generate 13,549 comparisons.
- Keep every validation/test byte fixed. Report the three suspect generated TEST
  labels explicitly, and calculate an additional sensitivity analysis excluding
  their IDs for both parent and new predictions. It is not a newly blind test.
- Independently assert all affected IDs, counts, file hashes and disposition
  changes; recompute quotas, scalers and source manifests before launching.

No model training took place under the failed preparation. This amendment uses
source evidence and membership only, with no new model predictions available.

## Durable post-run closure

Before any new predictions are available, extend the paired-comparison analyzer
to update this family's README and canonical registry only after collection,
prediction verification, cohort census and full reporting all pass. It must retain
the frozen recipe, append actual validation/test metrics and hardware/exposure,
link the matched-parent and suspect-label sensitivity analyses, and state that
validation selection does not automatically promote a model. This allows the
existing continuation to close the experiment even across client interruptions.
## Closeout verification amendment before new validation results

Keep initial pre-prediction analyzer snapshots unchanged. Verify a second reporting
revision using historical parent artifacts only in a temporary directory: binary
tables must show the total evaluated count, terminal reports must remove stale
pending wording, failed gates must not publish completion, repeated closeout must
be idempotent, and unrelated registry entries must remain byte-identical. Freeze
the revised analyzer and verifier separately before using new prediction outputs.
