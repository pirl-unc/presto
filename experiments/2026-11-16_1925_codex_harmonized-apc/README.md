# Harmonized APC state: implemented and development-validated

Date: 2026-11-16. Agent/model: Codex / GPT-6. Extends Claude's existing state-bottleneck work.
Implementation commit: `23063189`; clean integration merge: `ada65722`, based on
`2d371d9e` and the canonical modular branch. Launches used that base plus the
uncommitted implementation; all 223 frozen runtime files match the subsequently
committed source byte-for-byte. The frozen per-file hashes are in
`reproduce/frozen-files.json`; actual source is retained in `reproduce/runtime/`.
The root checkout's unrelated changes were preserved.

The shared encoder is integrated in canonical `Presto`, the existing Context
schema, source adapter, training dependencies, input-support checks and checkpoint
format. It accepts component state/activity/expression, cell type, cytokines and
namespaced KO/KD/OE/transduction/variant/drug factors. One latent state feeds
peptide/flank supply and MHC-dependent register selection with direct-pathway class
masks. No fabricated expression fold changes or parallel flat APC route.
Intrinsic-affinity functions are frozen during these fits.

## Result and interpretation

| Model | Pooled AUROC | Class I AUROC | Class II AUROC | Pair win |
|---|---:|---:|---:|---:|
| Frozen canonical reference | 0.883057 | 0.912308 | 0.678372 | 0.875893 |
| Matched continuation, no APC state | 0.888882 | 0.911285 | 0.713403 | 0.883036 |
| Harmonized APC state | 0.896948 | 0.918001 | 0.739341 | 0.886607 |
| Interventions shuffled within study/cell/class | 0.896948 | 0.917937 | 0.740627 | 0.885714 |
| Whole APC context shuffled within class | 0.893278 | 0.914775 | 0.729846 | 0.884821 |

- Real APC minus Interventions shuffled within study/cell/class: AUROC +0.000001, 95% CI [-0.000518, +0.000513].
- Real APC minus Whole APC context shuffled within class: AUROC +0.003670, 95% CI [+0.002313, +0.004978].
- Real APC minus Matched continuation, no APC state: AUROC +0.008067, 95% CI [+0.005988, +0.010161].
- Real APC minus Frozen canonical reference: AUROC +0.013891, 95% CI [+0.007517, +0.020067].

Intervals use 1,000 paired source-protein cluster bootstrap resamples on the same
selected-checkpoint development predictions. Both classes improve in AUROC against
the whole-context shuffle: I +0.003226 [0.002017, 0.004615]; II +0.009494
[0.000975, 0.017782]. Within-pair ordering differences remain uncertain: pooled
real-minus-state-absent +0.003571 [-0.003452, 0.010519]. These intervals describe
this cohort; they do not include study-transfer, training-seed or model-selection
uncertainty. Checkpoint selection uses this validation set, so the estimates are
exploratory rather than independent confirmation.

**Decision:** retain the harmonized mechanism and this experimental checkpoint.
There is a context-associated presentation gain over state-absent and globally
shuffled controls. The within-study/cell/class permutation ties the real labels,
so these experiments do **not** establish an isolated gene- or drug-specific gain.
Cell type and other context shared within studies can explain the improvement.
No claim that the model now predicts calibrated B2M/TAP total-yield loss is made.
Canonical deployment/reference promotion is not established by this experiment.

## Exact evidence and partition contract

- Current Hitlist 2.1.9 checkout `43e226621434a483141764ed54faba725bbb3869` overlays
  current metadata on the existing immutable observation cache. Cache SHA256:
  `0b34bd12471882eb28d4f9a1dad6cc48252fb93a3644f912f69a0a186228f847`.
- The public `generate_training_table(include_evidence="ms")` export is retained
  as `results/source-inputs.parquet`, 437,234 rows for nine source PMIDs, filtered
  to available sequence proposals. Full input hashes and source dispositions are
  in `results/cohort-v2/cohort.json`; this is not an exhaustive use of all Hitlist.
- PMIDs: 31844290, 32161166, 32488085, 32938616, 33592498, 33968037,
  35561310, 36215666, 40113210. Final inputs include five cell types, IFNG and 21
  namespaced factors. Clean training contrasts include all eleven HAP1 KOs,
  ERAP1 KD, TAX1BP1 KD and class-I CIITA transduction. Class-II parental glioma
  background identifications are excluded, not converted to true negatives.
- Decitabine/tazemetostat/DMSO contexts are represented, but the drug validation
  subset is only six pairs. Palbociclib is not represented after this cohort's
  source/proposal/protection gates despite its wider Hitlist support. No claim of
  drug-specific predictive validation is made. Expression is software-supported;
  these fits have no matched expression or quantitative abundance targets.
- Admission uses `data.modular_hitlist_ms`: unique annotated arms; known modified
  peptidoforms excluded; exact-condition sample genotype recovery retains the
  whole candidate set; DP/DQ partners are not invented. DR beta-only sources use
  an explicitly recorded DRA*01:01 reference assumption.
- The old cache lacks indexed contributor IDs. An explicit hash-pinned legacy
  lineage policy preserves its stable MS measurement IDs and study scope without
  inventing contributor IDs. HAP1 additionally uses the explicit
  `sequence_only_unknown_ptm` policy; unknown modification status is retained.
- The previous Claude pool supplies only human reference peptide/flank/protein
  and decoy proposals. Its condition labels, predicted allele assignments and
  intervention severity values are discarded. Cross-study sequence reuse has
  zero conflicting contexts among 187,274 unique pool peptides. All biological
  conditions and candidate MHC sets come from the current source export.
- Each positive has one same-protein, same-length synthetic background peptide.
  These are **not affinity-matched** and are not measured biological negatives.
  All cached MS positives and protected related sequences are excluded from
  backgrounds. BCE target 1 means source-reported detection; target 0 means this
  declared sampled background. No processing/abundance/affinity label is imputed.
- Final cohort: 10,634 TRAIN rows (5,317 pairs); 2,240 VALIDATION rows (1,120 pairs,
  837 proteins). Validation includes 933 class-I and 187 class-II pairs.
  Stable source-protein partitions, known canonical-presentation training proteins
  confined to TRAIN, and exact/core8/Hamming-1 cross-partition exclusion.
- An identity-only list of 180,116 protected sequences is embargoed before fitting.
  Independent verification finds zero protected or cross-partition sequence-family
  matches and zero known canonical-presentation training-protein exposures in
  validation. This is not a guarantee against all historical study/pretraining
  exposure; several source studies were previously explored.
- Canonical TEST remains sealed. No TEST predictions or metrics exist here because
  this is iterative development. The frozen corpus's test labels were never opened.

## Training and controls

Parent is the canonical reference from PR154:
`2026-11-08_1800_codex_unified-parity-retrain/fits-v3/reference/raw/selected.pt`,
SHA256 `1d0d4122abf022830ea97fcc247b1d28c0645d7d42d206130cdf8a5d7762815a`.
Parent architecture is d_model=128, 2 Transformer layers, 4 heads, d_ff=512,
source flanks enabled, staged affinity-downstream elution. Harmonized state width=8.
No new pretraining or affinity supervision. All 49 intrinsic-affinity dependency
parameters remain exactly equal to the parent after training.

Every continuation uses the canonical manifest trainer: 800 AdamW steps, batch=32,
learning rate=3e-4, weight decay=1e-4, seed=42, class-balanced sampling, binary
elution BCE with weight 1, no synthetic affinity target, no identity replay because
its shared dependencies are frozen, and no state-contrast auxiliary loss.
Validation every 200 steps; select by equal-study validation loss. The explicit
`predictive` context-support policy admits one-sided predictors without creating
absent/control states. Input coverage still requires real updates and rejects
untrained active features in qualified inference.

Pilot v1: 6,177 pairs, same-publication sequence proposals. Expanded v2: 6,437 pairs,
sequence-only reuse across studies recovers ERAP1/TAX1BP1 control/treatment evidence.
Both are exploratory parts of one family, not independent replication. Each has
reference, state-absent, real APC and within-study/cell/class shuffled comparisons.
V2 also has the capacity-matched whole-context shuffle within MHC class, with
verified context histograms and shared context within each positive/background pair.

Requested hardware `H100!`; observed NVIDIA H100 80GB HBM3, 85,009,367,040 bytes;
Torch 2.14.1+cu130. Seven valid fits completed with passed training-coverage gates.
Runtime per fit is in `results/runtime.csv` (about 147–347 seconds excluding
collection and post-fit affinity evaluation).

One additional global-shuffle attempt is invalid: its preparation mutated donor
contexts through shared references. It completed because the initial cancellation
command required a noninteractive flag. It is preserved separately, excluded from
all comparisons, and the corrected control deep-copies contexts and verifies
histogram/pair invariance. See `results/global-invalid.json` and its launch log.

## Affinity, software and inference verification

All fits preserve 49 affinity parameter tensors exactly. On 2,536 validation-only
assay rows, maximum decoded prediction drift is 5.908819e-7 log10(nM), consistent
across all arms: floating-point execution differences, not bitwise-identical
predictions. `results/affinity-summary.csv` records exact Spearman/Pearson/RMSE,
500-nM threshold metrics and before/after values by assay and class; complete
censored/exact metrics and per-example predictions remain in each run directory.
For pooled exact IC50: n=1,339, Spearman 0.582370, Pearson 0.636942, RMSE 0.868126.

441 focused/related tests pass. An initial broad run had twelve missing sparse-
checkout fixture errors; restoring the committed fixtures and rerunning those
twelve passes. Repository lint and format checks pass (538 files).
`results/verification.json` independently checks partitions, protected sequences,
pairs, frozen tensors and the qualified `predict` API for both classes.

## Reproduction and artifacts

- `reproduce/launch.sh` dispatches fit/collect/analyze/verify with H100 explicit.
  Existing output locations refuse overwrite; use a fresh volume prefix for a
  new training replay. `reproduce/launch.json` and per-run manifests freeze inputs
  and settings; source snapshots and per-file hashes preserve the exact versions.
- Data-generation code is in `code/export_inputs.py` and `code/prepare.py`;
  `reproduce/source/prepare-v1.py` preserves the earlier cohort policy. The final
  full-context permutation is `code/make_global_control.py`.
- `results/analysis-v2-final/metrics.csv`, `paired-bootstrap.json` and
  `aligned-predictions.parquet` contain final comparisons; v1 results are retained.
- Checkpoint/prediction paths and hashes: `results/artifacts.json`.
  The preferred experimental checkpoint is
  `results/runs-v2/harmonized-s42/raw/selected.pt`.
- Raw runs on Modal volume `presto-checkpoints`: roots
  `2026-11-16_1925_codex_harmonized-apc` and
  `2026-11-16_1925_codex_harmonized-apc-v2`. Checkpoint selection, full metrics,
  predictions, coverage, admission and runtime receipts are retained there.
- Frozen input/source archive and checksum are recorded in
  `reproduce/archive.json`; it permits recovery if local preparation caches change. Remote readback matches
  the complete archive hash. Extraction into a neutral temporary directory and
  replay of the independent cohort/inference verification also pass
  (`results/portable-verification.json`).
- Hitlist curation requests: [tracker #905](https://github.com/pirl-unc/hitlist/issues/905),
  new #899–904 and evidence comments on #715/#853/#816. Those improvements are not
  prerequisites for this implemented mechanism.
