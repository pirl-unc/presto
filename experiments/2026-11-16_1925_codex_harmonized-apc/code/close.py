from pathlib import Path
import csv,json,hashlib,shutil
import pandas as pd
E=Path(__file__).resolve().parents[1];ROOT=E.parents[1]
metrics=pd.read_csv(E/'results/analysis-v2-final/metrics.csv');ci=json.loads((E/'results/analysis-v2-final/paired-bootstrap.json').read_text());cohort=json.loads((E/'results/cohort-v2/cohort.json').read_text())
names={'reference':'Frozen canonical reference','state_absent-s42':'Matched continuation, no APC state','harmonized-s42':'Harmonized APC state','shuffled-s42':'Interventions shuffled within study/cell/class','shuffled_global_fixed-s42':'Whole APC context shuffled within class'}
def metric(model,group,key):return float(metrics[(metrics.model==model)&(metrics.group==group)].iloc[0][key])
lines=['| Model | Pooled AUROC | Class I AUROC | Class II AUROC | Pair win |','|---|---:|---:|---:|---:|']
for key,name in names.items():lines.append(f'| {name} | {metric(key,"pooled","auroc"):.6f} | {metric(key,"class_I","auroc"):.6f} | {metric(key,"class_II","auroc"):.6f} | {metric(key,"pooled","pair_win"):.6f} |')
table='\n'.join(lines)
intervals='\n'.join(f'- Real APC minus {names[r["comparator"]]}: AUROC {r["difference"]:+.6f}, 95% CI [{r["ci95"][0]:+.6f}, {r["ci95"][1]:+.6f}].' for r in ci if r['group']=='pooled' and r['metric']=='auroc')
runtime=[];artifacts=[]
for family,folder in [('pilot','runs'),('expanded','runs-v2')]:
 for p in sorted((E/'results'/folder).glob('*/raw/summary.json')):
  d=json.loads(p.read_text());model=p.parents[1].name;hardware=json.loads((p.parents[1]/'hardware.json').read_text());selected=p.parent/'selected.pt'
  runtime.append({'cohort':family,'model':model,'steps':d['steps'],'seconds':d['elapsed_seconds'],'coverage':d['coverage_gate']['status'],'requested_gpu':hardware['requested_gpu'],'observed_gpu':hardware['name'],'memory_bytes':hardware['memory_bytes']})
  artifacts.append({'cohort':family,'model':model,'checkpoint':str(selected),'sha256':hashlib.sha256(selected.read_bytes()).hexdigest(),'modal_volume':'presto-checkpoints','modal_prefix':'2026-11-16_1925_codex_harmonized-apc'+('-v2' if family=='expanded' else ''),'prediction_files':[str(p.parents[1]/n) for n in ('validation-predictions.json','affinity-predictions.json')]})
pd.DataFrame(runtime).to_csv(E/'results/runtime.csv',index=False);(E/'results/artifacts.json').write_text(json.dumps(artifacts,indent=2))
readme=f'''# Harmonized APC state: implemented and development-validated

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

{table}

{intervals}

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
  `reproduce/archive.json`; it permits recovery if local preparation caches change.
- Hitlist curation requests: [tracker #905](https://github.com/pirl-unc/hitlist/issues/905),
  new #899–904 and evidence comments on #715/#853/#816. Those improvements are not
  prerequisites for this implemented mechanism.
'''
(E/'README.md').write_text(readme)
entry=f'''\n\n## 2026-11-16_1925_codex_harmonized-apc — shared APC state, both MHC classes\n\nDate: 2026-11-16. Agent/model: Codex / GPT-6, extending Claude's state-bottleneck code.\nDirectory: [harmonized APC](2026-11-16_1925_codex_harmonized-apc/README.md).\nImplementation `23063189`, base `2d371d9e` dirty at launch; frozen runtime files match\nthe committed source. Reproduction: directory `reproduce/launch.sh`, launch JSON,\nsource snapshots, per-run manifests and input archive. No non-default environment\noverrides beyond explicit H100!, frozen PYTHONPATH and thread settings.\n\nCurrent Hitlist 2.1.9 metadata over the pinned older cache; nine PMIDs, source-resolved\nconditions and current MHC candidate sets. Explicit legacy lineage and HAP1 unknown-\nPTM sequence-only policy, no fabricated controls. Same-protein/length synthetic\nbackgrounds, not measured negatives or affinity-matched decoys. Protein-separated\nTRAIN 10,634 rows and VALIDATION 2,240 rows; 180,116 protected identities embargoed,\nexact/core8/Hamming-1 separation independently verified. No quantitative abundance\nor matched expression targets. Detailed source counts/policies/hashes in the README.\n\nCanonical PR154 reference parent; no new pretraining. Staged binary elution loss\nweight 1; all 49 affinity dependency tensors frozen. Shared state width8 routes\ncell/expression/cytokine/gene/drug features through class masks to supply and\nMHC-dependent selection. 800 steps, batch32, AdamW3e-4, decay1e-4, seed42, class\nbalance and equal-study validation-loss checkpoint selection. Requested H100!,\nobserved H10080GB HBM3 (85,009,367,040 bytes). Seven valid fits, 147–347 seconds\ntraining each; one invalid shuffle attempt is retained/excluded with explanation.\n\n{table}\n\nReal minus state-absent AUROC +0.008067 [0.005988,0.010161]; versus whole-context\nshuffle +0.003670 [0.002313,0.004978], paired protein bootstrap. Within-study/cell\nintervention shuffling ties, so isolated gene/drug benefits are not established.\nWithin-pair ordering changes remain uncertain. Preferred experimental condition:\nharmonized; retain the mechanism, with no production/reference promotion claim.\n\nValidation-only affinity check: 2,536 rows, max decoded drift5.908819e-7 log10(nM),\nexact parameter equality. Full Spearman/Pearson/RMSE and500-nM threshold metrics\nare in affinity-summary.csv and per-run JSON. Software:441 tests, lint/format pass,\nqualified inference checks pass. Per-example presentation and affinity validation\npredictions retained locally and on Modal (artifact paths/hashes in artifacts.json).\nTEST stays sealed because this remains iterative development; no TEST metrics or\npredictions were produced. This is a within-cohort selected-model result, not\nindependent study-transfer or causal severity confirmation.\n'''
(E/'results/registry-entry.md').write_text(entry)
for root in (ROOT,Path('/private/tmp/presto-harmonized-apc-20261116')):
 p=root/'experiments/experiment_log.md'
 if '## 2026-11-16_1925_codex_harmonized-apc —' not in p.read_text():
  with p.open('a') as f:f.write(entry)
print('Wrote README, runtime, artifacts, affinity closure and registry entries')
