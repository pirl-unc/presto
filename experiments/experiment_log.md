## 2026-10-16 — Molecular MHC species and independent source-role audit

- **Agent/model:** Codex / GPT-6; [experiment and reproduction](2026-10-16_1440_codex_molecular-species-lineage/README.md).
- **Recipe:** clean `ec0446fcb820d72d0041067cbee6074882837b3f`, Presto 0.1.19 candidate;
  hash-pinned native Hitlist 1.78.0 corpus and historical Presto 0.1.16 membership/
  inspection tables. All 9,053 observations and 7,241/906/906 assignments preserved.
- **Conditions:** data-only canonical label and native provenance comparison; no model,
  pretraining, synthetic data, training, loss/weight change or new predictions. All splits
  inspected for source integrity, no new held-out performance estimate or preferred model.
- **Result:** eight rat-host/HLA elution labels (seven train, one validation) change from
  host-derived murine to molecular human; all 2,614 IC50 values/operators/partitions and
  effective targets remain unchanged. Native context remains 50 resolved, 47 partial,
  8,956 unknown. Reviewed presenters include 15 rat rows; distinct source roles retained.
- **Resources:** CPU only, no cloud/GPU; builder 128.576 s, peak process RSS 631,259,136 bytes.
- **Decision:** bind the prospective molecular policy, preserve native role evidence and
  [publish the full source-linked audit](https://github.com/pirl-unc/presto/releases/tag/molecular-species-lineage-audit-v1).
  Processing conditioning, antigen attribution, unknown biological lineage and failed
  IC50 quality remain open. Historical run and qualification contract are unchanged.

## 2026-10-16 — Physical affinity reporting audit; failed quality decision unchanged

- **Agent/model:** Codex / GPT-6; [experiment and reproduction](2026-10-16_1405_codex_physical-affinity-metrics/README.md).
- **Recipe:** clean `9a8de46dee32dbe626153f7b7442fbe8c8ce0e1f`, Presto 0.1.18 candidate;
  hash-verified public 0.1.16/Hitlist 1.78.0 run, all 90 archive files and 2,614 IC50
  observations (train/validation/test 2,074/265/275). Original source contexts,
  raw nM values/censor operators, predictions and frozen cohort membership retained.
- **Conditions:** data-only replay of every original predictor in effective and raw
  target spaces; no training, pretraining, synthetic data, loss or checkpoint changes.
  Original AdamW 200-step canonical fit and mean/ridge baselines remain the contract.
- **Result:** four exact 500 nM training labels corrected; six constant-mean Pearson
  entries now undefined. All 20 comparisons retain exact RMSE/MAE/Spearman. Validation/
  test classifications unchanged; Presto effective RMSE 1.165477/1.310036 versus ridge
  1.037626/1.138279. Selected training threshold accuracy changes .630918 to .628986.
  Canonical summaries use physical nM for all classifiable observations, no fallback.
- **Resources:** CPU only, no GPU/cloud; builder .835 s, process peak RSS 623,738,880 bytes.
- **Decision:** quality screen still fails, ridge still preferred. This additive
  [public audit](https://github.com/pirl-unc/presto/releases/tag/affinity-metric-numerics-audit-v1)
  preserves the historical run. Shared studies, unknown biological identities,
  species-role ambiguity and narrow predictions remain limitations; #53 remains open.

## 2026-10-16 — Frozen effective-target IC50 fit: finite fitting, failed quality screen

- **Agent/model:** Codex / GPT-6; [experiment and full metrics](2026-10-16_1030_codex_effective-target-fit/README.md).
- **Source/reproduction:** clean `1067714c768af2e4f3184154c85903bb3c46cf4b`, released Presto 0.1.16 / Hitlist 1.78.0;
  unchanged native manifest `b180eba8…`, all 9,053 source-linked observations and 7,241/906/906
  frozen peptide-component partitions; published prospective effective-target contract passed
  before model construction. All raw/effective values, units, censor operators and citations retained.
- **Training:** fresh d32/l1/h4 expanded, 200 shuffled batches ×64, AdamW .001/.01,
  5% warmup/cosine, uncertainty weighting, no pretraining/synthetic examples or specified priors.
  Active measured families optimized; only IC50 declared. Full aggregate validation selects epoch 1;
  fresh checkpoint reload, every validation/test prediction and complete loss ledger retained.
- **Conditions:** one Presto fit; train-only exact-IC50 mean and alpha=1 length/composition/class
  ridge comparators. Effective float32 clipped targets primary; raw physical targets diagnostic.

| Split | Predictor | Effective RMSE | Raw RMSE | Effective Spearman |
| --- | --- | ---: | ---: | ---: |
| val | Presto | 1.165477 | 1.173963 | 0.363285 |
| val | Train mean | 1.137633 | 1.149279 | undefined |
| val | Ridge | 1.037626 | 1.047534 | 0.415286 |
| test | Presto | 1.310036 | 1.320948 | 0.337271 |
| test | Train mean | 1.248686 | 1.263942 | undefined |
| test | Ridge | 1.138279 | 1.151585 | 0.393923 |

- **Fitting:** 200 finite steps, no nonfinite tracked elements; direct gradients every batch;
  exact training RMSE 1.607139→1.337708. Complete validation/test loss .499515671/.549559624.
  Selected predictions vary by less than .03 log10(nM); physical 500 nM binder recall is zero.
- **Decision:** quality screen fails; validation paired RMSE intervals versus mean
  [-.075366,.124810], ridge [.020137,.239649]. Ridge preferred; test is descriptive.
- **Resources:** CPU-only, 8 CPU/64 GiB/512 GiB scratch, zero GPU/workers, one Torch thread;
  4489.710s, runner peak RSS 14,530,875,392 bytes. No budget extension.
- **Evidence/limits:** [complete downloads](https://github.com/pirl-unc/presto/releases/tag/effective-target-fit-qualification-v1); 90 archive files, source joins,
  metrics, baselines, paired bootstrap and losses independently verified. Shared studies,
  unknown donor/specimen/acquisition identities; 220 other outputs undeclared. Additive native
  species audit exposes eight rat-host/HLA elution rows (#115); numerical errata #116 preserve
  original results; #117 owns prospective diagnosis of near-constant predictions. #53 stays open.

## 2026-10-16 — Corrected human IC50 target-policy audit; fit stopped before optimization

- **Agent/model:** Codex / GPT-6; [experiment directory](2026-10-16_0840_codex_corrected-human-fit/README.md).
- **Source/reproduction:** clean recipe `124c85b6da23d4c74dbef961999936c573025192`, tag
  `experiment-corrected-human-fit-20261016`; actual released Presto 0.1.13 / Hitlist 1.78.0,
  unchanged corrected native source, verified MHC catalog and all 9,053 frozen observations
  (7,241/906/906). Full config/source/pins and manual preservation are in `reproduce/`.
- **Finding:** canonical IC50 targets apply `log10(clip(nM,0.001,50000))` while the prospective
  plan required original log10(nM). Of 2,614 direct IC50 observations, 211 values/bounds
  change: train 157 (39 exact), validation 31 (8 exact), test 23 (7 exact). All 2,614 source
  joins verify; raw values, units, operators, original/resolved MHC, cell/condition and citations
  remain in the full inspection table. #110 tracks canonical policy disclosure and qualification.
- **Run/conditions:** CPU-only (8 CPU, 64 GiB RAM, 512 GiB scratch, one Torch/BLAS thread,
  zero loader workers); no GPU. Application started 08:56:49 UTC, preservation receipt
  09:07:24 UTC. No complete runner runtime or peak-RSS receipt. One attempt, manually
  stopped during admission before optimization; 25 original files preserved. Buffered log
  limitations are explicit. No manifest rebind or criterion change.
- **Planned, unexecuted optimization:** fresh expanded d32/l1/h4 full profile, two epochs ×
  at most 100 batches, shuffled batch 64, learning rate .001, weight decay .01, uncertainty
  weighting; all explicit regularization/synthetic ratios zero, all observed families,
  only direct IC50 claimed. Complete validation would select the checkpoint, followed by
  fresh-model validation/test evaluation. No pretraining or synthetic examples.
- **Validation/test metrics:** unavailable: zero checkpoints, predictions or optimization
  results. No preferred model; baseline arithmetic was validated only on training labels.
  This is a label-policy audit, not a trained-model result. A tested relaunch guard now
  rejects the mismatch before model work; the original source snapshot remains unchanged.
- **Decision:** publish the preserved stop and source-linked raw/effective audit; require
  explicit policy qualification and a separate prospective family before replacement fitting.
  Shared studies and unknown biological lineage remain limitations. [Public audit artifacts](https://github.com/pirl-unc/presto/releases/tag/corrected-human-target-policy-audit-v1)
  include the complete table, counts, plot, source recipe, archive inventory and hashes.


### 2026-03-15_1226_codex_exp21-seed-epoch-confirmation
- **Agent**: codex
- **Dir**: [2026-03-15_1226_codex_exp21-seed-epoch-confirmation](experiments/2026-03-15_1226_codex_exp21-seed-epoch-confirmation)
- **Source script**: `experiments/2026-03-15_1226_codex_exp21-seed-epoch-confirmation/code/launch.py`
- **Question**: Which legacy v6 benchmark family is strongest across seeds and epoch budgets on the 2-allele broad-numeric contract?
- **Dataset**: `data/merged_deduped.tsv`, `HLA-A*02:01` / `HLA-A*24:02`, `numeric_no_qualitative`, qualifier filter `all`
- **Training**: `v6` legacy distributional benchmark, `50/100/200` epochs, seeds `42/43/44/45`, requested GPU `H100!`
- **Result**: Within this legacy assay-conditioned benchmark family, `groove c02` at `50` epochs won.
  - mean test Spearman across seeds: `0.847549`
  - best single run: `dist-ba-v6-confirm-groove_c02-c02-cc0-e050-s43`, test Spearman `0.854139`
  - `50 > 100 > 200` on the primary metric
- **Historical note**: Later honest repeat [2026-03-16_2142_codex_exp21-honest-no-assay-repeat](experiments/2026-03-16_2142_codex_exp21-honest-no-assay-repeat) showed that this benchmark family depends on assay embeddings from `binding_context`. It should therefore be treated as a historical assay-conditioned benchmark, not the canonical no-assay-input baseline.

### 2026-03-15_1011_codex_exp16-mainpath-baseline-rebuild
- **Agent**: codex
- **Dir**: [2026-03-15_1011_codex_exp16-mainpath-baseline-rebuild](experiments/2026-03-15_1011_codex_exp16-mainpath-baseline-rebuild)
- **Source script**: `experiments/2026-03-15_1011_codex_exp16-mainpath-baseline-rebuild/code/launch.py`
- **Status**: launched
- **Dataset**: `{"measurement_profile": "numeric_no_qualitative", "panel": ["HLA-A*02:01", "HLA-A*24:02"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "split": "peptide_group_80_10_10_seed42"}`
- **Training**: `{"batch_size": 256, "config_version": "v6", "epochs": 50, "gpu": "H100!", "lr": "1e-3", "seed": 42, "warm_start": false, "weight_decay": 0.01}`
- **Tested**: `[{"cond_id": 1, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 2, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 3, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 4, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 5, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 6, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 7, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 8, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 9, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 10, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 11, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 12, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 13, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 14, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 15, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 16, "content_conditioned": false, "encoder_backbone": "historical_ablation"}, {"cond_id": 1, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 2, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 3, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 4, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 5, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 6, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 7, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 8, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 9, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 10, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 11, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 12, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 13, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 14, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 15, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 16, "content_conditioned": true, "encoder_backbone": "historical_ablation"}, {"cond_id": 1, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 2, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 3, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 4, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 5, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 6, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 7, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 8, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 9, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 10, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 11, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 12, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 13, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 14, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 15, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 16, "content_conditioned": false, "encoder_backbone": "groove"}, {"cond_id": 1, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 2, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 3, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 4, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 5, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 6, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 7, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 8, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 9, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 10, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 11, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 12, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 13, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 14, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 15, "content_conditioned": true, "encoder_backbone": "groove"}, {"cond_id": 16, "content_conditioned": true, "encoder_backbone": "groove"}]`

### 2026-03-13_1445_codex_clean-distributional-ba-heads
- **Agent**: codex
- **Dir**: [2026-03-13_1445_codex_clean-distributional-ba-heads](experiments/2026-03-13_1445_codex_clean-distributional-ba-heads)
- **Source script**: `experiments/2026-03-13_1445_codex_clean-distributional-ba-heads/code/launch.py`
- **Status**: launched
- **Dataset**: `{"assay_families": ["IC50", "KD", "KD (~IC50)", "KD (~EC50)", "EC50"], "measurement_profile": "numeric_no_qualitative", "panel": ["HLA-A*02:01", "HLA-A*24:02", "HLA-A*03:01", "HLA-A*11:01", "HLA-A*01:01", "HLA-B*07:02", "HLA-B*44:02"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "split": "peptide_group_80_10_10_seed42"}`
- **Training**: `{"batch_size": 256, "encoder": "FixedBackbone(embed=128,layers=2,heads=4,ff=128)", "epochs": 10, "gpu": "H100!", "lr": "1e-4", "max_records": 0, "probes": ["SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI"], "schedule": "warmup_cosine", "seed": 42, "weight_decay": 0.01}`
- **Tested**: `[{"assay_mode": "additive", "cond_id": 1, "head_type": "mhcflurry", "label": "c01_mhcflurry_additive_max50k", "max_nM": 50000, "n_bins": 128, "sigma_mult": 0.75}, {"assay_mode": "additive", "cond_id": 2, "head_type": "mhcflurry", "label": "c02_mhcflurry_additive_max200k", "max_nM": 200000, "n_bins": 128, "sigma_mult": 0.75}, {"assay_mode": "additive", "cond_id": 3, "head_type": "log_mse", "label": "c03_log_mse_additive_max50k", "max_nM": 50000, "n_bins": 128, "sigma_mult": 0.75}, {"assay_mode": "additive", "cond_id": 4, "head_type": "log_mse", "label": "c04_log_mse_additive_max200k", "max_nM": 200000, "n_bins": 128, "sigma_mult": 0.75}, {"assay_mode": "d2_logit", "cond_id": 5, "head_type": "twohot", "label": "c05_twohot_d2_logit_max50k_K64", "max_nM": 50000, "n_bins": 64, "sigma_mult": 0.75}, {"assay_mode": "d2_logit", "cond_id": 6, "head_type": "twohot", "label": "c06_twohot_d2_logit_max50k_K128", "max_nM": 50000, "n_bins": 128, "sigma_mult": 0.75}, {"assay_mode": "d2_logit", "cond_id": 7, "head_type": "twohot", "label": "c07_twohot_d2_logit_max200k_K64", "max_nM": 200000, "n_bins": 64, "sigma_mult": 0.75}, {"assay_mode": "d2_logit", "cond_id": 8, "head_type": "twohot", "label": "c08_twohot_d2_logit_max200k_K128", "max_nM": 200000, "n_bins": 128, "sigma_mult": 0.75}, {"assay_mode": "d2_logit", "cond_id": 9, "head_type": "hlgauss", "label": "c09_hlgauss_d2_logit_max50k_K64_s0.75", "max_nM": 50000, "n_bins": 64, "sigma_mult": 0.75}, {"assay_mode": "d2_logit", "cond_id": 10, "head_type": "hlgauss", "label": "c10_hlgauss_d2_logit_max50k_K128_s0.75", "max_nM": 50000, "n_bins": 128, "sigma_mult": 0.75}, {"assay_mode": "d2_logit", "cond_id": 11, "head_type": "hlgauss", "label": "c11_hlgauss_d2_logit_max200k_K64_s0.75", "max_nM": 200000, "n_bins": 64, "sigma_mult": 0.75}, {"assay_mode": "d2_logit", "cond_id": 12, "head_type": "hlgauss", "label": "c12_hlgauss_d2_logit_max200k_K128_s0.75", "max_nM": 200000, "n_bins": 128, "sigma_mult": 0.75}]`

### 2026-03-14_1047_codex_mhcflurry-logmse-warmstart-20ep
- **Agent**: codex
- **Dir**: [2026-03-14_1047_codex_mhcflurry-logmse-warmstart-20ep](experiments/2026-03-14_1047_codex_mhcflurry-logmse-warmstart-20ep)
- **Source script**: `experiments/2026-03-14_1047_codex_mhcflurry-logmse-warmstart-20ep/code/launch.py`
- **Status**: launched
- **Dataset**: `{"assay_families": ["IC50", "KD", "KD (~IC50)", "KD (~EC50)", "EC50"], "measurement_profile": "numeric_no_qualitative", "panel": ["HLA-A*02:01", "HLA-A*24:02", "HLA-A*03:01", "HLA-A*11:01", "HLA-A*01:01", "HLA-B*07:02", "HLA-B*44:02"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "split": "peptide_group_80_10_10_seed42"}`
- **Training**: `{"batch_size": 256, "encoder": "FixedBackbone(embed=128,layers=2,heads=4,ff=128)", "epochs": 20, "gpu": "H100!", "lr": "1e-4", "max_records": 0, "probes": ["SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI"], "schedule": "warmup_cosine", "seed": 42, "warm_start_mode": "partial_encoder_only", "warm_start_source": "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt", "weight_decay": 0.01}`
- **Tested**: `[{"assay_mode": "additive", "cond_id": 1, "head_type": "mhcflurry", "label": "c01_mhcflurry_additive_max200k_cold", "max_nM": 200000, "n_bins": 128, "sigma_mult": 0.75, "warm_start": false}, {"assay_mode": "additive", "cond_id": 2, "head_type": "mhcflurry", "label": "c02_mhcflurry_additive_max200k_warm", "max_nM": 200000, "n_bins": 128, "sigma_mult": 0.75, "warm_start": true}, {"assay_mode": "additive", "cond_id": 3, "head_type": "log_mse", "label": "c03_log_mse_additive_max200k_cold", "max_nM": 200000, "n_bins": 128, "sigma_mult": 0.75, "warm_start": false}, {"assay_mode": "additive", "cond_id": 4, "head_type": "log_mse", "label": "c04_log_mse_additive_max200k_warm", "max_nM": 200000, "n_bins": 128, "sigma_mult": 0.75, "warm_start": true}]`

### 2026-03-16_1333_codex_exp21-winner-rerun
- **Agent**: codex
- **Dir**: [2026-03-16_1333_codex_exp21-winner-rerun](experiments/2026-03-16_1333_codex_exp21-winner-rerun)
- **Source script**: `experiments/2026-03-16_1333_codex_exp21-winner-rerun/code/launch.py`
- **Question**: Does the canonical experiment-local structure (`code/launch.py`, `manifest.json`, `launch_logs/`, `results/runs/`, `reproduce/`) reproduce the current EXP-21 winner exactly?
- **Dataset**: `data/merged_deduped.tsv`, `HLA-A*02:01` / `HLA-A*24:02`, `numeric_no_qualitative`, qualifier filter `all`, peptide-group `80/10/10` split with seed `42`
- **Training**: `groove`, `cond_id=2`, no content conditioning, `50` epochs, batch size `256`, `AdamW`, `lr=1e-3`, weight decay `0.01`, seed `43`, requested GPU `H100!`
- **Result**: Exact replay of the original best-seed EXP-21 run.
  - test Spearman `0.85413903`
  - test AUROC `0.94411862`
  - test AUPRC `0.91761374`
  - test RMSE log10 `0.81867343`
  - best val loss `0.02550036` at epoch `25`
  - best val Spearman `0.84519303` at epoch `39`
- **Comparison**: These held-out metrics match `dist-ba-v6-confirm-groove_c02-c02-cc0-e050-s43` exactly.
- **Takeaway**: No baseline change. The important result is workflow validation: the canonical experiment-local structure now reproduces the current best known 2-allele broad-numeric baseline exactly, so Claude and Codex should use the same `code/launch.py` / `results/runs/` / `reproduce/` layout going forward.

### 2026-03-16_1415_codex_presto-full-output-exp21-retry
- **Agent**: codex
- **Dir**: [2026-03-16_1415_codex_presto-full-output-exp21-retry](experiments/2026-03-16_1415_codex_presto-full-output-exp21-retry)
- **Source script**: `experiments/2026-03-16_1415_codex_presto-full-output-exp21-retry/code/launch.py`
- **Question**: Can current full-output Presto, using only sequence inputs and supervising the full affinity assay family, approach the EXP-21 groove winner on the exact same 2-allele contract?
- **Dataset**: `{"alleles": ["HLA-A*02:01", "HLA-A*24:02"], "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"], "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "measurement_profile": "numeric_no_qualitative", "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_seed": 43}`
- **Training**: `{"batch_size": 256, "binding_core_lengths": [8, 9, 10, 11], "binding_core_refinement": "shared", "binding_direct_segment_mode": "off", "binding_kinetic_input_mode": "affinity_vec", "d_model": 32, "epochs": 50, "groove_pos_mode": "concat_start_end_frac", "lr": 0.001, "n_heads": 4, "n_layers": 2, "optimizer": "AdamW", "peptide_pos_mode": "concat_start_end_frac", "ranking_losses": false, "requested_gpu": "H100!", "synthetic_negatives": false, "warm_start": "", "weight_decay": 0.01}`
- **Tested**: `[{"affinity_assay_residual_mode": "shared_base_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "log10", "condition_key": "PF03_log10_100k_full", "description": "Presto full loss, A03-style shared_base_segment_residual, split KD proxy, log10 100k target space", "kd_grouping_mode": "split_kd_proxy", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "PF07_mhcflurry_100k_full", "description": "Presto full loss, A07-style factorized_context_plus_segment residual, split KD proxy, mhcflurry 100k target space", "kd_grouping_mode": "split_kd_proxy", "max_affinity_nM": 100000}]`
- **Result**: Full-output Presto is now viable on the EXP-21 contract, but it does not beat the groove baseline.
  - `PF07_mhcflurry_100k_full`: test Spearman `0.84410185`, AUROC `0.93680900`, AUPRC `0.89547038`, RMSE log10 `0.86136419`
  - `PF03_log10_100k_full`: test Spearman `0.82043570`, AUROC `0.92695010`, AUPRC `0.88114345`, RMSE log10 `0.92678767`
- **Comparison**:
  - Best full-Presto condition is `PF07_mhcflurry_100k_full`
  - Relative to the EXP-21 winner `dist-ba-v6-confirm-groove_c02-c02-cc0-e050-s43`, `PF07` is lower by `0.0100` test Spearman, `0.0073` AUROC, and `0.0221` AUPRC, with RMSE log10 worse by `0.0427`
  - `PF07` is slightly better than the EXP-21 winner on thresholded `accuracy`, `balanced_accuracy`, and `F1`, but that does not outweigh the regression/ranking gap
- **Takeaway**: The earlier seq-only Presto collapse was a bad contract, not proof that full Presto could not learn this dataset. The careful retry works and `PF07` is the correct full-Presto branch to extend. Historical caveat added later: the EXP-21 groove comparator was an assay-conditioned legacy benchmark, and the honest no-assay-input comparison is superseded by [2026-03-16_2142_codex_exp21-honest-no-assay-repeat](experiments/2026-03-16_2142_codex_exp21-honest-no-assay-repeat).

### 2026-03-16_1441_codex_pf07-mainpath-optimization-extension
- **Agent**: codex
- **Dir**: [2026-03-16_1441_codex_pf07-mainpath-optimization-extension](experiments/2026-03-16_1441_codex_pf07-mainpath-optimization-extension)
- **Source script**: `experiments/2026-03-16_1441_codex_pf07-mainpath-optimization-extension/code/launch.py`
- **Status**: launched
- **Dataset**: `{"alleles": ["HLA-A*02:01", "HLA-A*24:02"], "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"], "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "measurement_profile": "numeric_no_qualitative", "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_seed": 43}`
- **Training**: `{"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "batch_size": 256, "binding_core_lengths": [8, 9, 10, 11], "binding_core_refinement": "shared", "binding_direct_segment_mode": "off", "binding_kinetic_input_mode": "affinity_vec", "d_model": 32, "epochs": 50, "groove_pos_mode": "concat_start_end_frac", "kd_grouping_mode": "split_kd_proxy", "max_affinity_nM": 100000, "n_heads": 4, "n_layers": 2, "optimizer": "AdamW", "peptide_pos_mode": "concat_start_end_frac", "ranking_losses": false, "requested_gpu": "H100!", "synthetic_negatives": false, "warm_start": "", "weight_decay": 0.01}`
- **Tested**: `[{"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "PF07_ctrl_lr1e3_constant", "description": "Current PF07 positive control: lr=1e-3, constant schedule", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "PF07_lr2p8e4_warmup_cosine", "description": "Historical A07 validation winner: lr=2.8e-4, warmup_cosine", "kd_grouping_mode": "split_kd_proxy", "lr": 0.00028, "lr_schedule": "warmup_cosine", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "PF07_lr2p8e4_onecycle", "description": "Historical A07 near-tie: lr=2.8e-4, onecycle", "kd_grouping_mode": "split_kd_proxy", "lr": 0.00028, "lr_schedule": "onecycle", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "PF07_lr1e4_warmup_cosine", "description": "Historical A07 lower-LR warmup comparator: lr=1e-4, warmup_cosine", "kd_grouping_mode": "split_kd_proxy", "lr": 0.0001, "lr_schedule": "warmup_cosine", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "PF07_lr1e4_constant", "description": "Historical A07 lower-LR constant comparator: lr=1e-4, constant schedule", "kd_grouping_mode": "split_kd_proxy", "lr": 0.0001, "lr_schedule": "constant", "max_affinity_nM": 100000}]`

### 2026-03-16_1454_claude_factorized-ablation-7allele
- **Agent**: claude
- **Dir**: [2026-03-16_1454_claude_factorized-ablation-7allele](experiments/2026-03-16_1454_claude_factorized-ablation-7allele)
- **Source script**: `scripts/benchmark_factorized_ablation.py`
- **Question**: Does factorized assay embedding (A07) help on the 7-allele / ~44K-row contract? What is the relative importance of pretraining, capacity, loss mode, target encoding?
- **Dataset**: 7 alleles (A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02), `numeric_no_qualitative`, qualifier_filter=all, train=32,805, val=4,184, test=4,060
- **Training**: `{"batch_size": 256, "d_model": "varies (128 or 32)", "epochs": 50, "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000, "n_heads": 4, "n_layers": 2, "synthetic_negatives": false, "warm_start": "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt (d=128 only)", "weight_decay": 0.01}`
- **GPU**: H100!
- **Tested**: 8 conditions — C1: d128/A07/full/pretrain/mhcflurry, C2: d128/A03/full/pretrain/mhcflurry, C3: d128/A07/full/cold-start/mhcflurry, C4: d128/A07/assay_heads_only/pretrain/mhcflurry, C5: d128/A07/full/pretrain/log10, C6: d32/A07/full/cold-start/mhcflurry, C7: d32/A03/full/cold-start/mhcflurry, C8: d128/pooled_single/full/pretrain/mhcflurry [NEG]
- **Result**: 5 of 8 conditions diverged (non-finite gradients). Stability is the dominant issue at lr=1e-3.
  - Stable conditions: C4 (d128, assay_heads_only, val_loss=0.0312, discrim=168x), C6 (d32 A07, val_loss=0.0313, discrim=113x), C7 (d32 A03, val_loss=0.0299, discrim=215x)
  - Best d=128: C4 (assay_heads_only + pretrain), best epoch 33, SLLQHLIGL@A02:01 Kd=11.4 nM, bind_prob=0.9254
  - Best d=32: C7 (A03, no factorized), best epoch 36, SLLQHLIGL@A02:01 Kd=44.6 nM, bind_prob=0.8954
  - C8 negative control: confirmed collapse, zero allele discrimination (1.0x ratio)
  - C5 log10 encoding: catastrophic, diverged epoch 3
- **Comparison**:
  - Factorized does not help at d=32: C7 (A03) > C6 (A07) on both val_loss and discrimination
  - Pretraining extends stability window (C1 epoch 49 vs C3 epoch 6) but does not prevent divergence
  - `assay_heads_only` is dramatically more stable than `full` loss at d=128 (C4 stable vs C1 diverged)
  - d=32 is more stable than d=128 at lr=1e-3 with `full` loss
- **Takeaway**: The `full` loss mode at d=128 with lr=1e-3 is too aggressive — trunk gradients cause divergence. Two viable paths forward: (1) d=128 with `assay_heads_only` (C4), or (2) d=32 with `full` loss (C7). To unlock d=128 + `full` loss, need lower lr or warmup schedule. Factorized assay embeddings do not help at d=32 and their effect at d=128 is confounded by instability.

### 2026-03-16_1549_codex_pf07-sequence-only-all-head-probe-rerun
- **Agent**: codex
- **Dir**: [2026-03-16_1549_codex_pf07-sequence-only-all-head-probe-rerun](experiments/2026-03-16_1549_codex_pf07-sequence-only-all-head-probe-rerun)
- **Source script**: `experiments/2026-03-16_1549_codex_pf07-sequence-only-all-head-probe-rerun/code/launch.py`
- **Status**: launched
- **Dataset**: `{"alleles": ["HLA-A*02:01", "HLA-A*24:02"], "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"], "assay_selector_inputs_forbidden": true, "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "measurement_profile": "numeric_no_qualitative", "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI", "IMLEGETKL"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_seed": 43}`
- **Training**: `{"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "batch_size": 256, "binding_core_lengths": [8, 9, 10, 11], "binding_core_refinement": "shared", "binding_direct_segment_mode": "off", "binding_kinetic_input_mode": "affinity_vec", "d_model": 32, "epochs": 50, "groove_pos_mode": "concat_start_end_frac", "kd_grouping_mode": "split_kd_proxy", "max_affinity_nM": 100000, "n_heads": 4, "n_layers": 2, "optimizer": "AdamW", "peptide_pos_mode": "concat_start_end_frac", "probe_artifact_schema": ["KD_nM", "IC50_nM", "EC50_nM", "KD_proxy_ic50_nM", "KD_proxy_ec50_nM", "binding_affinity_probe_kd"], "ranking_losses": false, "requested_gpu": "H100!", "synthetic_negatives": false, "warm_start": "", "weight_decay": 0.01}`
- **Tested**: `[{"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "PF07_ctrl_lr1e3_constant", "description": "Current PF07 positive control: lr=1e-3, constant schedule", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "PF07_lr2p8e4_warmup_cosine", "description": "Historical A07 validation winner: lr=2.8e-4, warmup_cosine", "kd_grouping_mode": "split_kd_proxy", "lr": 0.00028, "lr_schedule": "warmup_cosine", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "PF07_lr2p8e4_onecycle", "description": "Historical A07 near-tie: lr=2.8e-4, onecycle", "kd_grouping_mode": "split_kd_proxy", "lr": 0.00028, "lr_schedule": "onecycle", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "PF07_lr1e4_warmup_cosine", "description": "Historical A07 lower-LR warmup comparator: lr=1e-4, warmup_cosine", "kd_grouping_mode": "split_kd_proxy", "lr": 0.0001, "lr_schedule": "warmup_cosine", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "PF07_lr1e4_constant", "description": "Historical A07 lower-LR constant comparator: lr=1e-4, constant schedule", "kd_grouping_mode": "split_kd_proxy", "lr": 0.0001, "lr_schedule": "constant", "max_affinity_nM": 100000}]`

### 2026-03-16_1621_codex_pf07-output-tying-weight-sweep
- **Agent**: codex
- **Dir**: [2026-03-16_1621_codex_pf07-output-tying-weight-sweep](experiments/2026-03-16_1621_codex_pf07-output-tying-weight-sweep)
- **Source script**: `experiments/2026-03-16_1621_codex_pf07-output-tying-weight-sweep/code/launch.py`
- **Question**: Does weak output-side consistency regularization help the corrected sequence-only PF07 main-Presto affinity contract?
- **Dataset**: `data/merged_deduped.tsv`, `HLA-A*02:01` / `HLA-A*24:02`, `numeric_no_qualitative`, qualifier filter `all`, peptide-group `80/10/10` split with seed `42`
- **Training**: `PF07`, `mhcflurry`, `shared_base_factorized_context_plus_segment_residual`, `split_kd_proxy`, `50` epochs, batch size `256`, `AdamW`, `lr=1e-3`, weight decay `0.01`, seed `43`, requested GPU `H100!`, observed peak reserved GPU memory about `23.5 GiB`
- **Tested**: `12` conditions over:
  - KD-family tie weights `0.0`, `0.0025`, `0.01`, `0.04`
  - proxy-cross tie weights `0.0`, `0.001`, `0.004`
  - fixed `binding_output_consistency_beta = 0.25`
- **Result**: No regularized condition beat the untied control.
  - Winner: `kd=0.0`, `cross=0.0`
  - test Spearman `0.8196399`
  - test AUROC `0.9288369`
  - test AUPRC `0.8851608`
  - test RMSE log10 `0.9208454`
  - best regularized condition: `kd=0.01`, `cross=0.001`
  - best regularized test Spearman `0.8192348`, AUROC `0.9287891`, AUPRC `0.8849943`, RMSE log10 `0.9318614`
- **Comparison**:
  - very small tying came close on Spearman but still lost and had worse RMSE
  - stronger tying hurt consistently, especially any `cross=0.004` setting and the `kd=0.04` family
  - worst condition: `kd=0.0025`, `cross=0.004`, test Spearman `0.8004243`
- **Operational note**: The first detached-launch pass hit a Modal image-build race for `11 / 12` runs. Those launches failed before training started and were relaunched without any contract change; all `12 / 12` runs then completed and were collected locally.
- **Takeaway**: Weak output-tying did not improve this corrected sequence-only PF07 contract. The practical choice remains the untied PF07 control; no model-to-beat update is warranted from this sweep.

### 2026-03-16_2142_codex_exp21-honest-no-assay-repeat
- **Agent**: codex
- **Dir**: [2026-03-16_2142_codex_exp21-honest-no-assay-repeat](experiments/2026-03-16_2142_codex_exp21-honest-no-assay-repeat)
- **Source script**: `experiments/2026-03-16_2142_codex_exp21-honest-no-assay-repeat/code/launch.py`
- **Question**: Does the old EXP-21 groove/historical benchmark family still win once assay-selector inputs are actually disabled?
- **Dataset**: `data/merged_deduped.tsv`, `HLA-A*02:01` / `HLA-A*24:02`, `numeric_no_qualitative`, qualifier filter `all`, seed `43`
- **Training**: legacy `distributional_ba` v6 benchmark, `assay_input_mode=none`, no content conditioning, `50` epochs, batch size `256`, `AdamW`, `lr=1e-3`, weight decay `0.01`, requested GPU `H100!`
- **Tested**:
  - `groove c02`
  - `groove c01`
  - `historical c02`
  - all at seed `43`, `50` epochs
- **Result**: The old groove winner does not survive under the honest input contract.
  - `groove c02`: test Spearman `0.7995139`, AUROC `0.9206325`, AUPRC `0.8640233`, RMSE log10 `0.9383441`
  - `groove c01`: test Spearman `0.7936514`
  - `historical c02`: test Spearman `0.7932025`
- **Comparison**:
  - honest `groove c02` vs old cheating `groove c02`: `-0.0546252` Spearman, `-0.0234861` AUROC, `-0.0535905` AUPRC, `+0.1196707` RMSE log10
  - honest `groove c02` vs honest PF07 untied control: `-0.0201260` Spearman, `-0.0082044` AUROC, `-0.0211375` AUPRC, `+0.0174987` RMSE log10
- **Takeaway**: The old EXP-21 benchmark was materially benefiting from assay embeddings. Once those forbidden inputs are removed, the best legacy benchmark run is weaker than the current honest PF07 full-output Presto control. The active no-assay-input baseline should therefore move away from EXP-21 `groove c02`.

### 2026-03-16_2227_claude_lr-stability-7allele
- **Agent**: claude
- **Dir**: [2026-03-16_2227_claude_lr-stability-7allele](experiments/2026-03-16_2227_claude_lr-stability-7allele)
- **Source script**: `scripts/benchmark_lr_stability_7allele.py`
- **Question**: Can we stabilize d=128 + `full` loss (which diverged at lr=1e-3) by lowering lr or adding warmup? Does `full` loss beat `assay_heads_only` once stable?
- **Prior**: `2026-03-16_1454_claude_factorized-ablation-7allele`
- **Dataset**: 7 alleles (A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02), `numeric_no_qualitative`, qualifier_filter=all, train=32,805, val=4,184, test=4,060
- **Training**: `{"batch_size": 256, "d_model": 128, "epochs": 50, "kd_grouping_mode": "split_kd_proxy", "max_affinity_nM": 100000, "n_heads": 4, "n_layers": 2, "synthetic_negatives": false, "warm_start": "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt", "weight_decay": 0.01}`
- **GPU**: H100!
- **Tested**: 6 conditions — S1: A07/full/lr=3e-4/warmup_cosine, S2: A07/full/lr=3e-4/onecycle, S3: A07/full/lr=1e-4/warmup_cosine, S4: A07/full/lr=1e-3/warmup_cosine, S5: A07/heads_only/lr=1e-3/constant [C4 ctrl], S6: A03/full/lr=3e-4/warmup_cosine
- **Result**: lr=3e-4 + warmup_cosine stabilizes d=128 + full loss. lr=1e-3 still diverges even with warmup (S4 diverged ep14).
  - S1 (lr=3e-4 warmup_cosine): stable 50ep, val_loss=0.0306, probe_rank_corr=0.8839, discrim=115.5x
  - S2 (lr=3e-4 onecycle): stable 50ep, val_loss=0.0320, probe_rank_corr=0.8393, discrim=26.7x
  - S3 (lr=1e-4 warmup_cosine): stable 50ep, val_loss=0.0330, probe_rank_corr=0.8929, discrim=151.1x
  - S4 (lr=1e-3 warmup_cosine): diverged ep14, val_loss=0.0376
  - S5 (heads_only lr=1e-3 constant): stable 50ep, val_loss=0.0294, probe_rank_corr=0.5536, discrim=358.7x
  - S6 (A03 lr=3e-4 warmup_cosine): stable 50ep, val_loss=0.0310, probe_rank_corr=0.8393, discrim=22.2x
- **Comparison**:
  - S5 (heads_only) has lowest val_loss (0.0294) and highest discrimination (358.7x) but poor probe rank corr (0.5536)
  - S1 (full loss, lr=3e-4 warmup_cosine) is best balanced: good val_loss (0.0306), excellent rank corr (0.8839), SLLQHLIGL@A02:01 IC50=32.4 nM / Kd=26.9 nM
  - warmup_cosine > onecycle at lr=3e-4 (S1 vs S2)
  - A07 slightly > A03 once stable (S1 vs S6): val_loss 0.0306 vs 0.0310, discrimination 115.5x vs 22.2x
- **Takeaway**: lr=3e-4 + warmup_cosine is the recommended recipe for d=128 + full loss on the 7-allele contract. S1 is the new 7-allele d=128 full-loss baseline. S5 (heads_only) remains a valid alternative if pure discrimination is the goal, but its low rank correlation suggests the assay heads are less coherent.

### 2026-03-16_2355_codex_pf07-assay-structured-dag-sweep
- **Agent**: codex
- **Dir**: [2026-03-16_2355_codex_pf07-assay-structured-dag-sweep](experiments/2026-03-16_2355_codex_pf07-assay-structured-dag-sweep)
- **Source script**: `experiments/2026-03-16_2355_codex_pf07-assay-structured-dag-sweep/code/launch.py`
- **Dataset**: `{"alleles": ["HLA-A*02:01", "HLA-A*24:02"], "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"], "assay_selector_inputs_forbidden": true, "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "measurement_profile": "numeric_no_qualitative", "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI", "IMLEGETKL"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_seed": 43}`
- **Training**: `{"affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "batch_size": 256, "binding_core_lengths": [8, 9, 10, 11], "binding_core_refinement": "shared", "binding_direct_segment_mode": "off", "binding_kinetic_input_mode": "affinity_vec", "d_model": 32, "epochs": 50, "groove_pos_mode": "concat_start_end_frac", "kd_grouping_mode": "split_kd_proxy", "max_affinity_nM": 100000, "n_heads": 4, "n_layers": 2, "optimizer": "AdamW", "peptide_pos_mode": "concat_start_end_frac", "probe_artifact_schema": ["KD_nM", "IC50_nM", "EC50_nM", "KD_proxy_ic50_nM", "KD_proxy_ec50_nM", "binding_affinity_probe_kd"], "ranking_losses": false, "requested_gpu": "H100!", "synthetic_negatives": false, "warm_start": "", "weight_decay": 0.01}`
- **Tested**: `[{"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_control", "description": "Honest PF07 control with flat assay residual heads", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_family", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_family", "description": "Output-side family-anchor DAG: KD -> {IC50 family, EC50 family} -> leaf heads", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf", "description": "Family-anchor DAG with output-side assay-method leaves for IC50/EC50", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf", "description": "Family-anchor DAG with factorized output-side prep/readout leaves for IC50/EC50", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}]`
- **Result**:
  - `pf07_control`: test Spearman `0.8130993`, AUROC `0.9242548`, AUPRC `0.8781925`, RMSE log10 `0.9376745`
  - `pf07_dag_family`: test Spearman `0.8207888`, AUROC `0.9263466`, AUPRC `0.8738952`, RMSE log10 `0.9088196`
  - `pf07_dag_method_leaf`: test Spearman `0.8359941`, AUROC `0.9357420`, AUPRC `0.8852642`, RMSE log10 `0.8722534`
  - `pf07_dag_prep_readout_leaf`: test Spearman `0.8381589`, AUROC `0.9358775`, AUPRC `0.8780973`, RMSE log10 `0.8703900`
- **Winner**: `pf07_dag_prep_readout_leaf`
- **Comparison**:
  - Relative to the flat honest PF07 control in the same sweep, `dag_prep_readout_leaf` gains `+0.02506` test Spearman and `+0.01162` AUROC with `-0.06728` RMSE log10; AUPRC is effectively flat (`-0.00010`)
  - `dag_method_leaf` is the strongest secondary comparator and the AUPRC winner, but it trails `dag_prep_readout_leaf` slightly on test Spearman and RMSE
  - Relative to the earlier honest PF07 untied baseline from [2026-03-16_1621_codex_pf07-output-tying-weight-sweep](experiments/2026-03-16_1621_codex_pf07-output-tying-weight-sweep), the winner improves by `+0.01852` Spearman, `+0.00704` AUROC, and `-0.05046` RMSE log10, with AUPRC lower by `0.00706`
- **Takeaway**: Output-side assay-structured DAGs help under the strict no-assay-input contract. The gain does not come from a coarse family anchor alone; it comes from adding leaf structure. `dag_prep_readout_leaf` is the new honest baseline to beat for the 2-allele broad-numeric affinity contract.

### 2026-03-17_1042_claude_7allele-bakeoff
- **Agent**: claude
- **Dir**: [2026-03-17_1042_claude_7allele-bakeoff](experiments/2026-03-17_1042_claude_7allele-bakeoff)
- **Source script**: `experiments/2026-03-17_1042_claude_7allele-bakeoff/launch.py`
- **Status**: complete
- **Question**: Which combination of residual architecture, loss mode, LR recipe, capacity, and pretraining produces the best 7-allele binding model on held-out test metrics + probe discrimination?
- **Dataset**: 7 alleles (A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02), ~44K rows, `numeric_no_qualitative`, qualifier `all`, split_seed=42, mhcflurry encoding
- **Training**: 50 epochs, batch_size=256, n_layers=2, n_heads=4, weight_decay=0.01, split_kd_proxy, max_affinity_nM=100000, binding_core_lengths=8,9,10,11, shared refinement, no synthetics, no contrastive. 3 seeds (42,43,44) per condition. H100!
- **Conditions**: 13 conditions varying: residual mode (A07, A03, DAG), loss mode (full, assay_heads_only), LR (3e-4 warmup_cosine, 1e-3 constant), d_model (128, 32), pretrain (yes, no)
- **Result** (top 5 by test Spearman):
  - `D3` (d32 DAG full no-pretrain lr=1e-3): **Spearman 0.8101±0.0032**, AUROC 0.9353, F1 0.8169, probe ratio 1279x
  - `L2` (d128 DAG heads_only pretrain lr=3e-4 warmup): Spearman 0.8062±0.0060, AUROC 0.9336, **F1 0.8202**, **probe ratio 2502x**
  - `D2` (d128 DAG heads_only pretrain lr=1e-3): Spearman 0.8030 (1 seed survived)
  - `P2` (d128 DAG full no-pretrain lr=3e-4 warmup): Spearman 0.8020±0.0034
  - `D1` (d128 DAG full pretrain lr=3e-4 warmup): Spearman 0.7914±0.0088
- **Stability**: D2 (2/3 NaN), A2 (2/3 NaN) — both lr=1e-3 constant at d128. All warmup_cosine conditions 3/3 stable.
- **Winner**: `L2` — best combined regression + discrimination + stability. Promoted to 7-allele baseline in `model_to_beat.md`.
- **Key findings**:
  1. DAG dominates: all top-5 are `dag_prep_readout_leaf`, +0.02-0.04 Spearman over A07/A03
  2. A07 ≈ A03: the factorized context vector in A07 provides no benefit
  3. lr=3e-4 warmup_cosine is critical for stability and probe calibration; lr=1e-3 constant causes NaN at d128 for DAG/A03
  4. heads_only preserves allele discrimination; full reshapes trunk toward regression at cost of probe quality
  5. Pretraining has small, inconsistent effect at 7-allele scale
  6. d32 wins regression but collapses all non-binders to identical predictions
- **Takeaway**: The recommended recipe for full class I training is L2: `dag_prep_readout_leaf + assay_heads_only + lr=3e-4 warmup_cosine + d128 + pretrained`

### 2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun
- **Agent**: codex
- **Dir**: [2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun](experiments/2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun)
- **Source script**: `experiments/2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun/code/launch.py`
- **Status**: completed
- **Dataset**: `{"alleles": ["HLA-A*02:01", "HLA-A*24:02"], "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"], "assay_selector_inputs_forbidden": true, "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "measurement_profile": "numeric_no_qualitative", "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI", "IMLEGETKL"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_seed": 43}`
- **Training**: `{"affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "batch_size": 256, "binding_core_lengths": [8, 9, 10, 11], "binding_core_refinement": "shared", "binding_direct_segment_mode": "off", "binding_kinetic_input_mode": "affinity_vec", "d_model": 32, "epoch_val_metrics_frequency": 1, "epochs": 50, "groove_pos_mode": "concat_start_end_frac", "kd_grouping_mode": "split_kd_proxy", "max_affinity_nM": 100000, "n_heads": 4, "n_layers": 2, "optimizer": "AdamW", "peptide_pos_mode": "concat_start_end_frac", "probe_artifact_schema": ["KD_nM", "IC50_nM", "EC50_nM", "KD_proxy_ic50_nM", "KD_proxy_ec50_nM", "binding_affinity_probe_kd"], "probe_plot_frequency": "final", "requested_gpu": "H100!", "weight_decay": 0.01}`
- **Tested**: `[{"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_control_constant", "description": "Flat honest PF07 control rerun: 1e-3 constant", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_family", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_family_constant", "description": "Family-anchor DAG rerun: 1e-3 constant", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf_constant", "description": "Method-leaf DAG rerun: 1e-3 constant", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf_constant", "description": "Prep/readout-leaf DAG rerun: 1e-3 constant", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf_warmup_cosine", "description": "Method-leaf DAG schedule variant: 3e-4 warmup_cosine", "kd_grouping_mode": "split_kd_proxy", "lr": 0.0003, "lr_schedule": "warmup_cosine", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf_warmup_cosine", "description": "Prep/readout-leaf DAG schedule variant: 3e-4 warmup_cosine", "kd_grouping_mode": "split_kd_proxy", "lr": 0.0003, "lr_schedule": "warmup_cosine", "max_affinity_nM": 100000}]`
- **Observed runtime / artifacts**:
  - All `6 / 6` Modal runs were fetched locally into `results/runs/`
  - Requested GPU: `H100!`
  - Peak observed allocated GPU memory across runs stayed around `8.86 GiB`
  - New local reproducibility artifacts landed in `results/epoch_metrics_by_condition.csv`, `results/probe_affinity_by_condition_long.csv`, and the `val_*_over_epochs.png` plot set
- **Held-out validation curve summary**:
  - Final-epoch validation winner: `pf07_dag_prep_readout_leaf_constant` with val Spearman `0.8277`, val AUROC `0.9339`, val AUPRC `0.8923`, val RMSE log10 `0.8912`
  - Best observed validation point: the same condition at epoch `18`, with val Spearman `0.8338`, val AUROC `0.9341`, val AUPRC `0.8931`, val RMSE log10 `0.8573`
- **Held-out test metrics**:
  - `pf07_dag_prep_readout_leaf_constant`: Spearman `0.8379`, AUROC `0.9367`, AUPRC `0.8907`, RMSE log10 `0.8778`
  - `pf07_dag_method_leaf_constant`: Spearman `0.8339`, AUROC `0.9336`, AUPRC `0.8854`, RMSE log10 `0.8819`
  - `pf07_dag_method_leaf_warmup_cosine`: Spearman `0.8313`, AUROC `0.9307`, AUPRC `0.8812`, RMSE log10 `0.8649`
  - `pf07_dag_prep_readout_leaf_warmup_cosine`: Spearman `0.8312`, AUROC `0.9325`, AUPRC `0.8822`, RMSE log10 `0.8740`
  - `pf07_dag_family_constant`: Spearman `0.8206`, AUROC `0.9270`, AUPRC `0.8765`, RMSE log10 `0.9203`
  - `pf07_control_constant`: Spearman `0.8160`, AUROC `0.9281`, AUPRC `0.8826`, RMSE log10 `0.9225`
- **Winner**: `pf07_dag_prep_readout_leaf_constant`
- **Comparison**:
  - The rerun reproduced the earlier DAG sweep winner with effectively unchanged test Spearman (`-0.00028` vs the prior `dag_prep_readout_leaf` winner), slightly better AUROC (`+0.00082`) and much better AUPRC (`+0.01263`), with modestly worse RMSE (`+0.00741`)
  - Constant `1e-3` remained better than the warmup-cosine variants on the primary test metric, even though the warmup-cosine leaf models were competitive on AUROC/AUPRC curves
  - The coarse `dag_family` model again improved on the flat control, but most of the gain still came from the leaf-structured DAG variants
- **Artifacts**:
  - Per-condition test table: [condition_summary.csv](experiments/2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun/results/condition_summary.csv)
  - Combined epoch curves: [epoch_metrics_by_condition.csv](experiments/2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun/results/epoch_metrics_by_condition.csv)
  - Combined probe trajectories: [probe_affinity_by_condition_long.csv](experiments/2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun/results/probe_affinity_by_condition_long.csv)
  - Final probe panel: [final_probe_predictions.csv](experiments/2026-03-17_1205_codex_pf07-dag-epoch-curve-rerun/results/final_probe_predictions.csv)
- **Takeaway**: The stronger epoch-curve artifact contract does not change the scientific answer. `PF07` with the `dag_prep_readout_leaf` output DAG, `mhcflurry` targets, `split_kd_proxy`, and strict sequence-only inputs remains the best honest 2-allele affinity model in the repo.

### 2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep
- **Agent**: codex
- **Dir**: [2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep](experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep)
- **Source script**: `experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/code/launch.py`
- **Status**: completed
- **Dataset**: `{"alleles": ["HLA-A*02:01", "HLA-A*24:02"], "assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"], "assay_selector_inputs_forbidden": true, "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "measurement_profile": "numeric_no_qualitative", "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI", "IMLEGETKL"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_seed": 43}`
- **Training**: `{"downstream": {"affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "batch_size": 256, "binding_core_lengths": [8, 9, 10, 11], "binding_core_refinement": "shared", "binding_direct_segment_mode": "off", "binding_kinetic_input_mode": "affinity_vec", "d_model": 32, "epochs": 50, "groove_pos_mode": "concat_start_end_frac", "kd_grouping_mode": "split_kd_proxy", "max_affinity_nM": 100000, "n_heads": 4, "n_layers": 2, "optimizer": "AdamW", "peptide_pos_mode": "concat_start_end_frac", "requested_gpu": "H100!", "synthetic_negatives": false, "weight_decay": 0.01}, "pretraining": {"batch_size": 192, "checkpoint_name": "mhc_pretrain.pt", "d_model": 32, "mode": "mhc_pretrain", "n_heads": 4, "n_layers": 2, "pretrain_epochs": [1, 2], "seed": 42, "targets": ["chain_type", "species", "class"]}}`
- **Tested**: `[{"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_control_constant", "description": "Flat honest PF07 control with constant LR", "init_checkpoint": "", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000, "pretrain_epochs": 0, "pretrain_key": "pretrain_0ep"}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_control_constant", "description": "Flat honest PF07 control with constant LR", "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000, "pretrain_epochs": 1, "pretrain_key": "pretrain_1ep"}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_control_constant", "description": "Flat honest PF07 control with constant LR", "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e02/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000, "pretrain_epochs": 2, "pretrain_key": "pretrain_2ep"}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf_constant", "description": "Method-leaf DAG with constant LR", "init_checkpoint": "", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000, "pretrain_epochs": 0, "pretrain_key": "pretrain_0ep"}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf_constant", "description": "Method-leaf DAG with constant LR", "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000, "pretrain_epochs": 1, "pretrain_key": "pretrain_1ep"}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf_constant", "description": "Method-leaf DAG with constant LR", "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e02/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000, "pretrain_epochs": 2, "pretrain_key": "pretrain_2ep"}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf_constant", "description": "Prep/readout-leaf DAG with constant LR", "init_checkpoint": "", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000, "pretrain_epochs": 0, "pretrain_key": "pretrain_0ep"}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf_constant", "description": "Prep/readout-leaf DAG with constant LR", "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000, "pretrain_epochs": 1, "pretrain_key": "pretrain_1ep"}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf_constant", "description": "Prep/readout-leaf DAG with constant LR", "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e02/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000, "pretrain_epochs": 2, "pretrain_key": "pretrain_2ep"}]`
- **Observed runtime / artifacts**:
  - All `2 / 2` fresh pretrains and `9 / 9` downstream finetune runs were fetched locally
  - Requested GPU: `H100!`
  - Peak observed reserved GPU memory for downstream runs stayed around `23.52 GiB`
  - Pretrain checkpoints were preserved under `results/pretrains/`
- **Pretrain-only metrics**:
  - `1` epoch: val loss `0.1732`, val species acc `0.9105`, val class acc `1.0000`
  - `2` epochs: val loss `0.0943`, val species acc `0.9615`, val class acc `1.0000`
- **Held-out validation / test metrics**:
  - `pf07_control_constant`, `0` ep: val Spearman `0.7978`, test Spearman `0.8155`, test AUROC `0.9263`, test AUPRC `0.8778`, test RMSE log10 `0.9215`
  - `pf07_control_constant`, `1` ep: val Spearman `0.7818`, test Spearman `0.8002`, test AUROC `0.9197`, test AUPRC `0.8556`, test RMSE log10 `0.9290`
  - `pf07_control_constant`, `2` ep: val Spearman `0.7846`, test Spearman `0.8082`, test AUROC `0.9255`, test AUPRC `0.8852`, test RMSE log10 `0.9161`
  - `pf07_dag_method_leaf_constant`, `0` ep: val Spearman `0.8219`, test Spearman `0.8376`, test AUROC `0.9365`, test AUPRC `0.8981`, test RMSE log10 `0.8850`
  - `pf07_dag_method_leaf_constant`, `1` ep: val Spearman `0.8174`, test Spearman `0.8227`, test AUROC `0.9284`, test AUPRC `0.8781`, test RMSE log10 `0.8962`
  - `pf07_dag_method_leaf_constant`, `2` ep: val Spearman `0.8199`, test Spearman `0.8295`, test AUROC `0.9326`, test AUPRC `0.8880`, test RMSE log10 `0.8794`
  - `pf07_dag_prep_readout_leaf_constant`, `0` ep: val Spearman `0.8203`, test Spearman `0.8356`, test AUROC `0.9333`, test AUPRC `0.8877`, test RMSE log10 `0.8636`
  - `pf07_dag_prep_readout_leaf_constant`, `1` ep: val Spearman `0.8194`, test Spearman `0.8377`, test AUROC `0.9352`, test AUPRC `0.8881`, test RMSE log10 `0.8769`
  - `pf07_dag_prep_readout_leaf_constant`, `2` ep: val Spearman `0.8151`, test Spearman `0.8347`, test AUROC `0.9330`, test AUPRC `0.8815`, test RMSE log10 `0.8866`
- **Winner**: `pf07_dag_prep_readout_leaf_constant` with `1` epoch of MHC pretraining
- **Comparison**:
  - Best within-sweep result (`0.8376657`) was essentially tied with the current honest baseline but did not beat it (`-0.0004932` Spearman vs the prior `0.8381589`)
  - Mean test Spearman by pretrain duration was best at `0` epochs (`0.82958`), then `2` epochs (`0.82412`), then `1` epoch (`0.82020`)
  - Pretraining clearly hurt the flat PF07 control and the `dag_method_leaf` model
  - Only `dag_prep_readout_leaf` showed a slight positive Spearman bump at `1` epoch, but that came with worse RMSE and no practical baseline change
- **Artifacts**:
  - Per-condition table: [condition_summary.csv](experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/condition_summary.csv)
  - Summary bundle: [summary_bundle.json](experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/summary_bundle.json)
  - Raw pretrains: [results/pretrains/](experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/)
  - Raw finetune runs: [results/runs/](experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/runs/)
- **Takeaway**: Short MHC class/species pretraining does not produce a robust improvement on the honest seq-only PF07 affinity contract. The pretrain objective itself works, but it does not transfer cleanly enough to justify changing the current baseline.

### 2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep
- **Agent**: codex
- **Dir**: [2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep](experiments/2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep)
- **Source script**: `experiments/2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep/code/launch.py`
- **Status**: launched
- **Dataset**: `{"assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"], "assay_selector_inputs_forbidden": true, "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "measurement_profile": "numeric_no_qualitative", "probe_alleles": ["HLA-A*02:01", "HLA-A*24:02"], "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI", "IMLEGETKL"], "qualifier_filter": "all", "sequence_resolution": "mhcseqs_first_with_index_fallback", "source": "data/merged_deduped.tsv", "source_filter": "iedb", "source_refresh": "canonical rebuild 2026-03-17", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_all_alleles": true, "train_mhc_class_filter": "I", "train_seed": 43}`
- **Training**: `{"downstream": {"affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "batch_size": 256, "binding_core_lengths": [8, 9, 10, 11], "binding_core_refinement": "shared", "binding_direct_segment_mode": "off", "binding_kinetic_input_mode": "affinity_vec", "d_model": 32, "epoch_grid": [10, 25, 50], "groove_pos_mode": "concat_start_end_frac", "kd_grouping_mode": "split_kd_proxy", "max_affinity_nM": 100000, "n_heads": 4, "n_layers": 2, "optimizer": "AdamW", "peptide_pos_mode": "concat_start_end_frac", "requested_gpu": "H100!", "synthetic_negatives": false, "weight_decay": 0.01}, "pretraining": {"d_model": 32, "mode": "mhc_pretrain", "n_heads": 4, "n_layers": 2, "warm_start_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "warm_start_epochs": 1}}`
- **Tested**: `[{"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_control_constant", "description": "Flat honest PF07 control on rebuilt all-class-I numeric data", "epoch_budget": 10, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_control_constant", "description": "Flat honest PF07 control on rebuilt all-class-I numeric data", "epoch_budget": 25, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_control_constant", "description": "Flat honest PF07 control on rebuilt all-class-I numeric data", "epoch_budget": 50, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf_constant", "description": "Method-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 10, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf_constant", "description": "Method-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 25, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf_constant", "description": "Method-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 50, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf_constant", "description": "Prep/readout-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 10, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf_constant", "description": "Prep/readout-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 25, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf_constant", "description": "Prep/readout-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 50, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}]`

### 2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun
- **Agent**: codex
- **Dir**: [2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun](experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun)
- **Source script**: `experiments/2026-03-18_1828_codex_pf07-all-classI-cleanup-validation-rerun/code/launch.py`
- **Status**: launched
- **Dataset**: `{"assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"], "assay_selector_inputs_forbidden": true, "comparison_target": "2026-03-17_1404_codex_pf07-all-classI-1ep-pretrain-epoch-sweep", "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "measurement_profile": "numeric_no_qualitative", "probe_alleles": ["HLA-A*02:01", "HLA-A*24:02"], "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI", "IMLEGETKL"], "qualifier_filter": "all", "sequence_resolution": "mhcseqs_first_with_index_fallback", "source": "data/merged_deduped.tsv", "source_filter": "iedb", "source_refresh": "canonical rebuild 2026-03-17", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_all_alleles": true, "train_mhc_class_filter": "I", "train_seed": 43, "validation_purpose": "post_mhcseqs_cleanup_rerun"}`
- **Training**: `{"downstream": {"affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "batch_size": 256, "binding_core_lengths": [8, 9, 10, 11], "binding_core_refinement": "shared", "binding_direct_segment_mode": "off", "binding_kinetic_input_mode": "affinity_vec", "d_model": 32, "epoch_grid": [10, 25, 50], "groove_pos_mode": "concat_start_end_frac", "kd_grouping_mode": "split_kd_proxy", "max_affinity_nM": 100000, "n_heads": 4, "n_layers": 2, "optimizer": "AdamW", "peptide_pos_mode": "concat_start_end_frac", "requested_gpu": "H100!", "synthetic_negatives": false, "weight_decay": 0.01}, "pretraining": {"d_model": 32, "mode": "mhc_pretrain", "n_heads": 4, "n_layers": 2, "warm_start_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "warm_start_epochs": 1}}`
- **Tested**: `[{"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_control_constant", "description": "Flat honest PF07 control on rebuilt all-class-I numeric data", "epoch_budget": 10, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_control_constant", "description": "Flat honest PF07 control on rebuilt all-class-I numeric data", "epoch_budget": 25, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_control_constant", "description": "Flat honest PF07 control on rebuilt all-class-I numeric data", "epoch_budget": 50, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf_constant", "description": "Method-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 10, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf_constant", "description": "Method-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 25, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_method_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_method_leaf_constant", "description": "Method-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 50, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf_constant", "description": "Prep/readout-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 10, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf_constant", "description": "Prep/readout-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 25, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_loss_mode": "full", "affinity_target_encoding": "mhcflurry", "condition_key": "pf07_dag_prep_readout_leaf_constant", "description": "Prep/readout-leaf DAG on rebuilt all-class-I numeric data", "epoch_budget": 50, "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "kd_grouping_mode": "split_kd_proxy", "lr": 0.001, "lr_schedule": "constant", "max_affinity_nM": 100000}]`

### 2026-03-19_1007_codex_cpu-vs-mps-focused-pf07-smoke
- **Agent**: codex
- **Dir**: [2026-03-19_1007_codex_cpu-vs-mps-focused-pf07-smoke](experiments/2026-03-19_1007_codex_cpu-vs-mps-focused-pf07-smoke)
- **Source script**: `experiments/2026-03-19_1007_codex_cpu-vs-mps-focused-pf07-smoke/code/launch.py`
- **Question**: Is the new local Apple Silicon `mps` path stable enough to reproduce a tiny focused PF07 run, or should local continuation stay on `cpu`?
- **Dataset**: `{"assay_selector_inputs_forbidden": true, "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "max_records": 200, "measurement_profile": "numeric_no_qualitative", "probe_alleles": ["HLA-A*02:01", "HLA-A*24:02"], "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "source_filter": "iedb", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_mhc_class_filter": "I", "train_seed": 43}`
- **Training**: `{"downstream": {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_target_encoding": "mhcflurry", "batch_size": 8, "d_model": 32, "devices": ["cpu", "mps"], "epochs": 1, "kd_grouping_mode": "split_kd_proxy", "matmul_precision": "default", "n_heads": 4, "n_layers": 2}, "pretraining": {"warm_start_checkpoint_local": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt"}}`
- **Tested**: `[{"batch_size": 8, "condition_key": "cpu", "description": "Matched tiny focused PF07 smoke run on CPU", "device": "cpu", "epochs": 1, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 200, "seed": 43, "split_seed": 42}, {"batch_size": 8, "condition_key": "mps", "description": "Matched tiny focused PF07 smoke run on Apple Silicon MPS", "device": "mps", "epochs": 1, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 200, "seed": 43, "split_seed": 42}]`
- **Result**:
  - `cpu` completed
    - val Spearman `-0.0591`, AUROC `0.5469`, AUPRC `0.3866`, RMSE log10 `1.8723`
    - test Spearman `-0.1048`, AUROC `0.5833`, AUPRC `0.3142`, RMSE log10 `1.4804`
  - `mps` diverged
    - divergence epoch `1`
    - divergence reason `non_finite_train_loss`
    - no held-out metrics were produced
- **Operational note**: The `mps` launch log also reported a fallback warning for `nonzero`, so the backend was not operating as a clean fully-native path even before divergence.
- **Takeaway**: Apple Silicon `mps` is not yet a safe replacement for `cpu` on this focused PF07 training path. The local continuation workflow should keep defaulting to `cpu` until the MPS numeric instability is fixed.

### 2026-03-19_1201_codex_cpu-vs-mps-focused-pf07-smoke-mpssafe
- **Agent**: codex
- **Dir**: [2026-03-19_1201_codex_cpu-vs-mps-focused-pf07-smoke-mpssafe](experiments/2026-03-19_1201_codex_cpu-vs-mps-focused-pf07-smoke-mpssafe)
- **Source script**: `experiments/2026-03-19_1201_codex_cpu-vs-mps-focused-pf07-smoke-mpssafe/code/launch.py`
- **Question**: Does the explicit MPS-safe runtime fix make Apple Silicon stable on the matched tiny focused PF07 smoke contract, and are the resulting metrics still close to CPU?
- **Dataset**: `{"assay_selector_inputs_forbidden": true, "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "max_records": 200, "measurement_profile": "numeric_no_qualitative", "probe_alleles": ["HLA-A*02:01", "HLA-A*24:02"], "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "source_filter": "iedb", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_mhc_class_filter": "I", "train_seed": 43}`
- **Training**: `{"downstream": {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_target_encoding": "mhcflurry", "batch_size": 8, "d_model": 32, "devices": ["cpu", "mps"], "epochs": 1, "kd_grouping_mode": "split_kd_proxy", "matmul_precision": "default", "n_heads": 4, "n_layers": 2}, "pretraining": {"warm_start_checkpoint_local": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt"}}`
- **Tested**: `[{"batch_size": 8, "condition_key": "cpu", "description": "Matched tiny focused PF07 smoke run on CPU", "device": "cpu", "epochs": 1, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 200, "seed": 43, "split_seed": 42}, {"batch_size": 8, "condition_key": "mps", "description": "Matched tiny focused PF07 smoke run on Apple Silicon MPS", "device": "mps", "epochs": 1, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 200, "seed": 43, "split_seed": 42}]`
- **Results**:
  - `cpu`: test Spearman `-0.10479`, AUROC `0.58333`, AUPRC `0.31422`, RMSE log10 `1.48043`
  - `mps`: test Spearman `-0.10811`, AUROC `0.57540`, AUPRC `0.30583`, RMSE log10 `1.47338`
- **Runtime note**:
  - `mps_safe_mode_applied = zero_dropout`
  - zeroed `20` dropout modules and `17` multihead-attention dropout sites on `mps`
  - no backend fallback warnings were observed in the launch logs
- **Takeaway**: The explicit MPS-safe runtime fix removed the earlier Apple Silicon divergence on the matched 1-epoch smoke contract, and the resulting metrics stayed very close to CPU.

### 2026-03-19_1236_codex_cpu-vs-mps-focused-pf07-minitrain-mpssafe
- **Agent**: codex
- **Dir**: [2026-03-19_1236_codex_cpu-vs-mps-focused-pf07-minitrain-mpssafe](experiments/2026-03-19_1236_codex_cpu-vs-mps-focused-pf07-minitrain-mpssafe)
- **Source script**: `experiments/2026-03-19_1236_codex_cpu-vs-mps-focused-pf07-minitrain-mpssafe/code/launch.py`
- **Question**: Does the same MPS-safe path stay stable beyond startup on a short multi-epoch matched CPU-vs-MPS training run?
- **Dataset**: `{"assay_selector_inputs_forbidden": true, "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "max_records": 200, "measurement_profile": "numeric_no_qualitative", "probe_alleles": ["HLA-A*02:01", "HLA-A*24:02"], "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "source_filter": "iedb", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_mhc_class_filter": "I", "train_seed": 43}`
- **Training**: `{"downstream": {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_target_encoding": "mhcflurry", "batch_size": 8, "d_model": 32, "devices": ["cpu", "mps"], "epochs": 3, "kd_grouping_mode": "split_kd_proxy", "matmul_precision": "default", "n_heads": 4, "n_layers": 2}, "pretraining": {"warm_start_checkpoint_local": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt"}}`
- **Tested**: `[{"batch_size": 8, "condition_key": "cpu", "description": "Matched tiny focused PF07 smoke run on CPU", "device": "cpu", "epochs": 3, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 200, "seed": 43, "split_seed": 42}, {"batch_size": 8, "condition_key": "mps", "description": "Matched tiny focused PF07 smoke run on Apple Silicon MPS", "device": "mps", "epochs": 3, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 200, "seed": 43, "split_seed": 42}]`
- **Results**:
  - `cpu`: test Spearman `-0.29232`, AUROC `0.40873`, AUPRC `0.22934`, RMSE log10 `1.41349`
  - `mps`: test Spearman `-0.27928`, AUROC `0.40476`, AUPRC `0.22817`, RMSE log10 `1.40135`
- **Runtime note**:
  - `mps_safe_mode_applied = zero_dropout`
  - the same Apple Silicon safeguard remained stable through `3` epochs with no `non_finite_train_loss`
  - no backend fallback warnings were observed in the launch logs
- **Takeaway**: The MPS-safe path remains stable through a short multi-epoch focused PF07 run and stays closely aligned with CPU on held-out metrics, making Apple Silicon usable for local focused training with the explicit caveat that dropout is disabled on `mps`.

### 2026-03-19_1338_codex_cpu-vs-mps-manual-dropout-parity
- **Agent**: codex
- **Dir**: [2026-03-19_1338_codex_cpu-vs-mps-manual-dropout-parity](experiments/2026-03-19_1338_codex_cpu-vs-mps-manual-dropout-parity)
- **Source script**: `experiments/2026-03-19_1338_codex_cpu-vs-mps-manual-dropout-parity/code/launch.py`
- **Question**: Can MPS keep a real nonzero dropout rate if `nn.Dropout` is replaced with explicit manual dropout, and does that improve the CPU/MPS contract mismatch?
- **Dataset**: `{"assay_selector_inputs_forbidden": true, "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "max_records": 200, "measurement_profile": "numeric_no_qualitative", "probe_alleles": ["HLA-A*02:01", "HLA-A*24:02"], "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "source_filter": "iedb", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_mhc_class_filter": "I", "train_seed": 43}`
- **Training**: `{"downstream": {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_target_encoding": "mhcflurry", "batch_size": 8, "d_model": 32, "devices": ["cpu", "mps"], "epochs": 3, "kd_grouping_mode": "split_kd_proxy", "matmul_precision": "default", "n_heads": 4, "n_layers": 2}, "pretraining": {"warm_start_checkpoint_local": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt"}}`
- **Tested**: `[{"batch_size": 8, "condition_key": "cpu", "description": "Matched tiny focused PF07 smoke run on CPU", "device": "cpu", "epochs": 3, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 200, "seed": 43, "split_seed": 42}, {"batch_size": 8, "condition_key": "mps", "description": "Matched tiny focused PF07 smoke run on Apple Silicon MPS", "device": "mps", "epochs": 3, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 200, "seed": 43, "split_seed": 42}]`
- **Results**:
  - `cpu`: test Spearman `-0.22546`, AUROC `0.42460`, AUPRC `0.24257`, RMSE log10 `1.41110`
  - `mps`: test Spearman `-0.27383`, AUROC `0.39286`, AUPRC `0.22799`, RMSE log10 `1.41937`
- **Runtime note**:
  - `mps_safe_mode_applied = manual_dropout`
  - `mps_safe_dropout_modules_replaced = 20`
  - this intermediate run still used device-local RNG for the dropout masks
- **Takeaway**: Manual dropout preserves a real nonzero dropout rate and keeps MPS stable, but by itself it did not remove the CPU/MPS metric drift. This motivated the seeded-manual follow-up.

### 2026-03-19_1412_codex_cpu-vs-mps-seeded-manual-dropout-parity
- **Agent**: codex
- **Dir**: [2026-03-19_1412_codex_cpu-vs-mps-seeded-manual-dropout-parity](experiments/2026-03-19_1412_codex_cpu-vs-mps-seeded-manual-dropout-parity)
- **Source script**: `experiments/2026-03-19_1412_codex_cpu-vs-mps-seeded-manual-dropout-parity/code/launch.py`
- **Question**: If CPU and MPS use the same seeded CPU-generated manual dropout masks, does the remaining CPU/MPS gap disappear?
- **Dataset**: `{"assay_selector_inputs_forbidden": true, "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "max_records": 200, "measurement_profile": "numeric_no_qualitative", "probe_alleles": ["HLA-A*02:01", "HLA-A*24:02"], "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI"], "qualifier_filter": "all", "source": "data/merged_deduped.tsv", "source_filter": "iedb", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_mhc_class_filter": "I", "train_seed": 43}`
- **Training**: `{"downstream": {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_target_encoding": "mhcflurry", "batch_size": 8, "d_model": 32, "devices": ["cpu", "mps"], "epochs": 3, "kd_grouping_mode": "split_kd_proxy", "matmul_precision": "default", "n_heads": 4, "n_layers": 2}, "pretraining": {"warm_start_checkpoint_local": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt"}}`
- **Tested**: `[{"batch_size": 8, "condition_key": "cpu", "description": "Matched tiny focused PF07 smoke run on CPU", "device": "cpu", "epochs": 3, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 200, "seed": 43, "split_seed": 42}, {"batch_size": 8, "condition_key": "mps", "description": "Matched tiny focused PF07 smoke run on Apple Silicon MPS", "device": "mps", "epochs": 3, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 200, "seed": 43, "split_seed": 42}]`
- **Results**:
  - `cpu`: test Spearman `-0.22238`, AUROC `0.39286`, AUPRC `0.22614`, RMSE log10 `1.37729`
  - `mps`: test Spearman `-0.28853`, AUROC `0.38889`, AUPRC `0.22469`, RMSE log10 `1.40880`
- **Runtime note**:
  - `mps_safe_mode_applied = manual_dropout`
  - `mps_safe_dropout_modules_replaced = 20`
  - dropout masks were generated from seeded CPU RNG on both backends
  - no fallback warnings or `non_finite_train_loss` events occurred
- **Takeaway**: The dropout implementation can now be made genuinely hardware-independent, but exact CPU/MPS metric parity still does not follow. The remaining drift is backend math / optimization behavior elsewhere in the stack, not a dropout-contract mismatch.

### 2026-03-19_1505_codex_cpu-vs-mps-allclass1-manualdropout-validation
- **Agent/model**: `codex`
- **Dir**: [2026-03-19_1505_codex_cpu-vs-mps-allclass1-manualdropout-validation](experiments/2026-03-19_1505_codex_cpu-vs-mps-allclass1-manualdropout-validation)
- **Source script**: `experiments/2026-03-19_1505_codex_cpu-vs-mps-allclass1-manualdropout-validation/code/launch.py`
- **Dataset / curation contract**:
  - canonical rebuilt `data/merged_deduped.tsv`
  - `source=iedb`
  - `train_mhc_class_filter=I`
  - `train_all_alleles=True`
  - `measurement_profile=numeric_no_qualitative`
  - `qualifier_filter=all`
  - reduced validation cap: `max_records=5000`
  - honest inputs only: `nflank`, `peptide`, `cflank`, `mhc_a`, `mhc_b`
  - supervised outputs: `IC50`, `KD`, `KD(~IC50)`, `KD(~EC50)`, `EC50`
- **Training / pretraining contract**:
  - downstream design: `presto_pf07_dag_prep_readout_leaf_constant`
  - `d_model=32`, `n_layers=2`, `n_heads=4`
  - `epochs=3`, `batch_size=256`, `lr=1e-3`, `weight_decay=0.01`
  - warm start: `experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt`
  - hardware-parity dropout contract: force `--mps-safe-mode manual_dropout` on both CPU and MPS
- **Tested conditions**:
  - `cpu`: test Spearman `0.20249`, AUROC `0.59125`, AUPRC `0.43531`, RMSE log10 `1.38687`, loss `0.06661`
  - `mps`: test Spearman `0.30026`, AUROC `0.65745`, AUPRC `0.48325`, RMSE log10 `1.36865`, loss `0.06579`
- **Runtime note**:
  - `mps_safe_mode_applied = manual_dropout` on both backends
  - `mps_safe_dropout_modules_replaced = 20`
  - `mps_safe_dropout_modules_zeroed = 0`
  - no fallback warnings, NaNs, or `non_finite_train_loss` events appeared in the launch logs
- **Takeaway**:
  - Once dropout implementation is held constant across hardware, Apple Silicon `mps` completes a materially larger honest all-class-I PF07 slice cleanly.
  - MPS did not underperform CPU on this run; it was slightly better on all headline metrics.
  - This is not proof of exact backend equivalence, but it is strong enough to treat `mps` as usable for local focused training under the manual-dropout contract.

### 2026-03-20_0920_codex_cpu-vs-mps-allclass1-auto-default-confirmation
- **Agent/model**: `codex`
- **Dir**: [2026-03-20_0920_codex_cpu-vs-mps-allclass1-auto-default-confirmation](experiments/2026-03-20_0920_codex_cpu-vs-mps-allclass1-auto-default-confirmation)
- **Source script**: `experiments/2026-03-20_0920_codex_cpu-vs-mps-allclass1-auto-default-confirmation/code/launch.py`
- **Dataset / curation contract**:
  - canonical rebuilt `data/merged_deduped.tsv`
  - `source=iedb`
  - `train_mhc_class_filter=I`
  - `train_all_alleles=True`
  - `measurement_profile=numeric_no_qualitative`
  - `qualifier_filter=all`
  - reduced confirmation cap: `max_records=5000`
  - honest inputs only: `nflank`, `peptide`, `cflank`, `mhc_a`, `mhc_b`
  - supervised outputs: `IC50`, `KD`, `KD(~IC50)`, `KD(~EC50)`, `EC50`
- **Training / pretraining contract**:
  - downstream design: `presto_pf07_dag_prep_readout_leaf_constant`
  - `d_model=32`, `n_layers=2`, `n_heads=4`
  - `epochs=3`, `batch_size=256`, `lr=1e-3`, `weight_decay=0.01`
  - warm start: `experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt`
  - launched with default `--mps-safe-mode auto`
- **Tested conditions**:
  - `cpu`: test Spearman `0.20254`, AUROC `0.59127`, AUPRC `0.43533`, RMSE log10 `1.38687`, loss `0.06661`
  - `mps`: test Spearman `0.30026`, AUROC `0.65745`, AUPRC `0.48325`, RMSE log10 `1.36865`, loss `0.06579`
- **Runtime note**:
  - although the launch used `--mps-safe-mode auto`, the runtime summaries show:
    - `mps_safe_mode_requested = auto`
    - `mps_safe_mode_applied = manual_dropout`
  - `mps_safe_dropout_modules_replaced = 20` on both backends
  - no fallback warnings, NaNs, or `non_finite_train_loss` events appeared in the logs
- **Takeaway**:
  - The default path now truly uses manual dropout on CPU and MPS alike.
  - This confirmation reproduced the same result pattern as the explicit-manual run from `2026-03-19_1505...`, so manual dropout is now the actual shared default rather than a special override.
- **Status**: launched
- **Dataset**: `{"assay_families_supervised": ["IC50", "KD", "KD(~IC50)", "KD(~EC50)", "EC50"], "assay_selector_inputs_forbidden": true, "input_fields": ["nflank", "peptide", "cflank", "mhc_a", "mhc_b"], "max_records": 5000, "measurement_profile": "numeric_no_qualitative", "probe_alleles": ["HLA-A*02:01", "HLA-A*24:02"], "probe_peptides": ["SLLQHLIGL", "FLRYLLFGI", "NFLIKFLLI", "IMLEGETKL"], "qualifier_filter": "all", "sequence_resolution": "mhcseqs_first_with_index_fallback", "source": "data/merged_deduped.tsv", "source_filter": "iedb", "source_refresh": "canonical rebuild 2026-03-17", "split_policy": "peptide_group_80_10_10", "split_seed": 42, "train_all_alleles": true, "train_mhc_class_filter": "I", "train_seed": 43, "validation_purpose": "cpu_vs_mps_all_class_i_auto_default_confirmation"}`
- **Training**: `{"downstream": {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "affinity_target_encoding": "mhcflurry", "batch_size": 256, "d_model": 32, "design_id": "presto_pf07_dag_prep_readout_leaf_constant", "devices": ["cpu", "mps"], "epochs": 3, "kd_grouping_mode": "split_kd_proxy", "mps_safe_mode": "auto", "n_heads": 4, "n_layers": 2}, "pretraining": {"mode": "mhc_pretrain", "warm_start_checkpoint_local": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt"}}`
- **Tested**: `[{"affinity_assay_residual_mode": "dag_prep_readout_leaf", "batch_size": 256, "condition_key": "cpu", "description": "Matched all-class-I reduced PF07 validation on CPU under default auto mode", "device": "cpu", "epochs": 3, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 5000, "mps_safe_mode": "auto", "seed": 43, "split_seed": 42}, {"affinity_assay_residual_mode": "dag_prep_readout_leaf", "batch_size": 256, "condition_key": "mps", "description": "Matched all-class-I reduced PF07 validation on Apple Silicon MPS under default auto mode", "device": "mps", "epochs": 3, "init_checkpoint": "experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/results/pretrains/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt", "max_records": 5000, "mps_safe_mode": "auto", "seed": 43, "split_seed": 42}]`

### 2026-08-26_1200_claude_excision-detectability-e2e
- **Agent**: claude
- **Dir**: [2026-08-26_1200_claude_excision-detectability-e2e](experiments/2026-08-26_1200_claude_excision-detectability-e2e)
- **Source script**: `experiments/2026-08-26_1200_claude_excision-detectability-e2e/reproduce/launch.sh`
- **Kind**: validation run, not a result. It answers "does every branch train together in the canonical path", not "is the model better".
- **Question**: Do the hitlist data source, expanded latent DAG, machinery-conditioned excision head, and non-MHC shotgun corpus train together in one `presto train unified` invocation, with every loss receiving support?
- **Dataset**: hitlist curated indexes (`--data-source hitlist`), class I, `HLA-A*02:01`, caps binding 1200 / elution 1200 / tcell 300 / vdjdb 200 / processing 200 / stability 200 / kinetics 50, `cap_sampling=head`; plus hitlist `bulk_proteomics` HeLa, 1500 observed peptides across four protease arms; T-cell / TCR / processing still from `data/merged_deduped.tsv` because hitlist does not carry them.
- **Curation contract**: `response_measured` routed 1:1 onto Presto assay families; rows with no numeric value dropped (parity with the merged-TSV contract); per-response unit guard; exploded protein mappings collapsed to one row per evidence row preferring the canonical transcript.
- **Training**: `latent_topology=expanded`, `d_model=32`, 2 layers, 4 heads, 510,251 params, 1 epoch, batch 32, CPU. No pretraining, no synthetic negatives beyond defaults.
- **Assay-label -> output mapping**: see [docs/assay_learning_scheme.md](docs/assay_learning_scheme.md).
- **Loss terms added**: `ms_detectability` (`bce`, soft depth-graded targets, weight 0.5) and `excision` (`bce`, weight 1.0), both in the canonical `LOSS_TASK_SPECS` registry.
- **GPU**: none (CPU validation run).
- **Result**: trains end to end. Best val loss `1.6547`; total loss `4.98 -> 2.18` over 226 batches. Resolved MHC sequences `215/219` alleles via mhcseqs. Flank coverage binding `63.5%`, elution `88.5%` — the merged TSV supplies `0%`, so this is the first Presto run with real junction context. Total samples `9,012` (MHC branch `6,012` + shotgun branch `3,000`). **25 tasks received support.**
  - `binding` train `2.7577` / val `1.8248`
  - `presentation` train `0.6573` / val `0.4409`
  - `ms_detectability` train `0.6620` / val `0.6344`
  - `elution` train `0.6365` / val `0.3406`
  - `tcell` train `0.5222` / val `0.2432`
  - `excision` train `0.2694` / val `0.2584`
  - mean rows per batch: `excision` 17.0, `ms_detectability` 9.0
- **Held-out metrics**: the wiring landed during this session, so the headline run above predates it; a smaller companion run with the same configuration scored **13 tasks** — excision AUPRC `0.9889` (n=132), immunogenicity `0.9821` (n=34), presentation `0.9740` (n=110), elution `0.9512` (n=110), tcell `0.8866` (n=34), ms_detectability `0.8080` (n=65), binding_ic50 Spearman `0.2554` (n=25), binding Spearman `-0.0606` (n=136). Plumbing evidence only. Two honest caveats: excision's AUPRC largely measures its own pinned constraint, since negatives are cleavage-rule violations (`pin_profiles=False` is the informative setting); and binding Spearman is negative at one epoch / `d_model=32`, which is exactly why the Stage 4 factorial is the gate. A target-space mismatch in the prediction dump was found and fixed during this session (raw nM against transformed predictions, which invalidated RMSE/Pearson); it did not explain the negative Spearman, which is monotone-invariant.
- **Comparison**: not comparable to any baseline in `model_to_beat.md`. Those were all trained through the merged-TSV path with empty flank segments; see the caveat at the top of that file.
- **Takeaway**: the plumbing is real — both corpora co-train, the expanded DAG trains, and the two new losses get substantive support. Nothing is claimed about quality: one epoch, `d_model=32`, one allele, CPU. The decision gate is the Stage 4 factorial (arms A-E), which has not been run.
- **Defects fixed en route**: (1) `train unified` silently dropped every MHC row because sequence resolution was gated on `if args.index_csv:` while the resolver is mhcseqs-first with the CSV as fallback; (2) `float("nan")` passing a null check in the hitlist adapter, which would have admitted NaN affinity targets.

---

## 2026-09-02 · Panel-scale fix validation — the first run that learns

**Agent:** claude (Opus 5) · **Dir:** [`2026-09-02_0130_claude_panel-scale-fix-validation`](2026-09-02_0130_claude_panel-scale-fix-validation/) · **Commit:** `2688399` (clean) · **Hardware:** local CPU, no GPU

### Why

`AffinityPredictor.predict_assay_panel` returns a KD offset in normalized log10
space; the panel loss regressed it against `bind_target` **raw** (nanomolar, to
50,000). `loss_binding_assay_panel` sat near 142,900 — an order of magnitude
above the total loss — and could not fall, because no head output reaches
50,000. It dominated every gradient, and the model learned nothing:

| | before | after |
|---|---|---|
| synthetic val loss | 9924.72 → 9923.51 / 10 epochs (**0.012%**) | 1.9511 → 0.9156 / 40 (**53%**) |
| `loss_binding_assay_panel` | 142,888 → 142,879 (frozen) | 2.82 → 1.56 |

Nothing about the bug was exceptional — finite loss, training ran to
completion, 1,417 tests passed. Only the magnitude gave it away.

### Contract

hitlist 1.55.2, MS + binding evidence, `map_source_proteins=True`, restricted to
`HLA-A*02:01`. Caps: binding 2500 / elution 2500 / t-cell 800 / VDJdb 800 /
stability 400 → 21,189 samples. Synthetic negatives pmhc 1.000, processing
0.500, derived 0.500. d_model 64, 2 layers, 4 heads, expanded topology, 8
epochs, batch 32. Peptide-grouped split.

### Held-out results

Best val loss **0.6708** (epoch 5; 6–8 overfit, expected at this size).

- **Binding, genuinely learned:** Spearman **0.823** (n=1137), affinity probe
  **0.828**, KD **0.878** (n=70), IC50 **0.774** (n=236), t_half 0.736 (n=73).
- **Elution / presentation AUROC 0.996 / 0.981 is not presentation skill.**
  `elution_real_only_n_negatives = 0`: the corpus supplies no real negatives, so
  every negative is a synthetic decoy and the number measures peptide realism.
  The decoy-stratified metrics report this rather than the headline figure.
- **T-cell / immunogenicity at chance on real negatives:** AUROC 0.512 and
  0.519 (n=112), against 0.74–0.82 versus decoys. Under-powered at 800 capped
  T-cell records rather than proven broken.

### Second arm: pan-allele at the same budget

Same caps and hyperparameters with `--hitlist-mhc-class I`: 567 alleles, 479
resolved, 12,502 samples. Run to check the first arm was not memorising one
allele's motif.

| metric | HLA-A*02:01 | pan-allele (479) |
|---|---:|---:|
| binding Spearman (n≈1130) | 0.823 | **0.781** |
| binding affinity probe | 0.828 | **0.766** |
| binding KD | 0.878 | **0.850** |
| binding IC50 | 0.774 | 0.222 |
| immunogenicity real-only AUROC | 0.512 | 0.570 |
| tcell real-only AUROC | 0.519 | 0.597 |

Binding generalises — 0.78 across 479 alleles against 0.82 on one. IC50 falling
to 0.222 is the one number worth chasing. Elution/presentation read AUROC 1.000
with `elution_real_only_n_negatives = 0`, so a perfect score is a reminder that
it is decoy detection, not a result.

### Takeaway

The pipeline trains end to end on real data and binding is genuinely learned,
pan-allele. Presentation and T-cell claims are not supported by either arm —
the first for lack of real negatives in the corpus, the second for lack of
scale. Both need a GPU run.

## 2026-09-02--03 · Source-junction ambiguity masking

> **Validity erratum (2026-09-04):** This family is preserved as historical
> output but is not decision-grade. In the legacy arm, nullable unmapped flanks
> became the amino-acid string `NAN`; masking cleared them. Thus 6,212 unmapped
> binding rows changed for an accidental normalization reason alongside the
> intended ambiguous junctions. Model seeds also changed capped membership,
> sparse targets lacked held-out support, and elution was all-positive. The
> stated mapping-policy winner is withdrawn pending the gated 2026-09-04 rerun.

**Agent/model:** Codex / GPT-5 · **Dir:**
[`2026-09-02_1720_codex_source-junction-masking`](2026-09-02_1720_codex_source-junction-masking/)
· **Commit:** `a1cdcc0d27d63a7440188f98f0dc503c596c8dfd` (dirty)

### Question

Does replacing arbitrary source-protein junction choices with explicit unknown
flanks improve held-out prediction enough to justify candidate-junction
marginalization?

### Contract

- **Dataset/curation:** hitlist 1.55.8, `HLA-A*02:01`, 7--30 aa; all 16,721
  qualifying numeric binding rows plus reservoir-capped elution 20,000,
  stability 1,000, and kinetics 2, for 37,723 rows per seed. Peptide-disjoint
  80/10/10 split; seeds 42/43/44 and modality sampling seeds 59/60/61. Binding
  inequalities retained for censor-aware loss; exact metrics use
  `qualifier == 0`; threshold metrics use a qualifier-aware `<=500 nM` label.
  No merged TSV, bulk proteomics, processing, T-cell, TCR, or RNA expression.
- **Frozen inputs:** rebuilt 5,880,924-row mapping contract v2 with
  `gene_biotype`; artifact hashes and exact curation details are in the
  [experiment README](2026-09-02_1720_codex_source-junction-masking/README.md).
- **Training/pretraining:** expanded Presto d128/l2/h4, 10 epochs, batch 256,
  AdamW `2.8e-4`, weight decay 0.01, AMP; no pretraining. Task-mean supervised
  aggregation with uncertainty weighting.
- **Synthetic-data contract:** pMHC, class-I-no-B2M, processing-negative, and
  MHC-augmentation counts zero. The configured UniProt ratio was the default
  0.1, but no UniProt file was mounted, so zero such rows entered the audited
  37,723-row dataset.
- **Loss and assay mapping:** affinity -> censor-aware canonical KD and
  KD/IC50/EC50 assay outputs; half-life/Tm -> censor-aware stability outputs;
  kon/koff -> MSE kinetics outputs; positive MS -> BCE elution and presentation
  outputs. Ordinary task base weight 1.0; MHC auxiliary type losses 0.1.
  Consistency weights: cascade 0.2, assay-affinity 0.1,
  assay-presentation 0.1, no-B2M 0.5, T-cell-context 0.05, T-cell-upstream 0.2.
- **Conditions:** `legacy_global_canonical` versus `mask_unresolved`, paired at
  seeds 42/43/44. The masked policy preserves single/agreed mappings and a
  unique canonical source only within one gene; cross-gene and residual
  within-gene disagreements become `?/?`. Targets and target weights are
  identical.
- **Hardware/runtime:** requested `H100!`; all six workers observed NVIDIA H100
  80GB HBM3 with 81,559 MiB. Peak allocated memory was 31.63--32.00 GiB and
  peak reserved memory 53.05--64.28 GiB. Mean runtime 4,870 s legacy /
  4,828 s masked; 29,095 GPU-seconds total.

### Held-out results

Mean +/- sample SD over three seeds; RMSE is `log10(nM)`.

| Split / scope | Policy | Exact n/seed | Spearman | Pearson | RMSE | <=500 nM bal. acc. | AUROC | AUPRC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Validation overall | legacy | 1320 +/- 27 | 0.7523 +/- .0145 | 0.7401 +/- .0138 | 0.9969 +/- .0157 | 0.8439 +/- .0164 | 0.9157 +/- .0083 | 0.8594 +/- .0128 |
| Validation overall | masked | 1320 +/- 27 | 0.7554 +/- .0163 | 0.7455 +/- .0160 | 0.9824 +/- .0168 | 0.8481 +/- .0086 | 0.9170 +/- .0110 | 0.8590 +/- .0171 |
| Test overall | legacy | 1384 +/- 50 | 0.7643 +/- .0091 | 0.7542 +/- .0107 | 0.9873 +/- .0293 | 0.8584 +/- .0052 | 0.9186 +/- .0050 | 0.8607 +/- .0117 |
| Test overall | masked | 1384 +/- 50 | 0.7673 +/- .0131 | 0.7571 +/- .0157 | 0.9813 +/- .0236 | 0.8576 +/- .0072 | 0.9186 +/- .0078 | 0.8631 +/- .0151 |
| Test unresolved | legacy | 29 +/- 11 | 0.7608 +/- .0634 | 0.7594 +/- .0799 | 0.9853 +/- .1764 | 0.9033 +/- .0372 | 0.9689 +/- .0273 | 0.9728 +/- .0099 |
| Test unresolved | masked | 29 +/- 11 | 0.7470 +/- .0347 | 0.7520 +/- .0758 | 0.9982 +/- .2337 | 0.9306 +/- .0078 | 0.9764 +/- .0146 | 0.9761 +/- .0176 |

Full validation/test metrics for `single`, `flanks_agree`,
`within_gene_canonical`, both unresolved categories, their union, and
`unmapped` are in
[`condition_metric_summary.csv`](2026-09-02_1720_codex_source-junction-masking/results/condition_metric_summary.csv).
All six per-example validation/test prediction dumps are preserved under
`results/runs/`.

### Winner and takeaway

**Preferred policy: `mask_unresolved`; do not build candidate marginalization
now.** The choice is based on semantic correctness plus overall parity, not a
headline accuracy win. On the predeclared primary unresolved exact test scope,
masked-minus-legacy Spearman deltas were -0.0532/+0.0156/-0.0037 (mean
**-0.0138**) and RMSE deltas were +0.0723/+0.0159/-0.0495 (mean **+0.0129**).
The direction was inconsistent and missed both marginalization thresholds.
Overall test deltas were +0.0030 Spearman and -0.0059 RMSE, with no material
regression. Keep valid peptide/MHC labels at full weight, encode unresolved
junctions as unknown, defer the GM12878-only expression join, and revisit a
costlier candidate model only with materially broader transcript-aware data or
a separately powered multi-allele experiment.

The original `20260902a` launch was stopped before results were read after an
X-filter parity confound was found. The corrected `20260902b` arms passed exact
row parity, frozen-hash, hitlist-version, MHC-index, requested-GPU, and observed
H100 checks. Reproduction and raw artifact paths are recorded in the
[experiment directory](2026-09-02_1720_codex_source-junction-masking/).
The comparator preserves the prior semantic behavior (inject one arbitrary
junction) but uses stable candidate ordering instead of reproducing the old
frame-order accident byte for byte.

## 2026-09-03 - Flank context wired end to end, on corrected mapping data

> **Validity erratum (2026-09-04):** This rerun fixed terminus and checkpoint
> handling but retained the same null-to-`NAN` policy confound on 6,212 unmapped
> binding rows. Its archived metrics remain reproducible; its mapping-policy
> conclusion is withdrawn. The 2026-09-04 remediation also fixes seed-dependent
> caps, sparse held-out targets, and all-positive binary supervision before any
> replacement GPU run can start.

**Agent/model:** Claude / Opus 5 - **Dir:**
[`2026-09-03_1746_claude_flank-context-fixes`](2026-09-03_1746_claude_flank-context-fixes/)
- **Commit:** `e58dd9ffbcd09163a7e08aabeed0d6be318184be` (branch `claude/flank-context-and-data-fixes`, dirty)

### Question

Does the `20260902b` legacy-vs-masked verdict survive on corrected code, and
what changes once a protein terminus actually reaches the model? It never had:
`flank_n_is_terminus` was declared on three record dataclasses, moved to the
device by `PrestoBatch.to` and branched on by `Presto._pad_for_side`, but no
`PrestoSample(...)` construction in `data/loaders.py` passed it. Every row
arrived `False`, so the `X` boundary pad was unreachable and every flank -
terminus or not - padded with `?`. On this corpus that is 3.2% of binding rows
with an N-terminus and 2.9% with a C-terminus, all previously told the context
was unknown when in fact it definitively did not exist.

### Contract

- **Dataset/curation:** identical to `20260902b` - hitlist 1.55.8 frozen
  artifacts (same five SHA256s, hash-verified in-container before the GPU is
  touched), `HLA-A*02:01`, 16,721 binding + 20,000 elution + 1,000 stability
  + 2 kinetics = 37,723 rows per seed, peptide-disjoint 80/10/10, seeds
  42/43/44, sampling seeds 59/60/61. Mapping categories are byte-identical to
  `20260902b` seed for seed, so the data-correctness fixes below are
  defensive on this corpus rather than corpus-changing.
- **Split seed changed.** The three-way splitter reused the two-way helper's
  `seed + 53` peptide order and filled *test* from the front where two-way
  fills *validation* from the front, so at a given seed every held-out test
  peptide had already been used for model selection. Now `seed + 101`. Test
  splits are therefore not comparable row-for-row with `20260902b`.
- **Training:** Presto expanded d128/l2/h4, AdamW lr 2.8e-4 wd 0.01, batch 256,
  10 epochs, bf16 autocast. Synthetic negatives, MHC augmentation and probe
  tracking disabled. Loss terms and assay-label -> output mapping unchanged
  from `20260902b`.
- **Requested GPU** `H100!`; **observed** on all six:
  `NVIDIA H100 80GB HBM3, 81559 MiB, 580.95.05`.
  4,104-5,556 s per run, 28,168 GPU-seconds total, ~1h21m wall in parallel.
- **Supervision parity** between the two policies verified for all four
  evidence families before launch (`results/data_audit.json`).

### Corrections carried in this family

Terminus wiring; `?` removed from flank sequence strings entirely (one
representation per layer: `""` in the frame, `None` on the sample, a pad
token in the tensor); a candidate collapse that could let a left-join miss
outrank the real mapping and still be stamped `single`/resolved;
`_canonical_mask` reading a float-encoded flag as False; `_flank_coverage`
gating on resolution so it printed the same number for both arms; three
drifted copies of "which categories are resolved"; and the split fix above.

### Evaluation checkpoint

These runs launched before `train_iedb` reloaded the best-validation
checkpoint for the held-out pass, so `results/` scores whatever epoch training
stopped on while carrying `best_val_loss` from a different one. Four of six
runs selected an epoch before the last (7, 4, 10, 7, 10, 9). `model.pt` is the
selected epoch, so every run was re-scored offline from it - splits re-derived
and verified identical to each original before any metric was written - into
`results/best_checkpoint/`. **Prefer that set.**

### Held-out test, overall (best checkpoint, mean +- sd, seeds 42-44)

| Metric | `legacy_global_canonical` | `mask_unresolved` |
|---|---|---|
| exact Spearman | 0.7536 +- 0.0062 | 0.7604 +- 0.0057 |
| exact RMSE log10(nM) | 1.0039 +- 0.0177 | 0.9948 +- 0.0135 |
| <=500 nM AUROC | 0.9164 +- 0.0034 | 0.9190 +- 0.0025 |
| <=500 nM AUPRC | 0.8677 +- 0.0048 | 0.8711 +- 0.0080 |
| <=500 nM F1 | 0.8274 +- 0.0067 | 0.8191 +- 0.0068 |
| <=500 nM balanced acc. | 0.8484 +- 0.0089 | 0.8419 +- 0.0062 |

### Decision gate

| Quantity | 20260902b | this, as-run | this, best ckpt |
|---|---|---|---|
| `overall_exact_spearman_mean_delta` | +0.00300 | +0.00216 | +0.00683 |
| `overall_exact_rmse_mean_delta` | -0.00594 | +0.00367 | -0.00909 |
| `unresolved_exact_spearman_mean_delta` | -0.01377 | -0.00611 | -0.01779 |
| `unresolved_exact_rmse_mean_reduction` | -0.01290 | -0.01118 | +0.03204 |
| `invest_in_candidate_marginalization` | false | false | false |

### Winner and takeaway

**`mask_unresolved`, and defer candidate-junction marginalization.** The
`20260902b` verdict reproduces on corrected code and survives the terminus
wiring, the corrected mapping data and a new split seed.

Two things matter more than the verdict. First, re-scoring the *same six runs*
from their selected checkpoints flips the sign of
`unresolved_exact_rmse_mean_reduction` and nearly triples
`unresolved_exact_spearman_mean_delta` - the evaluation-checkpoint artifact
was comparable in magnitude to the effect being measured, which means any
earlier family evaluated at a non-selected epoch carries the same distortion.
Second, the unresolved test stratum is 37 rows (28 exact) and its per-seed
deltas flip sign (-0.0094 / +0.0459 / -0.0899). The negative result is a
statement about statistical power on `HLA-A*02:01`, not evidence that junction
context is unimportant - and it is the same limitation `20260902b` reported,
now with the checkpoint confound removed.

## 2026-09-05 - PR #45 data-integrity end-to-end smoke

**Experiment id:** `2026-09-05_0957_codex_pr45-integrity-e2e`

**Agent/model:** Codex / GPT-5

**Directory:**
[`2026-09-05_0957_codex_pr45-integrity-e2e`](2026-09-05_0957_codex_pr45-integrity-e2e/)

**Commit:** `c4dc1cf6c60ccca68eb3e467fffd70b3067945df` (clean)

### Question and data contract

Does the repaired PR load both named data sources through unique paths, resolve
MHC inputs directly from `mhcseqs`, train, reconstruct the selected checkpoint,
and emit fully traceable validation/test results?

- `merged_tsv` preflight: scanned 3,266,972 rows; reservoir-capped binding 96,
  kinetics 24, stability 24, processing 24, elution 96, T-cell 96, and TCR
  evidence 24; recorded all pre-cap candidates, cap losses, ingest drops, and
  MHC-filter losses. The final 287 samples split 173/57/57 by peptide.
- Exact Hitlist-only training path: HLA-A*02:01, reservoir-capped 96 binding,
  2 kinetics, 24 stability, and 96 elution observations. `kon`, `koff`, and
  `tm` were excluded, leaving binding, half-life, and elution supervision.
  Every synthetic ratio and MHC augmentation was zero. The final 216 samples
  split 130/43/43 by peptide with data seed and split seed independently fixed
  at 42.
- Both conditions used strict resolved-only MHC inputs with `mhcseqs` first and
  no index CSV. The merged preflight resolved 232/269 reported alleles before
  explicit filtering and 296/296 model-facing rows afterward. Hitlist resolved
  63/63 alleles and 717/717 row-wise inputs. CSV fallback count was zero.
- Both audits had zero lineage issues, duplicate sample IDs, and fake-null
  sequences. Hitlist support/dataset/supervision hashes were `2587f4cc...`,
  `e1c7012b...`, and `4a1d280e...`; complete values are in the experiment
  summary.

### Training, losses, and output mapping

No pretraining was used. The CPU smoke used expanded Presto d32/l1/h4 (411,722
parameters), AdamW lr 2.8e-4 and weight decay 0.01, batch 16, MIL cap 16, one
epoch, two training batches, and one in-loop validation batch. Learned
uncertainty weighting was off. Standard unified losses were active with
cascade/assay-affinity/assay-presentation/no-B2M/T-cell-context/T-cell-upstream
consistency weights 0.2/0.1/0.1/0.5/0.05/0.2, MIL contrastive/sparsity weights
0.1/0.02, and binding contrastive weight 1.0 where labels supplied support.

Binding affinity labels map to the binding/affinity-family outputs and their
IC50/KD variants; half-life maps to `assays.t_half`; elution detection maps to
elution/presentation outputs. Source organism and MHC annotations map to their
auxiliary species/class/type outputs. The selected epoch-1 checkpoint was
reconstructed with its expanded topology before full, uncapped held-out
evaluation. Per-example validation and test prediction dumps are committed.

### Conditions and held-out metrics

RMSE is `log10(nM)`; exact metrics exclude censored observations.

| Condition | Result | Runtime |
|---|---|---:|
| Default merged-TSV preflight | Complete funnel; 287 samples; all post-filter MHC rows resolved | 59 s |
| Exact Hitlist train/evaluate | Train + selected-checkpoint full validation/test dumps completed | 53 s |

| Split | Overall loss | Exact affinity n | Spearman | Pearson | RMSE | <=500 nM bal. acc. | AUROC | AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Validation | 3.3841 | 14 | -0.2044 | -0.2718 | 1.6213 | 0.5000 | 0.3889 | 0.3433 |
| Test | 3.2907 | 13 | -0.1593 | 0.0877 | 1.6278 | 0.5000 | 0.4167 | 0.2420 |

Half-life had four observations in each held-out split (validation/test RMSE
3.3849/2.6119). Elution was all-positive (21/20), so discrimination metrics
are undefined. These tiny-run values are recorded for reproducibility, not as
a model-quality result.

### Winner and takeaway

There is no model winner: this was a path-integrity verification. All hard
gates passed. Named sources now select exact loaders, merged curation is fully
auditable, MHC joins work from `mhcseqs` without an index, lineage remains
source-faithful, the selected checkpoint reloads exactly, and complete held-out
artifacts exist. The narrow, one-class Hitlist elution slice remains unsuitable
for claims of corpus completeness or model discrimination. Requested GPU:
none; observed hardware: local CPU. Total wall time was 112 seconds.

## 2026-09-06 - PR #45 class-II lineage and Hitlist funnel closure

**Experiment id:** `2026-09-05_2213_codex_pr45-lineage-funnel-closure`

**Agent/model:** Codex / GPT-5

**Directory:**
[`2026-09-05_2213_codex_pr45-lineage-funnel-closure`](2026-09-05_2213_codex_pr45-lineage-funnel-closure/)

**Code commit:** `15003dacf310de3534325cb2db7fd6f3e2e89481`
(the experiment bundle alone was uncommitted at launch; production and test
files matched this commit)

### Question and data contract

Does the production data path preserve standalone DQ/DP beta alleles as
resolved lineage when their real `mhcseqs` records export `groove2`, and does a
real flank-enabled Hitlist load promote its unresolved-flank removals into the
canonical funnel JSON and CSV?

- MHC condition: default `mhcseqs`, no index CSV, real `HLA-DQB1*06:02` and
  `HLA-DPB1*02:01` records.
- Hitlist condition: `/Users/iskander/.hitlist`, HLA-A*02:01,
  `mask_unresolved`, flanks enabled, reservoir caps of 96 binding, 2 kinetics,
  24 stability, and 96 elution observations. Data/split seeds were 42/42;
  `kon`, `koff`, and `tm` were excluded; all synthetic ratios and MHC
  augmentation were zero.
- The loader saw 16,721 binding, 2 kinetics, 2,150 stability, and 726,766
  elution observations before caps. It resolved 63/63 alleles and 717/717 MHC
  row inputs through `mhcseqs`, with zero index fallback.
- The 216 samples split 130/43/43 by peptide. Lineage issues, duplicate sample
  IDs, and fake-null sequences were all zero. The split support, dataset, and
  supervision hashes were `2587f4cc...`, `e1c7012b...`, and `4a1d280e...`;
  full hashes are in the experiment README and artifacts.

### Conditions and results

| Condition | Result |
|---|---|
| DQB1 beta lineage | 93-residue `groove2` populated `mhc_b`; sample and batch resolved lineage passed |
| DPB1 beta lineage | 91-residue `groove2` populated `mhc_b`; sample and batch resolved lineage passed |
| Hitlist funnel | 4 binding and 235 MS unresolved-flank drops present in both JSON and CSV |

No pretraining, model initialization, optimization, loss terms, or assay-output
mapping applied. This was a data preflight, so predictive validation/test
metrics and prediction dumps were intentionally not produced. Requested GPU:
none; observed hardware: local CPU. Runtime was approximately 73 seconds.

### Winner and takeaway

There is no model winner. Both reviewed audit contracts pass on the real
external data paths: chain-aware lineage now represents DQ/DP beta model inputs
correctly, while the common funnel exposes the Hitlist flank-filter loss without
discarding the detailed source-loader diagnostics.

## 2026-09-07_1851_codex_average-precision-correction

**Date:** 2026-09-07. **Agent/model:** Codex / GPT-6.
**Directory:** [average-precision correction](2026-09-07_1851_codex_average-precision-correction/).
**Code:** `93a9c7b24c6108577dcf40c94a9ff34e6c30cc5c`; experiment-only dirty files.
**Reproduce:** directory `reproduce/launch.sh`, `launch.json`, `inputs.json` and
frozen analyzer/estimator source. OMP/MKL threads 1, local CPU, no requested or
observed GPU, 7.065 seconds.

Retrospective correction of existing canonical validation/test predictions;
no new pretraining, training, inference or model selection. Discovered 301
local CSVs; selected 39 canonical-schema files / 428,537 rows with source
SHA-256 hashes and preserved split identity. The partial sample file is
identified; 262 other-schema/empty/probe files are excluded. Original source,
curation, synthetic-data, training, loss weights and assay contracts are linked
through each source experiment. BCE task labels retain their task logits;
binding targets use the existing qualifier-aware 500 nM classifier. No label,
source, split, synthetic mixture or prediction is changed.

| Condition | Validation corrected AP entries | Test corrected AP entries |
|---|---:|---:|
| September 2 source-junction masking: row-wise → threshold-group AP | 121 / 184 | 114 / 186 |
| September 3 flank-context fixes: row-wise → threshold-group AP | 253 / 364 | 250 / 364 |
| September 5 PR #45 smoke: row-wise → threshold-group AP | 0 / 12 | 0 / 13 |

All 1,123 compared old AP values reproduce their archives. Reissued 738 metrics
in 36 explicitly versioned summary copies; 385 are unchanged. Maximum absolute
delta 0.083333, in small strata. Corrected AP matches scikit-learn 1.5.2 across
500 seeded tied/untied arrays to 3.33e-16. `ap_deltas.csv` preserves every metric,
support/tie count and delta; summary JSON links full source validation/test
dumps and corrected files. Original non-AP metrics (including regression,
other threshold metrics and available loss metadata) remain exactly unchanged.
Missing historical loss/output artifacts are not reconstructed by this audit.

Preferred estimator: `average_precision_distinct_thresholds_v2`. No model
winner or new-encoder quality result. The earlier masking/null-to-NAN validity
errata still apply; changed AP values do not restore withdrawn causal claims.

## 2026-09-07_1908_codex_binding-metadata-preservation

**Date:** 2026-09-07. **Agent/model:** Codex / GPT-6.
**Directory:** [binding metadata preservation](2026-09-07_1908_codex_binding-metadata-preservation/).
**Code:** `ddc06448fc906ea4fd51dc12c670accc137a23b9`; only experiment/plan/design
files dirty. **Reproduce:** directory `reproduce/launch.sh`, `launch.json`,
actual before/after loader sources and frozen analyzer. OMP/MKL threads 1;
local CPU, no requested or observed GPU. Runtime 159.657 seconds overall.

**Dataset/curation:** identical full `data/merged_deduped.tsv`, 1,028,889,889
bytes, SHA-256 `46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`.
Canonical merged adapters instrumented before caps; head cap one retained
record per modality bounds memory. The scan measures all routed quantitative
source observations before MHC filtering or splitting. No source/synthetic
mixture, units, values, qualifiers, curation or sample order changes.

| Condition | Binding rows / missing method | Stability rows / missing method | Kinetic rows / missing method |
|---|---:|---:|---:|
| Before metadata preservation | 249,292 / 249,292 | 12,259 / 12,259 | 106 / 106 |
| After metadata preservation | 249,292 / 0 | 12,259 / 0 | 106 / 0 |

All ordered non-descriptor record fingerprints and loader statistics match.
Binding type-column counts are unchanged. Preparation/readout columns now
receive their observed selectors: e.g. 238,258 PURIFIED preparations and
106,010 FLUORESCENCE readouts. Observed OTHER values remain distinct from
unknown metadata. Culture fields are absent throughout this source; fixtures
verify preservation without claiming real culture support. A real 473 nM
EVMPVSMAK/A*03:01 observation traces through actual MHC resolution and dataset
collation to the intended fixed columns, with resolved sequence evidence saved.

**Training/evaluation contract:** no pretraining, optimization, augmentation,
new splits or predictive evaluation. Validation/test metrics and prediction
dumps are intentionally absent for this source/selector audit. Existing
binding labels supervise affinity and selected descriptor columns with their
existing censor-aware transform. Kinetic/stability descriptors are preserved
without adding binding-panel losses. No loss terms/weights are changed.

**Preferred condition/takeaway:** preserve observed metadata. This repairs
ingestion for all routed quantitative methods in the available source while
preserving numeric/source conservation. It does not establish post-curation
per-column/split adequacy or model quality (#48/#53). Full tables, missingness,
fingerprints and real-example trace are in the experiment results. Runtime
differences are not interpreted as performance evidence.

### 2026-09-09_1106_codex_output-coverage

- **Date / agent**: 2026-09-09; Codex / GPT-6.
- **Experiment**: [2026-09-09_1106_codex_output-coverage](2026-09-09_1106_codex_output-coverage/).
- **Status**: source inventory and routing trace completed; uncapped census/update phases pending.
- **Contract**: current merged TSV and exclusive Hitlist cache, read-only; prospective 80/10/10 peptide splits, data seed 17, split/model seed 42.
- **Training / synthetic data**: none in inventory; later measured/augmented conditions and bounded optimizer diagnostics are specified in the README.
- **Validation/test metrics**: not applicable to metadata inventory; no prediction-quality claim.
- **Hardware**: local CPU; no Modal GPU requested.
- **Reproduction**: experiment `reproduce/launch.sh`, phase invocation and source snapshot; pinned isolated environment.

Initial invocation at `96b3ff6` failed before scanning (0.476 seconds): the local
launcher assumed the provider's YAML was a mapping. The corrected launcher uses
Hitlist's canonical curation loader. Its three tests passed before the clean
`aada2b19ac74346c5a71ea603498352045c4cc91` inventory completed in **38.804 seconds**.
Both receipts and immutable source snapshots are retained; successful reproduction
uses `reproduce/inventory_v2.sh`. Pinned Hitlist 1.59.1, mhcseqs 2.5.12,
mhcgnomes 3.41.0 and all transitive versions/source hashes are recorded.

| Source condition | Total rows | Rows from MS-excluded studies |
|---|---:|---:|
| Observations parquet | 4,439,643 | 40,355 |
| Binding parquet | 891,885 | 472,497 |
| Default merged TSV | 3,423,737 | 514,190 |

The read-only inventory reproduces Hitlist #444 against the frozen observations
hash `f51440ab229fd187d2548b4dddcd1fc04580d97d45fb4d5b8e0222aa8080f928` and
curation hash `e0270f2b417619a318ef03549e7cb7d46231bb3dbb6a088d8367a239cd6b3308`.
MS exclusions do not invalidate all binding evidence from the same publications.
The unchanged merged file has SHA-256
`46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`.
Its complete flagged subset passes through the actual uncapped adapter: 375
invalid peptides dropped, 15,641 affinity, 11 Tm, 496,976 elution and 1,187
T-cell records emitted, all without PMID lineage. These are pre-MHC-filter
loader counts. A separate method trace will reconcile their attribution before
production routing changes. No preferred trained condition, changed loss/weight,
synthetic labels, validation/test predictions or predictive metrics exist in
this metadata-only phase. The next decision is to repair incorrect evidence
routing/lineage before treating these populations as supervision support.

The clean `4e949da09a9960031b09ee249eb3a57ad894c44e` route trace completed in
6.311 seconds and reconciled all 43 method/response groups exactly to the actual
loader buckets. There are 492,858 non-MS-method rows routed to elution (442,612
negative responses), including 418,890 microarray rows from PMID 32903714.
Another 4,118 rows explicitly say MS and require study-level curation. Two
numeric crystallography rows also enter affinity despite their `3D structure`
type. SQLite confirms the excluded observation subset's 33,101 global distinct
peptides. Full tables and immutable trace invocation/source are registered;
the frozen raw subset/SQLite paths are in the result. These are source-routing
findings, not evidence that the model has learned meaningful predictions.

## 2026-09-09_1133_codex_assay-routing

**Date/agent/model:** 2026-09-09; Codex / GPT-6.
**Directory:** [assay routing correction](2026-09-09_1133_codex_assay-routing/).
**Status:** before/after source audit and payload reconciliation completed.

Identical complete merged TSV (3,423,737 rows, SHA-256
`46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`). Observe
canonical classification and every constructed record before a one-record head
cap per modality; no MHC filtering, study blacklist, source edits or generated
data. Compare descriptor/response/route counts and ordered record payload hashes
before/after #59. Existing supported target transforms, loss terms and weights
are unchanged. No pretraining, training, validation/test splits, predictions or
predictive metrics: this is a source-ingestion audit. Reproduction uses directory
`reproduce/launch.sh` and per-condition immutable invocation/production snapshots,
with the pinned isolated inventory environment. Local CPU, OMP/MKL threads one,
no GPU requested. Runtime and condition results will be recorded after launch;
no preferred result is claimed prospectively.

The clean `1c5d1eacf94a6956f2e3cbc55869ffa07e94d396` baseline completed in
100.868 seconds with unchanged source hashes. All 3,423,737 input rows reconcile
to 156,765 invalid peptides and 3,266,972 classifications in 927 groups. Actual
pre-cap records: binding 249,292; kinetics 106; stability 12,259; elution 2,630,813;
T-cell 207,987; TCR evidence 166,285. Another 230 classified rows have missing
quantitative labels. The baseline wrongly routes 1,239 structural scalars and
6,250 qualitative-binding scalars into affinity, and four equilibrium association
constants into on-rate. Supported concentration families total 241,803 rows;
explicit MS-method presentation totals 2,073,797. Baseline group fingerprints,
complete metadata, runtime and source snapshots are preserved for the comparison
below. No predictive performance was measured by either source audit.

The corrected condition at clean `ce30bbe9b1ea80a1f29ecefd9234bf21e496ab24`
completed in **98.096 seconds** with identical source hashes. Reconciliation at
clean `e06cbaf` retains all 927 descriptor populations; 798 unchanged groups
preserve every ordered payload hash for **2,702,233 accepted records**. The
129 changed groups remove **564,509 incorrectly routed targets**, with no new
invented target and explicit, disjoint skip reasons.

| Old route → corrected source bucket | Rows |
|---|---:|
| Affinity → qualitative binding | 6,250 |
| Affinity → structure | 1,239 |
| On-rate → equilibrium association constant | 4 |
| Elution → missing affinity value | 248 |
| Elution → qualitative binding | 555,579 |
| Elution → structure | 23 |
| Elution → non-MS presentation | 1,166 |

Corrected pre-cap records: binding 241,803; kinetics 102; stability 12,259;
elution 2,073,797; T-cell 207,987; TCR evidence 166,285. Omissions are 564,739,
including 478 missing quantitative labels. Every row, skip reason and emitted
payload reconciles. `reproduce/compare.sh` freezes comparison inputs and source;
complete JSON/CSV transition tables and both production snapshots are retained.
Preferred condition is explicit assay semantics. No loss equation, parameter,
synthetic-data contract or predictive metric changes; no validation/test split
was used because this is ingestion verification. Prior models trained with the
old incorrectly routed population are not a controlled quality baseline for the
corrected one. Hitlist #444 and Presto #60/#48/#50/#53 remain separate work.

The correction merged in PR #61 as `8cebd17660b67c0e4908017e0de9fccf5b00c653`,
with a tree identical to reviewed `3c30d73`. Final branch/PR CI 34360660081 and
34360770090 passed lint/tests and docs; the earlier identical production-code
CI run 34360127179 reported 2,029 passed, 3 skipped. The next branch starts from
this merge and restores the publication-lineage path tracked in #60.

## 2026-09-09_1419_codex_merged-lineage

**Date/agent/model:** 2026-09-09; Codex / GPT-6.
**Directory:** [merged publication lineage](2026-09-09_1419_codex_merged-lineage/).
**Status:** ingestion metadata recovery and strict before/after reconciliation complete;
merged in PR #63.

Same unchanged `data/merged_deduped.tsv` as #61: 3,423,737 input rows, SHA-256
`46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`.
Every accepted record is observed before a one-record head cap per modality,
seed 17. No MHC filtering, curation/cache changes, new labels or synthetic data.
Corrected #61 assay routes, units and qualifiers are unchanged. Existing
binding/kinetics/stability, processing, elution, T-cell and receptor-evidence
label/output mappings and loss terms/weights are unchanged; this family runs
no pretraining, optimization, or predictive evaluation. No validation/test split
or prediction-quality metrics apply to an ingestion metadata audit. Deterministic
fixtures separately verify actual model inputs/labels and held-out row/bag exports.

Reproduction is `bash reproduce/launch.sh before`, `after`, then `compare` at
their recorded commits. Each phase freezes `results/<phase>/invocation.json`
and its own `reproduce/` bundle. Pinned isolated Python 3.12.6 / torch 2.7.0 /
Hitlist 1.59.1 release environment, complete dependency versions and linked
package-source receipt; OMP/MKL threads one, local CPU, no GPU/Modal run.
Raw logs: `artifacts/2026-09-09_1419_codex_merged-lineage/`.

| Condition | Clean commit | Seconds | Result |
|---|---|---:|---|
| Before metadata repair | `15f2855` | 147.505 | Every available publication value dropped |
| After shared propagation | `87ce95b` | 187.212 | Every available publication value retained |
| Exact source/payload comparison | `859be29` | 1.730 | All 42 field/modality groups and frozen snapshots reconcile |

| Modality | Accepted records | Recovered PMIDs | Recovered DOIs | Recovered reference texts |
|---|---:|---:|---:|---:|
| Binding | 241,803 | 62,712 | 0 | 179,091 |
| Kinetics | 102 | 102 | 0 | 0 |
| Stability | 12,259 | 3,131 | 0 | 9,128 |
| Processing | 0 | 0 | 0 | 0 |
| Elution | 2,073,797 | 2,061,734 | 0 | 12,063 |
| T-cell | 207,987 | 171,899 | 0 | 36,088 |
| TCR evidence | 166,285 | 127,095 | 1,316 | 125,806 |

**Preferred condition:** repaired publication propagation. Every accepted record
retains identical ordered non-lineage payloads; all source hashes, populations,
skip reasons, caps and field availability agree. After repair there are zero
dropped, invented or changed metadata values. Publication fields overlap and
must not be summed as distinct observations. Absent original assay IDs remain
absent, with fallback identity visible to the census. Processing's zero real
source support is not filled by its propagation fixture. Metadata-inclusive
diagnostic hashes change; actual input/target/selector tensors do not.

This resolves #60's adapter omission. It does not establish model fitting or
held-out quality, close #48/#50/#53, or bypass Hitlist #444's upstream curation
problem. Alternate binding-selector descriptor loss was separately reproduced
and filed as Presto #62. Local verification passed 434 affected regressions and
48 focused-probe/audit checks; final full CI passed. The README preserves the
initial no-tests-collected command failure and links the corrected passing log.

PR #63 merged as `48182ae1d2a36994470338b9422329590bc7c413`, with a tree
identical to reviewed `2cce640`. Final branch/PR CI 34365828888 and 34365866862
passed lint/tests/docs; the final PR log reports 2,039 passed, 3 skipped in
946.36 seconds and is preserved under the raw-artifact root. The next repair
is #62, then the uncapped census/update and predictive acceptance work resumes.

## 2026-09-09_1457_codex_binding-selectors

**Date/agent/model:** 2026-09-09; Codex / GPT-6.
**Directory:** [alternate selector descriptors](2026-09-09_1457_codex_binding-selectors/).
**Status:** before/after descriptor recovery and exact population reconciliation
complete; PR review/full CI pending.

Read the unchanged 3,423,737-row merged TSV SHA-256
`46c5722ce92a28a6002c028a8584ea5d6f62d6f8d950aaca82518cd25b2e359c`.
Run the actual panel/bootstrap selectors for HLA-A*02:01 / HLA-A*03:01, seed 17.
Panel observations include every construction before a one-record head cap;
bootstrap uses max_records=2000, max_peptides=500, max_rows_per_peptide=4 and
selects 1,000 records. These overlapping scopes are not a full-corpus census.
No MHC resolution, curation/cache change, pretraining, training, synthetic data,
validation/test split or predictive metrics. Prediction dumps are intentionally
absent for this ingestion/selector audit. Numeric targets, qualifiers, units,
loss equations/weights and source families are unchanged; source descriptors
select their observed output columns. Cultures are copied without claiming a
new culture-specific quantitative loss.

Pinned isolated Python 3.12.6 / torch 2.7.0 / Hitlist 1.59.1 release environment,
all dependency versions, explicit isolated PYTHONPATH, OMP/MKL threads one,
local CPU and no GPU/Modal run. Raw logs live under
`artifacts/2026-09-09_1457_codex_binding-selectors/` in the primary checkout.
Run each phase's exact `results/<condition>/reproduce/launch.sh` at its recorded
commit. Baseline commit is preserved on `codex/binding-selector-audit`; the
repair branch was rebased onto merged #63's identical tree before source edits.

| Condition | Clean commit | Seconds | Result |
|---|---|---:|---|
| Before omission | `e6d6b60` | 111.439 | Every observed type/method dropped |
| After eight-line repair | `37a06cd` | 49.342 | Every observed descriptor retained |
| Source/selection reconciliation | `c6b0eeb` | 0.191 | Populations, non-descriptor payloads and contracts identical |

| Scope | Constructed records | Retained records | Recovered type values | Recovered method values |
|---|---:|---:|---:|---:|
| Panel | 23,210 | 1 | 23,210 | 23,210 |
| Bootstrap | 1,000 | 1,000 | 1,000 | 1,000 |

**Preferred condition:** repaired source descriptor propagation. All available
fields match the normalized source exactly, with zero lost/invented/changed
values. Every actual post-repair type/method/preparation/geometry/readout column
matches the source's collator factorization. Affinity-family column populations
already agreed via measurement fallback in this real subset. Culture fields
are absent here; fixtures verify known/missing cultures and conflicting labels.
These timing observations are not a controlled throughput benchmark.

Every selector statistic, ordered constructed/retained non-descriptor payload
hash and source-descriptor hash reconciles. Publication lineage, numerical values,
qualifiers, units and selected records are unchanged. Complete per-field and
per-column JSON/CSV counts, source hashes, frozen launchers and the comparison
are preserved. New regressions first failed 10 cases (12 passed), then the
repaired code passed 247 affected tests in 21.46 seconds. Four overlapping audit
checks pass, including negative controls. Ruff lint/format passes. PR #64 merged
as `190360a7` with a tree identical to reviewed `b1f4646`; #62 is closed. Both
final-head CI runs passed 2,061 tests / 3 skipped. #48/#50 uncapped coverage/update evidence and #53 real-data
fitting/held-out performance remain separate work.

## 2026-09-09 15:41 — Canonical full-source output coverage (registered)

- Agent/model: Codex / GPT-6.
- Directory: [2026-09-09_1541_codex_canonical-coverage](2026-09-09_1541_codex_canonical-coverage/).
- Status: `merged_measured` completed as attempt `serialized_entry`; the other
  five conditions/update diagnostics remain pending. Production base is merged
  PR #64, `190360a7`; frozen executed source is `d6a690a`.
- Dataset: frozen merged TSV, exclusive Hitlist and explicit Hitlist-plus-bulk,
  each measured-only and augmented, no modality caps. Ten input files total
  1,464,177,529 bytes; exact hashes and complete arguments are in the directory.
  Data seed 17, peptide-group train/validation/test 80/10/10, split seed 42,
  strict complete MHC resolution plus canonical filtering, ambiguous flanks masked.
- Assay/qualifier policy: repaired canonical loaders; unsupported source families
  remain explicit omissions. No source union or cache edits. Direct/proxy/auxiliary
  and generated labels stay separate; aliases do not inflate endpoint coverage.
- Synthetic contract: measured conditions disable generation; augmented explicit
  pMHC/elution/cascade-elution/cascade-T-cell ratios 1.0/0.5/0.5/0.5, missing-beta
  0.25, processing 0.5, MHC-only 60,000 at fraction 0.05. Wrong-enzyme bulk labels
  are disabled in measured conditions and enabled at ratio 1.0 in augmented bulk.
- Training/pretraining: none in the census. Actual objective mapping, weights and
  multiplicity come from the executable registry. Later gradient diagnostics must
  freeze real train candidates and optimization settings before execution.
- Hardware: first merged measured condition alone, Modal CPU 4/8 requested/limit,
  RAM 64/192 GiB requested/limit, four-hour timeout, no GPU. Observed memory was
  192 GiB; peak RSS was 14.31 GiB over 12,652.535 seconds. No preferred condition
  or predictive winner yet.
- Validation/test: support counts only; no predictive loss or metrics. Prediction
  dumps are inapplicable to this phase; #53 still requires fresh predictive eval.
- Raw artifacts: `artifacts/2026-09-09_1541_codex_canonical-coverage/` and uniquely
  named Modal volume prefixes recorded by each launched receipt.
- Decision: the merged measured audit is complete and reconciled. Reduce repeated
  reporting work, then schedule the remaining source/augmentation conditions.
- Preparation update: 76 instrumentation/canonical checks pass, including exact
  report parity and exclusion of unrelated raw corpora from the code archive.
  Automatic approval review rejected the ten-file Modal upload pending explicit
  authorization for its payload and destination. No source transfer, remote
  census or training occurred; the frozen code/input manifest are reviewable.
- Further author review fixed live-launcher/archive mismatch and null diagnostic
  IDs before launch, with 85 local checks passing in 9.85 seconds. The pinned
  client's read-only lookup confirmed workspace `iskandr`; app and volume
  environment is explicitly `main`. This is preparation evidence, not a source
  census or predictive result. Data-upload approval remains pending.
- Upload closure: the user explicitly approved transfer on 2026-09-09 at16:32 UTC.
  All ten files (1,464,177,529 bytes) reached the specified `iskandr/main` workspace/
  environment, `presto-data` volume and family prefix without overwriting. Local
  before/after hashes and remote names/sizes match; content hashing will repeat
  before the census. `results/upload/` preserves the exact dirty source diff,
  launcher, invocation and receipt. Nineteen focused checks pass after correcting
  receipt-variable shadowing. No census or training condition has launched yet.
- Initial launch at clean `37cbb17`: app `ap-2NjIAHmEYvAgM1ghTB4SEz` failed
  installing Presto because the source archive omitted declared package
  `inference`. No worker call or data load occurred; no source-support or
  predictive metrics exist. The initial receipt and raw build log are preserved.
  Derive archive roots from the authoritative package manifest, verify a wheel
  locally from an isolated snapshot copy, then retry with an explicit attempt
  name. Dataset bytes and source/hardware conditions remain unchanged.
- Packaging verification at `d8f06ee`: wheel build succeeds with all nine declared
  packages/all archived production Python sources; five canonical packaging
  tests pass in 2.09 seconds from an isolated copy. The Modal image built, but
  `package_manifest` failed before worker entry because the SDK relocated its
  file import to `/root/launch.py`. App `ap-e5tn40CmWftp3a6wCcf0qy` was explicitly
  stopped; failed-attempt receipt includes call ID and elapsed time. No data was
  loaded. Explicit serialized transport now passes a fresh-process round-trip
  with Modal 1.1.4 (one test, 0.48 seconds), plus 28 focused tests in 4.29 seconds.
  Retry as `serialized_entry` with the same condition/input/hardware contract.
- Startup history for `serialized_entry` at clean `d6a690a`: app
  `ap-PPrESGuLvIKrTYEYNnwTPG`, call `fc-01M23HN1ZCVA1M5NF5Q1P0GFNJ`.
  Worker startup and canonical full merged-data loading are confirmed after the
  frozen code/input/curation hash checks. Observed Python 3.12.1, CPU torch
  2.7.0+cpu, Hitlist 1.59.1, mhcseqs 2.5.12, mhcgnomes 3.41.0, Modal 1.1.4;
  `/proc/meminfo` reports 192 GiB. Startup receipts are copied into the attempt
  directory. Counts and reconciliation were pending at startup; their completed
  results follow. No fitting or predictive evaluation ran. CI at earlier `37cbb17` passed
  2,061 tests / 3 skipped in 865.08 seconds, with its complete log retained.
- Completed merged-measured result: 12,652.535 seconds, peak RSS 14.31 GiB,
  1,785 report count calls. Train/validation/test retained rows:
  **1,665,530 / 208,200 / 208,192**, total 2,081,922. IC50 observations:
  55,423 / 6,978 / 6,791; T-cell response: 118,222 / 15,266 / 14,795;
  TCR evidence: 133,170 / 10,456 / 18,744, all positive. Processing and pathway
  T-cell MIL support are zero; the T-cell counts above describe row labels.
  These are census counts, not predictive metrics or adequacy thresholds.
  No fitting/pretraining ran; prediction dumps are inapplicable.
- Report count calls are a wrapper call counter, not a runtime measure. The 1,785
  figure used the base `8b5f60f` census. The report-reuse census (PR #66) reuses sole
  evidence-group counts without calling `counts()`, so later conditions report fewer
  calls for the same work (7-8% fewer on the verification fixtures); compare wall
  time, peak RSS and SQL work instead. Differential parity against the base (Claude /
  Opus 5, `f54dd32` plus uncommitted fix): identical reports for 32 configurations and
  60,928 equal filtered class counts, in the family's
  `results/verification/report_reuse_parity.json`, reproduced by
  `analysis/verify_report_reuse.py`. A synthetic 150,000-sample check of the
  11-class endpoint (`results/verification/class_count_cost.json`) cuts report-style
  class work from 144 SQL statements to 16 while a cold single-column claim keeps
  the base's 3. Software verification only; no real-data speedup is claimed.
- The local watcher lost DNS connectivity, but the remote status and durable
  call result confirm success. Preserve both receipts. All final summaries,
  source/curation metadata, split support and training candidates are copied into
  the attempt's `remote/` directory; the 8.8 GB SQLite evidence remains in its
  original volume and the final worker log is retained locally. Input hashes
  before/after, evidence fingerprints, legacy dataset fingerprints and all
  output CSV/JSON cells reconcile. Reduce redundant report queries before more
  conditions; no preferred source condition or predictive winner is established.

2026-10-20 integration closure (Codex): report reuse was checked against newer
main `3418d0f`, retaining species/assay provenance and affinity-target auditing.
All 32 complete/repeated reports and 60,928 filtered class counts agree; see the
family's `results/verification/report_reuse_parity_20261020.json` and README for
source hashes and the exact verifier command. Historical verification bytes
remain preserved. This is software parity, with no new fitting or timing claim.

### 2026-10-15_2100_codex_scoped-proteomics-admission

- Date/agent: 2026-10-15 America/New_York, Codex / GPT-6.
- Experiment: [published-scope admission](2026-10-15_2100_codex_scoped-proteomics-admission/).
- Exact input: Hitlist PXD013455 HeLa Trypsin/P public manifest
  `441f3c66c6e6bc6a90c7cb8c54d0131676c3cb4e7a04ddbd9346061ad8f9d84b`;
  code `27cde9717ae7e59997581e4ae633e0cd644a7faa`. Full occurrence collapse,
  reservoir seed 42, cap 10,000. Original/reanalysis citations, cell/assay/search
  conditions and unresolved donor/specimen identities remain pinned.
- Training/pretraining/synthetic/loss contract: none. This is a registered data
  admission and canonical collation/support audit, not model fitting. Binary
  observed-in-search labels map to ms_detectability_logit as proxy support only;
  MS/MS, matching and scoped absence are separate. Excision and depth are absent.
- CPU only, one Torch thread; no Modal/GPU. Admission and support: 74.38 seconds,
  peak RSS 595,787,776 bytes. Reproduction/source/environment and all output hashes
  are recorded in the experiment directory; reusable TSV/parquet/reports live in
  shared datacache.

| Condition | MS/MS | Matching | Scoped absence | Total |
| --- | ---: | ---: | ---: | ---: |
| Full collapsed projection | 143,216 | 20,778 | 1,287,164 | 1,451,158 |
| Seed-42 admission | 960 | 154 | 8,886 | 10,000 |

All admitted rows retain candidate linkage; 1,114 positives retain original
observation IDs. Zero canonical lineage issues/duplicate sample IDs. Full input
has 153,980 peptides with ambiguous occurrences, whose parent/flanks are masked.
Validation/test metrics and prediction dumps: not applicable, because no split,
fitting or inference was performed. No winner/model baseline promotion. Takeaway:
the actual public archive is admitted with honest masks and lineage; the next
canonical trainer experiment must establish frozen membership and bounded fitting.
A single HeLa search and unresolved biological identities do not establish
independent experiment/donor/specimen generalization. Status: admission audit closed.


## 2026-10-15 — Codex: frozen within-search detectability preflight

Family: [scoped-detectability-qualification](2026-10-15_2255_codex_scoped-detectability-qualification/README.md). Clean producer `a15c1a45f9d9eda9d111459b6f1d9f86a56e8e29`; Presto 0.1.8 with distinct Trypsin/P and versioned scoring. Canonical data-only preflight on the pinned public PXD013455 HeLa search succeeded in 108.78 seconds, runner peak RSS 998,916,096 bytes, CPU one thread. Frozen train/validation/test membership is 8,000/1,000/1,000 peptides with positive proxy support 912/105/109 and scoped absences 7,088/895/891. No generated controls, excision/depth labels, MHC inputs or fitting. All prospectively registered support thresholds are met; other outputs remain undeclared.

No cross-partition peptides, candidate IDs or measured observation IDs; known proteins overlap and all splits share one search. Donor/specimen independence remains inconclusive. Complete membership TSV/parquet, source registry, label policies, overlap and output-coverage reports are in datacache, pinned by the family receipt. Run archive SHA-256 `fb0881f60ed993614e683a2f9e588e6f02b83d91228a78b01833db5ab47e9244`. This is admission/coverage evidence, not predictive quality. The registered bounded canonical fit and train-only baselines follow with the exact support fingerprint required.


### 2026-10-15_2255_codex_scoped-detectability-qualification — actual fit closure

- Agent/model: Codex (GPT-6); [experiment](2026-10-15_2255_codex_scoped-detectability-qualification/README.md).
- Clean fit commit `65173d2598746ebb848031783833f74c833679f2`; exact frozen launch, versions, datacache paths and public hashes in the experiment.
- PXD013455 HeLa Trypsin/P search, 10,000 collapsed peptides, source manifest `441f3c66c6e6bc6a90c7cb8c54d0131676c3cb4e7a04ddbd9346061ad8f9d84b`; 8,000/1,000/1,000 train/validation/test, 912/105/109 positives. MS/MS/matching/scoped absences are proxies, no synthetic labels, excision or depth supervision; MHC and binding concentration/censoring metrics not applicable.
- No pretraining; expanded d32/l1/h4, batch64, lr0.001, two epochs capped100 batches each, seed42. CPU/one Torch/BLAS thread, no GPU; 1,027.01s and runner peak RSS7,912,128,512bytes. Selected epoch1 by validation loss.
- Declared output only `ms_detectability_logit`; 200 gradient-bearing proxy batches. Balanced sampling exposed 6,314 positive labels of12,800, versus frozen train prevalence11.40%. Other outputs unsupported.
- Protocol deviation: six exposed consistency weights zero but hidden binding-orthogonality weight0.01 active; uncertainty task weighting and weight decay0.01 also active. Original results/config preserved; #95 tracks the omitted effective configuration.
- Validation Presto AUROC/AP/log loss/Brier:0.80628/0.33324/0.55073/0.18739. Test:0.82984/0.38070/0.54864/0.18618. Train-prevalence val/test log loss0.33634/0.34454; train-only length/composition logistic0.27268/0.26899. Complete metric tables, 1,000 paired bootstrap intervals, coefficients and every held-out prediction retained publicly.
- Preferred probability baseline: length/composition logistic. Presto improves AP point estimates but fails both baselines on probability quality. No readiness claim; #94 tracks a prospective follow-up. All splits share the search and proteins, donor/specimen independence unresolved.
- [Download](https://github.com/pirl-unc/presto/releases/tag/scoped-detectability-qualification-v1): partitions, source/cell/assay context, policies, lineage/leakage/support reports, predictions, checkpoint, logs and inventories; fit SHA-256 `ab664220951f912433022684aa408ed0280c7786f9772661694acb87caa7351c`.

## 2026-10-16 — Codex: actual human-corpus frozen-input preflight

Agent/model Codex (GPT-6); [complete contract and results](2026-10-16_0045_codex_human-corpus-preflight/README.md).
Clean source `2c73f2fb57c5d47c09c492e1f55d1b3ccecfe312`, Presto0.1.10;
Hitlist1.76 baseline manifest `039efc7854098951b5c10b3fb5b38294452eeb425e367ed1fafdeebddf34b38d`.
Both MHC classes, human peptide source, original endpoints/units/censoring,
strict sequence resolution, no flanks or synthetic/auxiliary data. Official MHC
FASTA commits/hashes and full environment retained. CPU8/64GiB, one thread, no
GPU; runtime1,950.43s, runner peak RSS21,010,509,824bytes (children excluded).
No pretraining, fitting, predictions or model/baseline comparison; all priors0.

| Selected evidence | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Samples | 7,200 | 900 | 900 |
| Binding | 4,121 | 177 | 192 |
| Direct KD | 506 | 0 | 4 |
| Elution positives / negatives | 2,447 / 0 | 679 / 0 | 679 / 0 |
| Half-life | 482 | 41 | 24 |
| Temperature | 131 | 3 | 5 |
| Off-rate | 19 | 0 | 0 |

No peptide/observation/pMHC/known-origin overlap, but studies overlap and all
donor/specimen/acquisition identities are unknown. Only79 selected roots have
known origins. All221 output declarations remain undeclared; positive-only
elution and missing held-out direct KD/kinetics support preclude broad readiness.
Validation/test performance is inapplicable because model construction was
forbidden. No winner. Protocol deviation: recorded reservoir flag ignored; actual
per-family Random(59).sample selection preserved (#103). Producer predates
curation fixes #694/#696. Original disk/worker failures also retained (#98/#99).

All46 original archive files verified; full catalogs, membership TSV/parquet,
source/cell/assay lineage, label/partition policies, support/leakage reports and
runtime receipts are [downloadable](https://github.com/pirl-unc/presto/releases/tag/human-corpus-preflight-v1).
Archive SHA-256 `444ac92e859da9afd3fcdf3c27ee3ae282f70a6134ac9c42f3c528cdcc1dae80`.
Takeaway: ingestion/partition pipeline accepted; preserve limitations and qualify
a separate corrected corpus before target-specific fitting. #678/#680/#48/#53 open.

## 2026-10-16 — Codex: scoped sampling and probability qualification

Agent/model Codex (GPT-6); [full contract and artifacts](2026-10-16_0110_codex_scoped-sampling-qualification/README.md).
Clean actual source `82be704840d2aa7e87a2f646ab2f9e5866116d9a`, Presto0.1.11.
Same pinned PXD013455 HeLa Trypsin/P source and 8,000/1,000/1,000 peptide-disjoint
membership as the original qualification,912/105/109 positives. All source,
assignment, parent and support fingerprints preserved; only producer policy hash
prospectively rebound. Proxy-only detectability; no MHC, binding, excision/depth,
synthetic or unrelated assay supervision. No pretraining. Expanded d32/l1/h4,
batch64, lr0.001, weight decay0.01, uncertainty weighting, seeds42, two epochs
with100 batches each. Every regularization weight explicitly zero.

| Condition | Positive exposure | Val log loss/Brier | Test log loss/Brier (exploratory) |
| --- | ---: | ---: | ---: |
| Balanced Presto | 6,314/12,800 | 0.57755/0.19747 | 0.57618/0.19732 |
| Shuffled Presto | 1,444/12,800 | 0.28928/0.08376 | 0.28081/0.08258 |
| Train prevalence | train11.40% | 0.33634/0.09406 | 0.34454/0.09714 |
| Length/composition logistic | train only | 0.27268/0.08142 | 0.26899/0.08129 |

Each neural condition completed200 finite-gradient proxy batches, zero direct
labels. Selected checkpoints balanced epoch1/shuffled epoch2; minimum unweighted
validation log loss selects shuffled. Shuffled val/test AUROC0.77236/0.81882 and
AP0.31172/0.37417. Logistic remains preferred for validation probability quality;
shuffled-minus-logistic paired log-loss95% interval[0.00410,0.02800] validation,
[-0.00015,0.02361] exploratory test. Complete predictions/metrics/coefficients and
1,000 peptide-bootstrap intervals retained. Binding regressions/500nM metrics
inapplicable. The already inspected test set never selects a condition.

CPU8/32GiB, one Torch/BLAS thread, no GPU; total2,542.76s, runner-only peak
RSS7,226,044,416bytes. Original shared-scratch preflight stopped before fitting;
partial evidence retained (#102), then local POSIX SQLite scratch fixed. Every
successful archive member and per-example log loss/Brier independently verified;
122 files, SHA-256`16935d480378d950367aba8e15bf2e85a654797bd1e7fde216413d123bbc9c0b`.
[Download](https://github.com/pirl-unc/presto/releases/tag/scoped-sampling-qualification-v1).
One search/seed, overlapping proteins, unknown donor/specimen identities and proxy
labels limit conclusions. No default change or readiness claim; #48/#53/#678 open.

## 2026-10-16 — Codex: label-blind component-allocation audit

Agent/model Codex (GPT-6); [complete contract/results](2026-10-16_0635_codex_component-allocation/README.md).
Clean source `83431016f30f9bae1ba47601ce4b2346db28cd52`, Presto 0.1.13; exact 9,000
human-preflight roots from Hitlist 1.76 manifest `039efc7854098951b5c10b3fb5b38294452eeb425e367ed1fafdeebddf34b38d`.
Original admission/caps/MHC filtering/labels unchanged; 5,843 components, largest 31.
V1 exactly reproduces original assignments. V2 compares remaining fractional
capacity; no assay labels or predictive outcomes enter allocation. Seed 42, 80/10/10.

| Endpoint | V1 train/val/test | V2 train/val/test |
| --- | ---: | ---: |
| All observations | 7,200/900/900 | 7,200/900/900 |
| Direct KD | 506/0/4 | 406/71/33 |
| Off rate | 19/0/0 | 15/1/3 |
| Stability | 613/44/29 | 570/57/59 |
| Elution positives | 2,447/679/679 | 3,003/418/384 |

3,060 roots move; every observation/component remains intact. Both peptide/root
leakage audits pass with known origins together; studies overlap and all donor/
specimen/acquisition identities remain unknown. Original source/registry predates
later curation fixes. Endpoints/proxies, units, censoring, classes and human-readable
source context retained. Positive-only elution and extremely sparse kinetics still
prevent broad qualification. No model/pretraining/optimizer/regularization, generated
labels, predictions or validation/test performance; regression/500 nM metrics do not
apply. Preferred allocation V2 for future Hitlist; historical/scoped policies preserved.

Local CPU/no GPU, 7.35 s, peak RSS 769,818,624 bytes; bounded 65,536-row scans. Hitlist 1.78
audit code, datacache 1.27.1, pandas 2.3.3 / numpy 2.5.3 / pyarrow 25.0.1 / Python 3.12.6.
All 17 output archive members and identity invariants verified; SHA256
f7a66f2a87e0d8b343730240c41f94e61cd649096f30c861504d102fb9d3994d.
[Downloads](https://github.com/pirl-unc/presto/releases/tag/component-allocation-audit-v1)
include proposed assignments, support/lineage reports and original runtime/source
receipts through datacache. This is support balance, not biological independence or
training readiness; #48/#53/#678/#680 remain open.

All 12 public assets were independently downloaded and hash-verified through
datacache; all 17 archive members and intact observation/component identities
were verified again from public bytes. Full local suite: 2,276 passed, four explicit
skips in 525.13 seconds. Software CI/merge/distribution remain tracked in PR #106.

## 2026-10-16 — Corrected frozen human corpus preflight

- Agent/model: Codex (GPT-6).
- Experiment: [2026-10-16_0750_codex_corrected-human-preflight](2026-10-16_0750_codex_corrected-human-preflight/).
- Actual recipe: clean `e2e1e827f7b677f7c4f4d37dfc4ca9f386c58ffb`; released Presto 0.1.13 / Hitlist 1.78.0. Corrected native manifest `b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372`. Exact source/config/dependency/producer pins and local environment overrides are in the reproduction bundle and manifest. A failed local prelaunch is preserved; #107 fixes shell failure propagation before Modal.
- Data: human-source/HLA, both classes; 13,190 binding / 3,909,515 elution / 19 kinetics / 741 stability eligible before caps. Explicit reservoir caps 5,000/5,000/1,000/1,000, effective independent family stream seed 59. Strict MHC filtering retains 4,492/3,856/19/686, totaling 9,053 roots/model samples. Full original/resolved MHC and source lineage remain available.
- Partition: explicit relative_deficit_v2, peptide/lineage components, seed 42, requested 80/10/10; actual train/validation/test 7,241/906/906. No generated samples or missing observation IDs. Every selected ID/peptide joins exactly to the public native source with cited cell/MHC/condition/assay context.
- Training/pretraining: none; canonical data_preflight_only and a forbidden model constructor. All synthetic ratios, augmentation and regularization weights zero. All 221 canonical output declarations remain undeclared. No prediction files, checkpoints or held-out predictive metrics exist; validation/test are retained for support and leakage audits.
- Prospective candidate: assays.IC50_nM, log10(nM), direct binding:IC50 from IEDB/CEDAR. Minimum 50 observations, 50 distinct peptides and 50 exact values in each split, fixed before launch. Actual counts pass without adjustment. The new supported-output manifest binds exact fingerprints for the next fit; its canonical gate has not been executed. Empty inequality is treated as exact under the frozen loader policy, with original emptiness retained.

| Condition / split | Direct IC50 observations | Distinct peptides | Exact values | Class I / II |
| --- | ---: | ---: | ---: | ---: |
| Corrected preflight, train | 2,074 | 1,031 | 1,599 | 877 / 1,197 |
| Corrected preflight, validation | 265 | 126 | 210 | 99 / 166 |
| Corrected preflight, test | 275 | 126 | 214 | 101 / 174 |

- Runtime/hardware: 985.768 seconds; runner peak RSS 11,749,728,256 bytes excluding children; eight CPUs, 64 GiB RAM, 512 GiB scratch, one Torch/BLAS thread, zero loader workers, GPU explicitly none. Catalog reused through unchanged datacache recipe with four exact public-baseline hashes.
- Leakage: zero peptide/observation/pMHC/known-origin overlap; shared studies 159 train/validation, 153 train/test, 109 validation/test. Only 87/9,053 roots have known origins. All donor/specimen/acquisition identities are unknown. Unknown is not an independence pass.
- Unsupported/sparse evidence: off-rate 19/0/0; direct KD exact 144/6/5; elution positive-only 3,088/394/374; no measured processing or T-cell/TCR response records. Source generation, cap algorithm and selected roots differ from the old preflight/allocation audit, so its task counts and model results do not transfer.
- Artifacts: [corrected-human-preflight-v1](https://github.com/pirl-unc/presto/releases/tag/corrected-human-preflight-v1), complete 48-member run archive SHA-256 `ce8d55cf1e7a14fb898f997918133f708526d145c65f01f97c37e2b0df6db0c8`, inventories, assignments/membership, full coverage, label policies, lineage audits, source-context tables, prospective manifest and reproduction recipes. Datacache owns acquisition, analysis and publication.
- Decision: proceed to the separately prospective bounded IC50 fitting diagnostic only after the unchanged canonical manifest gate passes. No winner or predictive-quality claim exists in this data-only family. #48/#53 and Hitlist #678/#680 remain open.

Publication closure: all 23 public downloads, 48 archive members, 9,053 source/partition joins and the fixed candidate counts independently verify through datacache. See the family verification receipt. No manifest gate or fitting result is implied.


## 2026-10-16 — Canonical affinity target-policy audit

- Agent/model: Codex / GPT-6. [Family](2026-10-16_1000_codex_affinity-target-policy/README.md).
- Clean producer: candidate Presto 0.1.16 `ca9e883034a8842b99394deb11b8155abd83b597`; frozen recipe tag `experiment-affinity-target-policy-20261016`. Datacache 1.27.1 retains the two public pinned inputs and complete output generation.
- Contract: all 2,614 published corrected-human IC50 labels, original IDs/units/censor operators and train/val/test membership. Canonical float32 `log10(clip(nM, 0.001, 50000))`, original direction retained. Label-only samples preserve class/allele identity; no MHC sequences, synthetic data, model, pretraining, training, loss optimization or predictions. The archived manifest supplies configuration only and is not rebound.
- Condition: canonical collator/census versus the independent released-0.1.13 posthoc target audit. All effective targets/directions and 211 clipped source IDs agree. Train/val/test clipping is 157/31/23, including 39/8/7 exact and 118/23/16 right-censored values; zero below-minimum values. Validation/test prediction and quality metrics are absent because no model is constructed. No preferred model.
- Hardware: local arm64 CPU, one Torch/BLAS thread, no GPU/workers, batch 128, ten-minute alarm. Observed wall time 2.014 s including inputs; peak RSS 271,433,728 bytes.
- Artifacts: [public audit](https://github.com/pirl-unc/presto/releases/tag/affinity-target-policy-audit-v1), complete canonical changed-row table, flat counts, full 86-field clipped-source context, producer/runtime receipt, source recipe, plan and hashes.
- Decision: canonical target-policy evidence agrees with frozen measurements. A new prospective fitting contract must explicitly accept effective targets and separately evaluate raw measurements/bounds. Unknown donor/specimen/acquisition identity and shared studies persist; this is not predictive readiness.


## 2026-10-16 — Native presenting-cell processing context audit

- Agent/model: Codex / GPT-6; [registered family](2026-10-16_1635_codex_native-presenter-context/README.md).
- Clean successful recipe `060fbfed9d2f99f2937e637624a9e88d1df786c3`, Presto 0.1.21 candidate. Original fit producer remains Presto 0.1.16 / Hitlist 1.78.0. The initial packaging failure at `fb39e8d302ef77a1b5a3160eb3fe442398418cf0` is retained with its exact recipe/log and published alongside the successful result (#125).
- Pinned original native parquet, frozen membership/IC50/species tables and prior role audit; datacache acquisition/generation. All 9,053 IDs/order/partitions and every prior/native column retained; 2,614 IC50 raw/effective float32 targets, units and operators unchanged.
- Native processing context: 109 human, 15 rat, 8,929 unknown. Historical host-based context changes on 8,936 rows (7,148 train / 887 validation / 901 test). All IC50 rows have unknown presenters. Twenty-seven known presenter facts retain candidate-arm consensus without an arm-specific review/reference; publication-level provenance remains, no arm identity is invented.
- Schema 4 binds the prospective presenter policy independently of molecular MHC. All 9,053 canonical processing inputs and human molecular auxiliary labels checked. Unit regressions cover row/MS-MIL/T-cell-MIL/device/export/parent/fingerprint paths.
- CPU data-only analysis: no cloud/GPU, model, MHC-sequence reconstruction, pretraining, fitting, optimizer, loss weights, synthetic training examples or new predictions. No new validation/test performance or preferred model; the old failed quality screen and biological-independence limitations persist. Both held-out partitions are included only for source/target preservation.
- Builder 53.443 seconds, peak process RSS 707,395,584 bytes. Complete/changed/IC50 tables, source policies, hashes, reproduction, independent verifier and failed-attempt evidence are published at [native-presenter-context-audit-v1](https://github.com/pirl-unc/presto/releases/tag/native-presenter-context-audit-v1), with all 18 public downloads, every original/native source join and unchanged IC50 targets independently verified.
- Decision: retain explicit native presenter context and unknown gaps. Antigen/repertoire roles and legacy API/sampler semantics remain #115/#46/#123; no broader training-readiness claim.


## 2026-10-16 — Assay-context applicability audit

- Agent/model: Codex / GPT-6; [registered family](2026-10-16_1828_codex_assay-context-applicability/README.md).
- Clean executed recipe `18d04533d3ef1f042c49910042b96022627aee3f`, Presto 0.1.22 candidate. Frozen source inputs, immutable recipe and 11 implementation hashes are recorded; historical fit producer remains Presto 0.1.16 / Hitlist 1.78.0.
- Contract: all 9,053 native-presenter audit rows and 2,614 IC50 rows, joined by observation ID to the pinned corrected native parquet. Every prior field, partition, raw/effective target, unit and censor operator remains unchanged. Datacache owns input acquisition, output generation and publication.
- Condition: explicit preparation/elution applicability. IC50: 2,533 purified-MHC observations have cellular processing context marked not applicable; 81 cellular-MHC observations require it and remain unresolved. A missing method or endpoint alone does not imply applicability. No new model inputs are introduced. Schema 5 and prospective membership/support receipts bind the inspection policy.
- Independent replay verifies all original/native joins, prior cells, annotations, coverage and 11 artifact hashes without importing Presto. Canonical collation agrees on all 9,053 metadata rows.
- CPU-only, Python 3.12.6, one Torch/BLAS thread; no GPU/cloud, pretraining/training, optimizer, loss weights, synthetic examples or predictions. Builder runtime 2.980692 seconds excluding imports/acquisition; peak process RSS 401,276,928 bytes. Both held-out partitions are used only for metadata preservation, so no new validation/test loss, regression or binding metrics exist and no model winner is selected.
- [Public evidence](https://github.com/pirl-unc/presto/releases/tag/assay-context-applicability-audit-v1): source-linked observation/IC50 tables, coverage, policy, source pins, code, independent verification and hashes. Original failed IC50 quality result and unknown donor/specimen/acquisition identities persist. This resolves applicability reporting, not broader training readiness; #48/#53/#115 and Hitlist #678/#680 remain open.

### 2026-10-21_0353_codex_modular-memorization-control — registered, preparation pending

- Agent/model: Codex / GPT-6. [Family](2026-10-21_0353_codex_modular-memorization-control/README.md), [prospective contract](agents/codex/plans/modular-assay-only-memorization.md).
- One supplied-metadata assay-only condition: 32 exact purified IC50 observations, 16 per class, explicit class-II chain pairs, pinned Hitlist v4/catalog; old held-out peptides excluded. No pretraining or synthetic labels.
- AdamW 0.0003, constant schedule, zero dropout/weight decay, seed 42, full batch; 2,000 updates/20-minute optimization ceiling. Local CPU, one thread, no GPU/Modal.
- Target/output: direct raw log10(nM) IC50, standardized training squared residual; no auxiliary loss. Fixed overall/per-class RMSE/Spearman/spread and real-update gates are specified before fitting.
- No validation/test split intentionally: training-only capacity diagnostic, no generalization claim. No results, winner or qualified checkpoint yet.

#### Closure: 2026-10-21_0353_codex_modular-memorization-control

- **Date/agent/model:** 2026-10-21 UTC; Codex / GPT-6. [Full writeup, source and artifact links](2026-10-21_0353_codex_modular-memorization-control/README.md).
- **Frozen execution:** commit `122b611124563b653fbddbbce30f921ea0f14fff`, tracked files clean; only redirected fit log untracked. Exact argv/environment/versions in `reproduce/launch.json`, rerun helper in `reproduce/launch.sh`, executed launcher snapshot under `reproduce/source/`.
- **Dataset/curation:** pinned Hitlist v4 binding parquet, pinned mhcseqs catalog and historical held-out exclusions. 814,268 raw rows →26,619 eligible exact purified IC50 candidates; 841 inconsistent input groups excluded for this diagnostic; 32 hash-selected consistent inputs, 16 I/16 II with explicitly named partners. All raw nM targets and original assay lineage retained. New cohort, not historical-fit membership; no clipping or partner imputation.
- **Training/pretraining:** fresh shared d128/l2/h4 modular network, supplied human molecular species/class, direct IC50 and learned II register pooling; no auxiliaries/pretraining/replay/synthetic data. Full batch 32, AdamW constant 0.0003, weight decay/dropout 0, seed 42, one squared standardized log10(nM) loss per exact assay. Censored/proxy/stability/MS/TCR/other assays excluded from this diagnostic only.
- **Hardware/runtime:** local CPU, one Torch/BLAS thread, no Modal/GPU requested or observed; 100 updates, 33.05 s, peak RSS 716,193,792 bytes. Prespecified 2,000-step/20-minute ceiling; first overall AND per-class gate pass at step 100.

| Condition / training subset | N | RMSE log10(nM) | Spearman | Pearson | Prediction/target SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Assay-only, overall | 32 | 0.03294 | 0.99927 | 0.99976 | 0.99978 |
| Assay-only, class I | 16 | 0.04333 | 1.00000 | 0.99973 | 1.00000 |
| Assay-only, class II | 16 | 0.01711 | 0.99706 | 0.99987 | 0.99897 |
| Training-mean comparator, overall | 32 | 1.42546 | undefined | undefined | 0 |

- **Evidence:** initial overall RMSE 1.43145/SD ratio .02760; final standardized loss .00053406. All 32 allocated parameter tensors and exposed features have gradient-driven updates; register weights updated on all 100 steps. A fresh process reproduced every prediction/metric and passed the public support guards. Initial/intermediate/final predictions and all threshold metrics are retained; raw-checkpoint/disposition paths/hashes are in the results receipts.
- **Validation/test:** intentionally absent because this is a training-only capacity gate; all observations are fit targets. Historical held-out peptides excluded; future independent evaluation must account for this diagnostic exposure. No generalization winner or endpoint qualification.
- **Decision/context:** fixed capacity gate passed with nonconstant predictions in both classes, without identity tasks. This does not demonstrate auxiliary benefit or isolate the cause of the historical failed architecture's collapse. Full-cohort mean/ridge comparisons, held-out evaluation and the wider holistic model requirements remain pending.

### 2026-10-21_0512_codex_weirdo-reference-pretraining

- **Date / agent:** 2026-10-21 UTC; Codex / GPT-6.
- **Directory:** [bounded peptide-source pretraining](2026-10-21_0512_codex_weirdo-reference-pretraining/README.md).
- **Status:** initial preparation stopped before scanning/fitting; amended preparation pending. No model evaluation executed.
- **Source/data contract:** pinned local Weirdo reference and category archive;
  uniform label-blind 5,000/1,000/1,000 train/validation/test eight-mers, seeds
  42/43/44. Every selected membership/fold/bucket must independently reconstruct
  against all reference proteins before fitting. Source mnemonic-prefix grouping
  does not establish homology independence. Hashes and exact policy are in the
  [prospective plan](agents/codex/plans/weirdo-reference-pretraining.md).
- **Training contract:** shared modular encoder, peptide-source head only; ten
  independent membership labels per peptide with mean CE. No assay labels,
  pretraining weights, replay, generated labels or synthetic sequences. AdamW
  3e-4, weight decay 1e-4, dropout .1, batch 64, seed 42; at most 20 complete
  epochs, validation patience five after five epochs. CPU, one Torch/BLAS thread;
  fitting ceiling 15 minutes. No requested/observed Modal GPU.
- **Reproduction:** family `reproduce/` and `code/`; actual git/env/package records
  are generated at preparation and fitting launch. Source and code remain pinned.
- **Qualification:** prospective validation CE comparison against prior and
  composition controls, paired source-component bootstrap; no test-based choices.
  Per-category support, calibration and discrimination remain required. No
  biological species-origin or assay-transfer claim follows from this screen.

| Condition | Data / outcome | Held-out validation | Held-out test |
| --- | --- | --- | --- |
| Training prevalence | Same reference memberships; smoothed category priors | Not run | Not run |
| Composition logistic | Same rows; 20 residue frequencies, fixed L2 logistic models | Not run | Not run |
| Modular peptide-source | Shared encoder plus ten independent source memberships | Not run | Not run |

No winner or predictive conclusion yet. This family remains incomplete until
admission, all executed metrics/predictions, checkpoint replay and closure are
recorded; the full MHC/TCR/APC/MS/affinity goal remains open.

Preparation attempt 0 (`42689fd`) stopped after 0.1085 seconds because the actual
pinned archive lacks `train_groups` (and all other grouping arrays). Actual array
headers, frozen source, invocation and failure/log are retained in the family.
No selected-row audit, optimization or test prediction occurred. The prospective
plan records an amendment before retry: unchanged source hashes/selection/labels,
explicit missing-group state, and reference-derived grouping required before
admission. No supplied bucket is invented; actual supplied buckets must still
agree with reference reconstruction. The retry has its own attempt directory.

#### Closure: 2026-10-21_0512_codex_weirdo-reference-pretraining

- **Date/agent/model:** 2026-10-21 UTC; Codex / GPT-6. [Full family](2026-10-21_0512_codex_weirdo-reference-pretraining/README.md).
- **Frozen source/reproduction:** preparation `de3cdfc`, fit `1d3c96b`; exact full hashes, dirty status, command/environment and package versions in the family receipts. Tracked fit sources were clean; generated cohort/log untracked. Frozen executed model/trainer/launcher source, failed preparation and amended attempt are retained.
- **Data:** same pinned Weirdo reference/archive/meta; fixed label-blind 5,000/1,000/1,000 eight-mers, ten overlapping reference-category memberships. All 7,000 memberships/folds agree with exact occurrences across 569,516 proteins. No selected cross-fold peptide. Train/validation/test have 3,486/705/723 connected mnemonic-family components; homology independence unverified. Missing archive buckets reconstructed explicitly, never represented as supplied evidence. No reselection/relabeling.
- **Assays/synthetic contract:** auxiliary-only; no affinity/stability/MS/TCR outcomes, exact/censored numeric assays, synthetic sequences or generated labels. Reference absence is not absence across all biological organisms. The old table-verification statistic is not this selected-row reference audit.
- **Training:** independent smoothed category priors; fixed `C=1` L2 logistic on 20 residue frequencies (all ten models converged); modular encoder d128/l2/h4 with ten independent source logits, mean binary CE, no initial weights/replay. AdamW 3e-4, weight decay 1e-4, dropout .1, batch 64, seed 42, max 20 epochs, patience 5 after 5. The replay weight is inactive in this auxiliary-only condition. Validation selects epoch 5 / 395 updates; stop at epoch 10 / 790 updates.
- **Runtime/hardware:** CPU, one Torch/BLAS thread, no requested/observed GPU/Modal. Reference audit 8.96 s; baseline fit 0.067 s; network child 64.30 s (trainer 58.82 s); whole comparison 69.93 s. Peak child/parent RSS 908,427,264/468,910,080 bytes; 900 s ceiling not reached.

| Fixed condition | Train CE | Validation CE | Test CE |
| --- | ---: | ---: | ---: |
| Training prevalence | 0.303358 | 0.292560 | 0.308435 |
| Composition logistic | 0.294973 | 0.286331 | 0.300864 |
| Modular source network | 0.295201 | 0.285826 | 0.302032 |

- **Screen:** failed. Paired validation network-minus-prior CE=-0.006734, 95% component-bootstrap interval[-0.009581,-0.003564]; versus composition=-0.000506[-0.001971,0.001117]. 2,000 resamples, 705 connected components, seed 20261021; all categories meet the fixed support minimum. Test was evaluated once after selection and did not change the recipe. No accepted neural winner; composition has the lowest descriptive test CE.
- **Evidence:** per-example predictions for all three splits/conditions, per-category CE/discrimination/confusions/Brier/ECE/spread, coefficients, history, coverage, plot, complete receipts and raw artifact hashes are retained. Independent sklearn AP/AUROC/CE agreement on all 90 condition/split/category cases. All 29 allocated parameter tensors/exposed features updated; fresh-process first-training-batch replay matches exactly (no test rescore). Test probability SD 0.0092–0.1108 across categories. No IC50/500 nM metrics apply to this auxiliary-only family.
- **Takeaway:** real source pretraining and nonconstant outputs are demonstrated on this bounded reference task, but improvement over composition and affinity transfer are not established. No checkpoint promotion. The 32-row IC50 memorization control addresses a different question; full held-out affinity, MHC/TCR auxiliaries, APC/MS/TCR qualification and the overall model goal remain open.

### 2026-10-21_0618_codex_mhc-reference-admission — registered before preparation

- **Date/agent/model:** 2026-10-21 UTC; Codex / GPT-6. [Family](2026-10-21_0618_codex_mhc-reference-admission/README.md), [prospective contract](agents/codex/plans/mhc-reference-admission.md).
- **Data:** pinned 56,440-row MHC reference; fixed exact human/mouse/rat/rhesus taxon mapping, functional/flag/domain checks, source-label conflict deferral and domain-alias collapse. All raw rows retain dispositions. No missing partner or class/pairing label generated.
- **Partition contract:** connected exact-domain/original-chain components; fixed SHA25670/15/15 assignment, no homology-independence claim. Readiness requires at least20 domains and3 components per species/split; no salt/vocabulary reselection after inspection.
- **Training/evaluation:** none; CPU data-only preparation, no GPU/Modal, optimizer, losses, synthetic sequences or predictions. Validation/test partitions are preparation outputs; predictive metrics are inapplicable until a separate registered fit.
- **Status:** not executed. Actual source/code/environment, dispositions, replay checks, counts and artifact hashes must be recorded before closure. This is one source-admission step in the full model goal.

#### Closure: 2026-10-21_0618_codex_mhc-reference-admission

- **Date/agent/model:** 2026-10-21 UTC; Codex / GPT-6. [Complete family](2026-10-21_0618_codex_mhc-reference-admission/README.md).
- **Frozen execution:** clean `1b218e7f77337398889bffdefa37309597dfa6bc`; exact command, environment, runtime versions and executed sources in the reproduction bundle. Pinned existing 56,440-row MHC catalog SHA256 `ddc3493c64f4328de174a6c2cda1f007ec2f93a1ef649add25d2ef9c1720add1`; its producer version is unknown, independently of installed package versions.
- **Data/curation:** four exact human/mouse/rat/rhesus taxonomy strings, explicit functional/flag/source-member/domain guards. One species annotation per unique actual groove slot/sequence, retaining every alias. Class-II single chains remain single chains. No fabricated partner, class/pairing target, synthetic sequence or inferred species label. Nonclassical source genes remain explicit and do not certify peptide presentation.
- **Conservation:** 12,612 original rows contribute new annotations, 7,211 are duplicate aliases and 36,617 are excluded. All 56,440 dispositions, 13,218 unique annotations and 33,696 admitted source-domain aliases replay independently against the original CSV. No conflicting species sequence; 3,687 connected components. Canonical species-only preflight passes.
- **Partition:** exact-domain/original-protein components; fixed hash 70/15/15 assignment, unchanged after inspection. Mechanical minimum of 20 annotations and three components per species/split passes. Homology independence is unverified.

| Species | Train annotations / components | Validation annotations / components | Test annotations / components |
| --- | ---: | ---: | ---: |
| Human | 4,848 / 2,060 | 419 / 416 | 6,160 / 430 |
| Mouse | 142 / 80 | 34 / 19 | 42 / 19 |
| Rat | 131 / 70 | 35 / 17 | 37 / 18 |
| Rhesus macaque | 971 / 389 | 206 / 92 | 193 / 77 |

- **Observed limitation:** a 5,664-annotation human B/C component supplies 91.95% of human test and 88.06% of all test annotations; a human A component supplies 54.10% of human training. The retained partition is strongly concentrated despite passing the mechanical minimum. This is not a broad generalization or calibration certificate; no reseeding or rebalancing was performed.
- **Training/evaluation contract:** data-only, no optimizer, pretraining, model predictions, loss weights or synthetic labels. No affinity/stability/processing/MS/TCR assay supervision or censoring applies. Both held-out partitions exist, but predictive metrics and per-example prediction dumps are inapplicable because no model was evaluated. No preferred model or checkpoint.
- **Hardware/runtime:** local CPU, one Torch/BLAS thread, no requested/observed GPU or Modal. Preparation 4.4094 seconds, peak process RSS 641,744,896 bytes. Independent source replay is separate from that timing.
- **Artifacts:** complete dispositions, annotation aliases, empty conflict artifact, preflight, source hashes, actual launch/source snapshots and independent replay/concentration tables/logs. Generated shared cohort manifest SHA256 `4e4d4a5d2b3d47fc16ddf63ddeeaf6111836138661356ab6805945f13c7ef06c`; observation SHA256 `5ba2ef6f40b047ddc256c61fb06e802c82bc2a400830a178d44a0a82455ef510`.
- **Decision:** source-traceable species annotations are available for a separately registered comparison; no MHC-class/pairing training, affinity transfer, calibrated species inference or biological qualification is established. This closes the admission family only. Controlled assay-only versus auxiliary affinity evidence and the full model objective remain open.

### 2026-10-21_0647_codex_full-ic50-admission — registered before execution

- **Date/agent/model:** 2026-10-21 UTC; Codex / GPT-6. [Family](2026-10-21_0647_codex_full-ic50-admission/README.md), [frozen policy](agents/codex/plans/full-ic50-admission.md).
- **Data:** hash-pinned Hitlist v4 direct IEDB/CEDAR purified human HLA IC50, all eligible raw exact/censored values and concentration units; canonical peptides and unambiguous explicit catalog-resolved MHC pairs. No source caps, no consistency filter on independent measurements, no partner imputation or target clipping.
- **Lineage/partition:** canonical original-assay deduplication and conflicting-copy deferral. Connect exact peptides/original assays; preserve historical and 32-row memorization exposure, defer incompatible fixed assignments, hash only new components at fixed80/10/10. Exact-only and exact-plus-censored manifests share all validation/test rows. No study/donor independence claim.
- **Training/evaluation:** data preparation only, local CPU/one thread, no requested/observed GPU/Modal, optimizer, pretraining, synthetic labels or model predictions. No predictive metrics/dumps apply. Subsequent fitting needs its own registered recipe and matched baselines; auxiliary qualification does not block the supplied-metadata control.
- **Status:** pending preparation and independent replay. Preserve all raw dispositions, aliases, target/split/source checks, full counts and fixed readiness outcomes before closing this family.

#### Closure: 2026-10-21_0647_codex_full-ic50-admission

- **Date/agent/model:** 2026-10-21 UTC; Codex / GPT-6. [Full evidence](2026-10-21_0647_codex_full-ic50-admission/README.md).
- **Executed source:** clean `2214086783b83b741a6c9f0d0dda7137cd718c09`; exact argv, environment, runtime/parser versions, source hashes and application/launcher snapshots retained. Existing catalog producer version remains unknown.
- **Dataset/curation:** all eligible pinned Hitlist v4 direct purified human HLA IC50 with canonical unmodified peptides, finite positive raw concentration, exact/censored qualifiers and explicit unambiguous MHC partners. All 814,268 raw dispositions replay independently; 29,283 measurements admitted, 751,790 excluded and 33,195 deferred (33,165 missing explicit partners; 30 unresolved names). No row caps, partner imputation, clipping, generated labels or synthetic sequences. Actual admitted measurements are all IEDB/nM and have unique original assay IRIs.
- **Independent outcomes:** 1,002 same-input sets have different reported values and/or qualifiers and remain separate votes. This count does not imply all censor intervals are inconsistent. No original-outcome copies/conflicting exports occurred in this actual eligible source; targeted software tests cover their handling. All 84 used catalog names/aliases and strict groove extraction verified.

| Split / class | All measurements | Exact | Exact peptide groups | Exact binders / nonbinders at 500 nM |
| --- | ---: | ---: | ---: | ---: |
| Train I | 15,266 | 14,375 | 6,532 | 7,815 / 6,560 |
| Train II | 8,289 | 7,261 | 1,117 | 1,789 / 5,472 |
| Validation I | 2,051 | 1,968 | 863 | 1,045 / 923 |
| Validation II | 874 | 758 | 141 | 212 / 546 |
| Test I | 1,845 | 1,758 | 777 | 991 / 767 |
| Test II | 958 | 842 | 133 | 204 / 638 |

- **Conditions:** exact-only has 21,636 training measurements; exact-plus-censored adds 1,919 bounds. Entire validation/test partitions are identical (2,925/2,803, both including censored observations). Source/manifest/label/split replay proves that only censored training exposure differs. Raw values above 50,000 nM remain: 1,577/214/188 train/validation/test, exact and censored combined.
- **Partitions:** 9,908 peptide components, original-assay uniqueness verified; fixed old assignments and 32-row memorization exposure retained, only new components use declared hash80/10/10. Historical assignment counts 2,117/230/219 and new counts 21,438/2,695/2,584. No target-based assignment or reseeding. Old held-outs remain previously inspected, not fresh.
- **Limits:** 114 train/validation, 117 train/test and 90 validation/test shared studies; donor/specimen/acquisition identity absent. This is not study/donor/homology independence or unseen-allele qualification. Frozen mechanical count/threshold-support readiness passes for both classes; predictive quality is untested.
- **Training/evaluation:** none; no optimizer/pretraining, loss weights or predictive model selection. Other affinity/stability/proxy/APC/MS/TCR outcomes are outside this IC50 diagnostic contract, not relabelled as IC50. Both held-outs exist; prediction dumps, losses, correlations and classifier metrics are inapplicable because no model was evaluated. No winner/checkpoint.
- **Hardware/runtime:** local CPU, one Torch/BLAS thread, no requested/observed GPU or Modal; 42.6482 seconds, peak process RSS 1,202,814,976 bytes. A separate four-worker regression ran concurrently; not an isolated runtime benchmark. Independent replay timing excluded.
- **Artifacts:** full source dispositions, components, catalog aliases, paired preflights/manifests, source/recipe snapshots, independent verification and logs. Local generated cohort 137 MiB, raw preparation 407 MiB linked in receipt. The first independent-analysis setup failed from a source-edit error; failed source/log preserved, corrected analysis passes without rerunning preparation or fitting.
- **Decision:** proceed to a separately registered supplied-metadata assay-only mean/ridge/network comparison. Neither auxiliary success nor this source audit establishes affinity learning. The full MHC/TCR/APC/MS and scientific qualification objective remains open.

### 2026-10-21_1700_codex_discrete-distribution-comparison — registered before fitting

- **Date/agent/model:** 2026-10-21 UTC; Codex / GPT-6. [Family](2026-10-21_1700_codex_discrete-distribution-comparison/README.md), [prospective contract](agents/codex/plans/discrete-distribution-comparison.md).
- **Data:** entirely artificial integer-symbol strings, length 16, alphabet 0–7;
  all 70 spans of lengths 1–5, four generic contexts per parent. Parent-disjoint
  96/24/24 split gives 384/96/96 rows; short substrings intentionally overlap.
  Fixed data/teacher seeds 20261021/20261022. Full target distributions are known
  from a frozen linear hierarchical oracle. No biological sequence or assay.
- **Training:** same two-stage start/length policy with two 83→32→1 Tanh MLPs;
  matched initialization and minibatches for seeds 42/43/44. Adam 0.003, zero
  weight decay, batch 16, 60 complete epochs/1,440 updates. Exact reverse KL
  versus REINFORCE with 8 or 64 sampled spans and a leave-one-out baseline.
  Seed-42 reward-only/no-entropy control intentionally optimizes another objective.
  No pretraining, real assay families, quantities/censoring or affinity thresholds.
- **Selection/evaluation:** validation reverse KL every five epochs; test once
  after selection. Save complete distributions, multiset samples and training-row
  replay. CPU/one thread; no requested or observed Modal GPU. Logical target-query
  counts are not an external-oracle runtime benchmark; both methods compute dense
  policy logits and share the precomputed teacher data.
- **Status:** verification/source freezing pending; no fitted model or predictive
  result yet. Full source/env/commit receipts, per-example outputs, independent
  metric/teacher checks and closure are required before concluding this family.
- **Claim boundary:** this tests generic learning and sampling on a known target;
  it does not implement biological processing, infer gene-loss effects, calibrate
  MS detection or establish a real nonbinder product distribution.

#### Closure: 2026-10-21_1700_codex_discrete-distribution-comparison

- **Date/agent/model:** 2026-10-21 UTC; Codex / GPT-6. [Complete report and artifacts](2026-10-21_1700_codex_discrete-distribution-comparison/README.md).
- **Frozen execution:** `adab28ad8ac389a5f3a395b0b8dfd6e89253221c`; executable source committed/frozen, dirty state limited to task/planning documents. Source SHA256 `e66c3a5cffc0d512989094ad653ba7ea4f3ff56778d3b298fecb8e809eee5945`. Exact launch/env/version/status/diff receipts retained in the reproduction bundle and verified archive.
- **Data/curation:** unique integer-symbol parents, alphabet 0–7, length 16; all 70 spans of lengths 1–5. Fixed 96/24/24 parent split and four generic contexts produce 384/96/96 observations. Data seed 20261021; fixed linear hierarchical probability teacher seed 20261022. Parent splits are disjoint but short substrings overlap. No real sequences, assay families, units, qualifiers/censoring, synthetic affinity labels or biological supervision.
- **Training:** identical start/conditional-length 83→32→1 Tanh MLPs, final scalar biases disabled; matched initialization/minibatches across three seeds. Adam 0.003, no weight decay, batch 16, 60 complete epochs/1,440 updates. No pretraining, dropout or early stop. Unit-weight exact reverse KL or its REINFORCE estimator (8/64 discrete samples, leave-one-out baseline); single-seed reward-only diagnostic removes entropy and deliberately changes the objective. All matched models select epoch 60 by minimum full validation reverse KL; reward-only selects epoch 5. Test scored once after selection.
- **Outputs/label mapping:** known synthetic target probabilities supervise the full span distribution. Arbitrary arithmetic compatibility/response factors define an evaluation-only selected distribution; they are not learned pMHC/MS outputs. Noncompatible spans retain positive mass. IC50/log10(nM)/500 nM classifier metrics are inapplicable. All split/context probability loss, RMSE, Pearson/Spearman, entropy and mass metrics are preserved.

| Condition | Seeds | Validation reverse KL | Validation TV | Test reverse KL | Test TV | Test noncompatible-mass MAE | Mean fit seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | Unfitted | 5.431799 | 0.836928 | 5.514706 | 0.831593 | 0.252382 | — |
| Exact differentiable | 42/43/44 | 0.000863 | 0.009968 | 0.001915 | 0.010716 | 0.002794 | 1.515 |
| REINFORCE 8 | 42/43/44 | 0.001129 | 0.012481 | 0.002274 | 0.013602 | 0.003634 | 1.122 |
| REINFORCE 64 | 42/43/44 | 0.000910 | 0.010509 | 0.001941 | 0.011359 | 0.003061 | 3.937 |
| Reward only 64 | 42 | 0.241687 | 0.236445 | 0.268665 | 0.253689 | 0.089269 | 3.412 |

- **Screen/preferred condition:** all three seeds of all matched methods pass the prospective validation recovery screen; reward-only fails. Exact enumeration is the preferred finite-support reference, with lowest mean validation/test error. REINFORCE 64 is close; REINFORCE 8 uses fewer logical fitting target terms with noisier gradients. No test-driven selection or hyperparameter changes.
- **Mechanism:** target test noncompatible mass 0.731255; exact/RL8/RL64 retain 0.731109/0.730798/0.731052. Target entropy 1.663642 versus matched 1.669352/1.666400/1.668913 and reward-only 1.135009. The prespecified fixed-batch 128-repeat gradient probe has summed variance 1.768517 for 8 samples and 0.207713 for 64. Logical fitting lookups/run are 1,612,800 exact, 184,320 RL8 and 1,474,560 RL64; both methods still compute dense policy logits and share precomputed teacher data, so this is not an external-oracle speedup result.
- **Runtime/hardware:** local CPU, one Torch/BLAS thread, macOS 26.5 arm64, Python 3.12.6, Torch 2.14.1, NumPy 2.5.4. No requested/observed GPU or Modal. Whole family 26.784 seconds, peak RSS 388,694,016 bytes. Fit timings include validation in one sequential execution; not a repeated isolated runtime benchmark.
- **Verification/artifacts:** six focused tests, explicit experiment Ruff checks, independent reconstruction of 576 teacher rows, all 99 original artifact hashes, 30 condition/split and 120 context metric checks, 61,440 decoded discrete draws, all six parameter tensors updated, and fresh-process replay of all ten selected checkpoints pass. Complete per-example train/validation/test predictions, checkpoints, histories, samples and receipts are preserved in a 10,181,857-byte compressed bundle; all 100 archived files match original bytes. Tables, plot, execution/analysis/replay/package receipts and exact source snapshots are directly linked from the report.
- **Takeaway relative to prior work:** both generic learning routes are executable and nonconstant on this known-oracle problem; reward-only optimization fails to reproduce the distribution. This closes the basic algorithm comparison requested alongside #132. It neither resolves earlier real IC50 underfitting nor qualifies species/APC/MS/TCR biology; application model/trainer code is unchanged and no synthetic checkpoint is promoted.

### 2026-10-22_1120_codex_grpo-ingredient-factorial — registered before fitting

- **Date/agent/model:** 2026-10-22 UTC; Codex / GPT-6. [Family](2026-10-22_1120_codex_grpo-ingredient-factorial/README.md), [prospective specification](agents/codex/plans/grpo-ingredient-factorial.md).
- **Data:** exact prior synthetic integer-span contract: 144 parent strings, 70 candidates each, four contexts, 384/96/96 rows by parent; fixed data/teacher seeds. No biological assays, concentration/censor labels or synthetic binding outcomes. Compatibility/response are arbitrary evaluation-only tags.
- **Training/conditions:** 2×2×2×2 clipping/standardization/reuse/reference-KL factorial, exact and original REINFORCE controls, two hybrid sample budgets and separate query-matched reuse controls; seeds 42/43/44, total 69 fits. Adam .003, batch 16, no pretraining, main 1,440-update budget; query-matched reuse controls 5,760 updates. Declare changed beta objective and its analytic optimum explicitly.
- **Evaluation/hardware:** validation selects by declared objective every 120 steps; test once after selection. Fixed pre-fit on/off-policy gradient probes; full split predictions, parameter/work accounting and fresh-process replay. Local CPU/one thread, no requested/observed GPU/Modal. IC50 and binding-threshold metrics inapplicable.
- **Status:** implementation/source verification pending; no fitted outcomes. This family does not close the active biological availability/concentration goal or the requested later partial-observation/multiset comparisons.

#### Closure: 2026-10-22_1120_codex_grpo-ingredient-factorial

- **Date/agent/model:** 2026-10-22 UTC; Codex / GPT-6. [Complete 69-fit report](2026-10-22_1120_codex_grpo-ingredient-factorial/README.md).
- **Source/reproduction:** committed frozen source `4a31eaa097e5494d490a2a5101bc612ec84348ce`; dirty status limited to task/pending planning docs. Exact invocation, environment, source/plan hashes and working-tree diff preserved. Launcher SHA256 `893ca18d317ea726daaeb34c8ff45c222ab4d70de0ab95bed49ad6a7cb81b5da`.
- **Dataset:** unchanged integer-symbol teacher/source contract, 70 spans per parent, four contexts, 384/96/96 observations by disjoint parent. Original teacher/data seeds and hashes retained. Short-span overlap remains. No biological data, pretraining, assay/censor targets, synthetic affinity labels or nM/binder metrics.
- **Training/loss/output:** full clipping × standardization × reuse × reference-KL factorial; three seeds per condition, six exact fits, original REINFORCE control, two hybrid budgets and separate query-matched controls. KL(q||target)+beta KL(q||initial reference), beta 0/.1, with analytic regularized optimum evaluated separately. Same two MLPs, Adam .003, batch 16, no weight decay/dropout/early stop; 1,440 main updates or 5,760 query-matched reuse updates. Validation every 120 steps selects the declared objective; test once after selection. GRPO-style recomputed density costs and joint-span ratios, not canonical frozen-advantage/token-wise GRPO.

| Selected condition | Validation TV to optimum | Test TV to original | Test TV to optimum | Mean target terms | Mean fit seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact beta 0 | 0.009968 | 0.010716 | 0.010716 | 1,612,800 | 1.025 |
| Original REINFORCE-8 | 0.012481 | 0.013602 | 0.013602 | 184,320 | 1.064 |
| Standardization, fresh samples | 0.014671 | 0.016237 | 0.016237 | 184,320 | 1.291 |
| Reuse 4, no standardization/clipping | 0.023842 | 0.025839 | 0.025839 | 46,080 | 1.123 |
| Reuse 4 plus clipping | 0.021277 | 0.023572 | 0.023572 | 46,080 | 1.174 |
| Reuse 4, query-matched, unclipped | 0.018304 | 0.019524 | 0.019524 | 184,320 | 4.222 |
| Reuse 4, query-matched, clipped | 0.019333 | 0.021820 | 0.021820 | 184,320 | 4.344 |
| Hybrid 2 starts | 0.015567 | 0.017576 | 0.017576 | 174,864 | 1.034 |
| Hybrid 8 starts | 0.011287 | 0.012454 | 0.012454 | 698,104 | 1.044 |
| Exact beta .1 | 0.010880 | 0.041213 | 0.011521 | 1,612,800 | 1.009 |

- **Results:** all 69 fits complete and all three seeds of every condition pass the prospective loose recovery screen. Exact enumeration remains the preferred reference. Group standardization worsens mean recovery here. Clipping is inactive at reuse 1 and mildly helps one reuse-4 comparison; it does not rescue query-matched reuse. Reuse trades target terms for error; extra optimizer work at matched target terms does not beat fresh REINFORCE. Hybrid eight-start enumeration improves fidelity at greater target cost, whereas two-start enumeration loses to REINFORCE at similar cost. KL regularization changes the target rather than improving original-target accuracy. Complete factorial, all metrics and context/seed values are in the family report/JSON/CSV.
- **Mechanism evidence:** initial-policy REINFORCE mean-gradient relative discrepancy .079714; standardization .748277 (mostly scaling, mean cosine .988222), hybrid two starts .155966, hybrid eight starts .062945 across 128 repeats. Standardization's smaller raw variance is not a fidelity gain. The fixed off-policy perturbation never activates clipping; this limitation is retained. Active clipping is confirmed in training and a separate analytic branch test. All six tensors update; no inference about biological learning follows.
- **Hardware/runtime:** local CPU/one Torch/BLAS thread, no requested/observed GPU/Modal. Python 3.12.6, Torch 2.14.1, NumPy 2.5.4, macOS 26.5 arm64. Whole family 108.284 seconds, peak RSS 392,314,880 bytes. Dense policy logits are evaluated in all methods; logical target terms are not external-oracle runtime savings. One sequential execution, not a repeated isolated runtime study.
- **Verification/artifacts:** 16 pre-fit tests plus a post-run active-clipping test pass. Independent reconstruction of 576 teacher rows, 1,035 split/context metric sets, all 423 raw hashes, source/plan/selection/budget checks, 69 fresh-process checkpoint replays, nine initial-reference split replays and 18 bitwise matches to prior exact/RL predictions pass. All 424 original files retained in a verified 18,320,086-byte compressed bundle, with complete train/validation/test arrays, references, checkpoints and histories. Archive SHA256 `c2849f8f25d737c3e85ebdaab15ef9388c72cf61ad33607a5e97ac556502de36`.
- **Decision:** retain exact/original REINFORCE as fixed references for the next partial-observation and multiset studies. No synthetic checkpoint is promoted. The biological availability/concentration goal, precursor normalization and real-source qualification remain open.

## 2026-10-22_1317 — Partial observations and whole-multiset objectives (registered)

- **Date / agent:** October 22, 2026; Codex / GPT-6.
- **Directory:** [2026-10-22_1317_codex_partial-observation-multiset](2026-10-22_1317_codex_partial-observation-multiset/README.md).
- **Contract:** Same synthetic integer strings/70 spans, 384/96/96 parent-disjoint
  rows as prior algorithm families. Known positive or partially blind observation
  weights; 32/256 nested counts per row. No biological assays, censoring or units.
- **Training:** 60 fits; exact CE, exact L2, unbiased multiset L2 K16/K64/G4 and
  naive MSE K16/G4; seeds42/43/44, Adam .003, batch16, 1,440 steps, no pretraining.
  Selection uses validation count CE only. Oracle is excluded from fitting.
- **Hardware:** local one-thread CPU; no Modal/GPU. Exact invocation/source hashes
  frozen in the reproduction bundle; execution commit recorded at launch.
- **Status:** eight mathematical/data-boundary tests passed before fitting;
  held-out metrics and conclusions pending. Full predictions/counts required.

### Closure — partial-observation/multiset family, October 22, 2026

All60 fits completed on frozen source `942cb62c925f01b6aaaadd96d3a1b41270e02fa0`;
Codex / GPT-6. Full contract and artifacts are in the [report](2026-10-22_1317_codex_partial-observation-multiset/README.md).
Same synthetic parent-disjoint integer dataset, positive/blind known observation
weights and nested32/256 empirical counts. No biological assays/qualifiers/censoring
or molecular units. Counts supervise conditional observed distributions; oracle
latent probabilities are excluded from fitting and selection. Exact CE, exact L2,
corrected multiset L2 K16/K64/G4, and naive histogram MSE K16/G4; naive expectation
adds `(1-sum q²)/K`. No additional entropy/RL penalties. All loss weights1.
Seed-matched two83→32→1 MLPs; Adam.003, batch16,1,440updates, no pretraining;
minimum validation count CE checkpoint. Every trainable tensor updates.

| Condition | Val count CE | Val observed TV | Test count CE | Test observed TV | Test latent TV | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| blind-n256-ce | 1.29438 | 0.02280 | 1.20015 | 0.02335 | 0.04418 | 1.004 |
| blind-n256-l2 | 1.29542 | 0.02638 | 1.20277 | 0.02990 | 0.05116 | 0.995 |
| blind-n256-multiset16 | 1.30260 | 0.04744 | 1.21093 | 0.05012 | 0.07811 | 1.108 |
| blind-n256-multiset64 | 1.29975 | 0.04082 | 1.20675 | 0.04131 | 0.06072 | 1.249 |
| blind-n256-naive | 1.30539 | 0.05182 | 1.21413 | 0.05532 | 0.08041 | 1.157 |
| blind-n32-ce | 1.29282 | 0.04440 | 1.19440 | 0.04457 | 0.06841 | 0.996 |
| blind-n32-l2 | 1.29768 | 0.05421 | 1.20044 | 0.05786 | 0.08080 | 1.009 |
| blind-n32-multiset16 | 1.30294 | 0.06471 | 1.20735 | 0.06725 | 0.08601 | 1.311 |
| blind-n32-multiset64 | 1.30318 | 0.06274 | 1.20814 | 0.06734 | 0.08703 | 1.202 |
| blind-n32-naive | 1.30715 | 0.07072 | 1.21110 | 0.07482 | 0.10067 | 1.107 |
| positive-n256-ce | 1.55126 | 0.02244 | 1.57073 | 0.02472 | 0.02678 | 1.069 |
| positive-n256-l2 | 1.55322 | 0.02773 | 1.57358 | 0.03175 | 0.03324 | 1.003 |
| positive-n256-multiset16 | 1.56509 | 0.05427 | 1.58678 | 0.05745 | 0.05978 | 1.109 |
| positive-n256-multiset64 | 1.55826 | 0.04355 | 1.58095 | 0.04855 | 0.05035 | 1.197 |
| positive-n256-naive | 1.56890 | 0.06080 | 1.59170 | 0.06634 | 0.07324 | 1.102 |
| positive-n32-ce | 1.55552 | 0.04870 | 1.60337 | 0.05059 | 0.05336 | 1.175 |
| positive-n32-l2 | 1.56347 | 0.06223 | 1.61056 | 0.06127 | 0.06784 | 1.174 |
| positive-n32-multiset16 | 1.57235 | 0.07194 | 1.62774 | 0.07808 | 0.08470 | 1.320 |
| positive-n32-multiset64 | 1.56796 | 0.06919 | 1.61918 | 0.07357 | 0.08194 | 1.240 |
| positive-n32-naive | 1.57884 | 0.07845 | 1.63527 | 0.08342 | 0.09448 | 1.111 |

Preferred condition is exact CE for every data contract here; exact L2 is the
objective-matched reference and beats corrected sampled multiset objectives.
K64 usually helps relative to K16 but not at blind/n32; naive MSE is worse than
corrected K16 in all means. Three seeds, no significance claim. Uniform test
latent TV0.83159, observed TV0.82462 positive/0.76559 blind. Blind invisible mass
is0.73125; low hidden-mass error under this teacher does not prove identifiability.
The exact counterexample preserves identical observations despite different mass.

No Modal/GPU: one-thread local macOSarm64 CPU; family72.504s, peakRSS436,043,776bytes.
All methods score1,612,800 span logits/access count cells; sampled variants add
1,474,560 or5,898,240 draws. No expensive-oracle or large-support speedup is tested.
Eight pre-fit tests, independent924metric sets/1,152observation rows/576teacher rows,
all60checkpoint replays, and368archive members pass. All split predictions/counts,
checkpoints, invocation/env, source snapshots, logs, plot, full RMSE/correlation/KL/
entropy/context metrics are retained. These synthetic outcomes do not resolve
real-data IC50 quality or qualify a biological concentration model; continue
source-qualified biological admission and held-out evaluation under the active goal.

## 2026-10-22_1410 — PXD057588 source inventory and shared-curation acceptance cases

- **Date / agent:** October22,2026; Codex / GPT-6.
- **Directory:** [2026-10-22_1410_codex_availability-source-audit](2026-10-22_1410_codex_availability-source-audit/README.md).
- **Sources / curation:** publisher Tables7–12 and community SDRF for mouse
  PXD057588, with project/file metadata; PXD009092 discovery metadata only.
  Seventeen original assets,11,532,567bytes, frozen URL/size/SHA-256 receipts and
  six publisher-MD5 checks; immutable datacache inputs under `data/availability_sources/`.
  Preserve native labels/scales, table blocks, scope, complete-case filtering and
  reported q≤.05 policy. No missing-to-zero conversion, no admitted training rows.
- **Training contract:** no model/pretraining/training/loss/head mapping/synthetic
  data; no train/validation/test split or prediction dumps because this is a source
  inventory, not predictive evaluation. No held-out loss/regression/binding metrics.
- **Reproduction:** source snapshot and exact invocation in the directory's
  `reproduce/`; base932ccef, dirty audit additions. Python3.12.6, datacache1.29.2,
  openpyxl3.1.5; local macOSarm64 CPU, no Modal or requested/observed GPU. Recorded
  inspection runtime excludes acquisition; not an optimizer benchmark.

| Source block | Peptide rows | Channels | Distinct parent labels |
|---|---:|---:|---:|
| Table7 pro-inflammatory macrophage MHC-I |2,824|6|2,139|
| Table8 anti-inflammatory macrophage MHC-I |3,554|6|2,562|
| Table9 macrophage MHC-I, ordinary tumor coculture |1,309|9|1,096|
| Table10 macrophage MHC-I, MHC-I-loss tumor partners |2,613|11|1,997|
| Table11 macrophage proteomics digest peptides |25,110|9|4,495|
| Table12 tumor-cell MHC-I |362|12|348|

All309,126quantities are positive numeric values in explicitly complete-case tables.
Parent counts are literal labels, not validated protein identities. Eight of nine
Table9/11channel assignments differ; matching TMT labels would mix conditions.
663parent labels overlap,433Table9labels lack an exactTable11match; no specimen or
precursor-normalization join is established. SDRF72rows×45columns covers6/50rawfiles,
not the macrophage/proteomics arms. Repeated headings, independently sorted panels,
Table8's M2-description conflict and Table10's n=3-versus-four-labeled-channel
conflict require preserved source assertions. Table10's loss is in the tumor
partner, not the measured macrophage. Independent OOXML/TSV verification of17hashes,
six inventories,309,126quantity cells and3,240SDRFcells passes.

**Decision:** use these actual cases to qualify existing Hitlist materials/protocols
and cross-modality work, then freeze biological eligibility. No preferred model is
selected here. The sources support observed presentation/digest signals, not the
full intracellular pool. Wolf-Levy SEARCH.7z remained incomplete after two timed-out
downloads and was excluded. Compared with the preceding129synthetic fits, this is
progress in biological-source definition, not biological model validation. Shared
metadata implementation, source admission and the availability goal remain open.

## 2026-10-22_1601 — PXD057588 quantity reconciliation

- **Date / agent:** October 22, 2026; Codex / GPT-6.
- **Directory:** [2026-10-22_1601_codex_pxd057588-quantity-reconciliation](2026-10-22_1601_codex_pxd057588-quantity-reconciliation/README.md).
- **Data / curation:** exact six source workbooks in the preceding immutable
  17-asset bundle; 10,662 MHC-I and 25,110 digest-peptide rows, 32 contrasts and
  200,724 primary fold-change cells. Preserve complete-case scope, source q≤0.05
  policy and quantities; no inferred zeros, parent mapping, aggregation or
  specimen joins. Source-policy contract and every source-cell comparison frozen.
- **Training / evaluation:** no model, pretraining, training, loss, head mapping,
  synthetic data or train/validation/test split: this is complete source arithmetic
  reconciliation, not predictive evaluation. Source-cell dumps replace prediction
  dumps; held-out IC50/500 nM metrics do not apply. No Modal/GPU.
- **Reproduction / runtime:** frozen launch and verifier scripts, input hashes and
  environment in `reproduce/`; dirty Presto51b895e; local macOS arm64, Python3.12.6,
  NumPy2.5.4, openpyxl3.1.5; calculation6.432s and final scalar verification5.067s.

| Policy comparison | Cells | Mismatches at 1e-8 log2 | Error |
|---|---:|---:|---|
| Published assay-specific calculation | 200,724 | 0 | Maximum absolute error1.63e-14 |
| Opposite normalization policy | 200,724 | 200,724 | Per-table RMSE0.0826–0.6782 log2 |

**Decision:** all published transformations reconcile; no model winner is selected.
Independent scalar replay verifies every cell, source coordinates, parent support
and aggregate errors. Tables9/11 share663 literal parent labels; only110 have
multiple observed ligands and multiple digest peptides (258ligand rows). Preserve
normalization and control membership in shared Hitlist evidence. Compared with the
prior inventory, target arithmetic is now qualified; biological fitting, exact
material/reference joins, normalized source admission and concentration claims are
still unresolved. This does not address the earlier IC50 predictive failure.

### Source-audit recovery addendum, October 22, 2026, 16:22 UTC

The previously partial PXD009092 `SEARCH.7z` completed on the third resumed
request. All51,736,986bytes match the PRIDE size and checksum (SHA-1); derived
SHA-256 is `ff1c9a2adcef2f1d4f585254c7f8ee0cfdd373dc545439fc3667985d60011237`.
All13archive entries decode successfully with CRC checks and no filesystem
extraction. Original archive is now immutable in datacache; separate recovery
manifest/script/receipt under the1410source-audit directory preserve the original
17-asset contract. Verification4.754s. This qualifies transfer integrity only;
processed-table semantics, study/sample curation and model admission remain open.

## 2026-10-22_1630 — MAPP source and processed-evidence admission

- **Date / agent:** October 22, 2026; Codex / GPT-6.
- **Directory:** [2026-10-22_1630_codex_mapp-source-admission](2026-10-22_1630_codex_mapp-source-admission/README.md).
- **Exact inputs:** 16 immutable originals, 53,293,496 bytes: seven PRIDE project/file
  inventories, recovered PXD009092 SEARCH.7z and primary accepted manuscript;
  five archive-member hashes. All source rows/flags/scales retained. No missing-zero
  conversion or parent abundance inferred from fragment-derived protein groups.
- **Contract:** 18 MAPP acquisitions; 10,673 unique peptide rows, 149,959 evidence
  rows; 2,003 indexable protein groups plus 36 quarantined source records. Preserve
  run versus aggregate rows, reverse/contaminants, direct versus transferred IDs,
  source FDR-level distinctions, bead-filtered quantities and reference uncertainty.
- **Training / evaluation:** no model, pretraining, training, loss/head mapping,
  synthetic data or train/validation/test split. No predictive metrics or dumps:
  every source quantity/support comparison is preserved instead. No Modal/GPU.
- **Reproduction / runtime:** frozen scripts/manifests/launch environment at dirty
  Presto30cc536; local Python3.12.6/macOS arm64. Initial invalid-ID stop is preserved;
  revised row-quarantine audit17.170s; independent pandas verification3.008s.

| Source comparison | Count |
|---|---:|
| Reported positive and evidence sum agree within declared rounding tolerance | 95,336 |
| Both zero | 55,483 |
| Reported zero despite positive underlying evidence | 9,276 |
| Other quantity discrepancies | 0 |
| Structural UT/cytokine support | 2,272 peptides / 572 groups |
| Multiple-peptide support within those groups | 1,971 peptides / 271 groups |
| Training-admitted rows | 0 |

**Decision:** keep PXD009092 as the concrete cell-line contrast candidate, but resolve
filtering/confidence/source context before fitting. An independent parser verifies
all160,095quantity comparisons and10,673support rows against original evidence.
The paper's4,983enriched-peptide result is not yet reconstructed. The archive's
protein-group LFQ is not independent precursor measurement. Shared Hitlist metadata
and actual whole-proteome pairing remain prerequisites; candidate PXD009360 archive
retrieval is recorded separately. Compared with the prior transfer audit, actual
biological support is now quantified. No concentration or APC generalization is
claimed, and the availability goal remains active.

Later source continuation: PXD009360's733,580,954-byte tryptic archive completed;
deposited checksum/gzip integrity pass, immutable SHA-256
`45b1f2b278937b43a40913798b96c7fb31a0d1a7753ebcf99951287c0ebe91c7`. Verification8.349s;
header audit6.938s. Nine actual Trypsin/P runs have bare experiment
IDs448–456, without a resolved treatment/specimen map. Candidate quantities remain
unqualified; do not infer a precursor join from numeric run order. Separate
receipt/commands preserve this later download without changing the16-input audit.

## 2026-10-22_1736 — MAPP publisher/deposit quantity reconciliation

- **Date / agent:** October 22, 2026; Codex / GPT-6.
- **Directory:** [2026-10-22_1736_codex_mapp-publisher-reconciliation](2026-10-22_1736_codex_mapp-publisher-reconciliation/README.md).
- **Source contract:** three hash-pinned original publisher assets, plus five
  original PXD009092 member tables from the prior source audit. Original cells,
  source IDs, condition columns, alternatives, missingness and quantities retained.
  All 6,311 publisher sequences match the deposit. No invented specimen mapping.
- **Training / eval:** no model/pretraining/loss/head mapping, synthetic data,
  GPU/Modal or predictive split. This is source reconciliation, with complete
  per-cell and mapping evidence instead of predictive metrics/dumps.
- **Reproduction / runtime:** dirty e80fd42, frozen scripts/manifest/commands;
  local Python 3.12.6, pandas 3.0.6, NumPy 2.5.4, openpyxl 3.1.5. Audit 57.791s;
  independent scalar replay 5.958s. Membership 0.416s; independent check 0.276s.
  Interrupted verifier with repeated NPZ decompression is preserved; cached
  immutable arrays resolve that I/O problem without changing numerical policy.

| Comparison | Verified result |
|---|---|
| Run intensity versus 2× median bead control |160,095/160,095 deposited cells|
| Other eight prespecified filter rules |2,357–22,851 mismatching cells|
| Publisher UT / TNF+IFN log medians |4,983/4,983 and 5,311/5,311 matches|
| Publisher membership without imposing unique group IDs |Both sets match exactly|
| Imputed E / F versus their named conditions |Only 2/6,311 matches each|
| Imputed E / F versus unexpected CHX / UT conditions |4,868 / 4,913 matches|
| Publisher UT / TNF+IFN rows with multiple group IDs |559 / 604|

**Decision:** the exact background reconstruction and publisher medians now resolve
an arithmetic admission gap. Keep the twofold criterion's stated provenance
separate from the empirically reconstructed run/median algorithm. Filtered zeros
are not biological absence. Imputed E/F columns have unresolved condition-label
conflicts and cannot provide measured targets. Preserve leading-razor assignments
and alternative parent mappings separately; do not identify unique origin from an
assignment. The selected 131-gene Data3 panel supplies no precursor acquisition map.
Original reference, confidence-setting discrepancy, parent mapping, shared Hitlist
import and endpoint-specific eligibility remain open. New inputs belong in
[Hitlist #746](https://github.com/pirl-unc/hitlist/issues/746), not a parallel registry.

Independent checks cover eight input hashes, 160,095 quantity cells, 1,440,855 rule
decisions, 229,160 publisher-cell comparisons, 10,294 membership rows with six
mapping fields, and 524 Data3 quantity cells. No biological fit or concentration
claim is made; the full availability goal stays active.

## 2026-10-22_1745 — Local CPU / Apple GPU IC50 training feasibility

- **Date / agent:** October22,2026; Codex / GPT-6.
- **Directory:** [2026-10-22_1745_codex_local-ic50-feasibility](2026-10-22_1745_codex_local-ic50-feasibility/README.md).
- **Exact data:** hash-pinned admitted manifest914e0930…; all21,636exact purified
  human HLA IC50 train observations;2,925validation/2,803test including bounds.
  Original peptide-disjoint historical splits; shared studies, unresolved specimen
  identity and prior held-out inspection. No clipping/averaging/generated labels.
- **Training:** unchanged455,041-parameter modular shared sequence encoder,
  explicit classII registers/flanks and pooled direct IC50 output. Supplied
  human species/class. Fresh seed42, AdamW3e-4/wd1e-4, batch64/dropout0.1, one
  complete epoch339updates. No auxiliary/pretraining/APC/MS/TCR modules; no synthetic
  data. Exact train-only standardized log10(nM) squared error weight1; one-sided
  squared error for held-out bounds. Other assay families excluded.
- **Hardware / reproduction:** local M1 Pro10CPU/16GPU cores,32GiB; no Modal.
  Frozen launch/source/environment receipts at dirtyef4b4c1. Four CPU threads,
  inter-op1, no MPS CPU fallback. Canonical trainer/checkpoints/evaluation path.
  CPU median0.4274s/update versusMPS0.4866s from8measured updates after4warmup,
  matching batch identities. Preliminary short timing comparison, not a replicated
  hardware ranking. CPU selected by timing alone.
- **Runtime / verification:** full fit and evaluation169.97s, full wrapper183.76s,
  CPU peakRSS2.59GiB. All32allocated tensors have gradient-driven updates and
  feature coverage passes. Independent82,092-record metric/source replay,
  ridge refit and fresh-checkpoint training replay pass (zero replay difference).
  Per-example predictions for all methods/splits, runtime/checkpoint hashes and
  coverage preserved.35training/resume tests pass. Constant-baseline Pearson
  reporting bug fixed with failing-before/passing-after regression evidence;
  seven original numerical artifacts explicitly corrected to undefined/null.

| Condition | Validation RMSE | Validation Spearman | Test RMSE | Test Spearman | Test500nM AUROC |
|---|---:|---:|---:|---:|---:|
| Train mean |1.3606|undefined|1.3420|undefined|0.5000|
| Fixed composition/class ridge |1.2235|0.4581|1.2291|0.4306|0.7067|
| One-epoch neural fit |1.1431|0.5521|1.1613|0.5117|0.7371|

**Decision:** local training is viable. Neural exact-value RMSE/ranking improve on
this fixed ridge, and test predictionSD0.6949 (target1.3419) is not constant.
However test binderF1 is0.6184 versusridge0.6199, classII recall only0.0882,
and none of203test bounds is satisfied. One seed/epoch is not convergence,
replication or source-independent qualification. Comparison to the historical
9,053-row failure is not controlled. Availability/concentration and source-admission
work remain open. Continue locally; CPU is the measured choice for this current
small audited loop, with GPU batching/synchronization improvements untested.

## 2026-10-22_1845 — MAPP parent-reference and exact-span audit

- **Date / agent:** October22,2026; Codex / GPT-6.
- **Directory:** [2026-10-22_1845_codex_mapp-parent-reference](2026-10-22_1845_codex_mapp-parent-reference/README.md).
- **Exact data:** all10,673 original PXD009092 peptide records, five pinned table
  members, frozen publisher/support ledgers; unchanged2,272 structurally supported
  untreated/TNF+IFN contrast peptides. No new training admission.
- **Reference conditions:** local2023 reviewed-human canonical DAT20,422entries,
  SHA256defd9fd7… with recorded secondary aliases; official UniProt2026_03 reviewed
  human isoform-inclusive FASTA42,562entries, SHA2568c4459cc… with exact IDs only.
  Different releases, scopes and alias policies; not a controlled isoform ablation.
  The original2016 search FASTA remains unavailable. No suffix stripping, I/L
  conflation, approximate matching or source-coordinate repair.
- **Contract:** exact peptide/position alternatives, original parent/group/razor
  fields, source flags, leading-coordinate/flank comparisons, reference inventory
  and full matched parent sequences. Uniqueness is reference-scoped. Every missing
  accession and sequence-negative resolved candidate remains represented.
- **Training/evaluation:** none. Relative captured MAPP signal is not fitted here
  and is not intracellular concentration. No pretraining, loss terms, assay-output
  mapping, synthetic training data, GPU/Modal, predictive split or prediction dumps.
  Synthetic strings test scanner software only. Flags and prior selection remain
  unchanged; missing/filtered signal is not relabeled as a biological zero.
- **Runtime / reproduction:** producer11.537s; independent verifier17.734s, local
  CPU. Dirty4d938e0, frozen code/commands/input receipts, Biopython1.88. Complete
  independent Aho-Corasick scan matches37,721 positions; source replay verifies
  64,538accession rows,26,978positive and14negative resolved candidates. All62,984
  reference records,10,673source records,14subset summaries and graph components
  checked. No predictive validation/test metrics apply.

| Contrast-subset result | 2023 canonical | 2026_03 with isoforms |
|---|---:|---:|
| Exact-matched peptides /2,272 |2,262|2,269|
| Source-leading coordinate and flank agreement |1,678|1,676|
| One distinct possible full parent sequence |2,030|1,013|
| Multiple distinct possible parent sequences |232|1,256|
| Peptides with additional non-source-listed matches |18|1,114|
| Exact position alternatives |2,613|4,529|
| All-source components intersecting subset |540|542|

**Decision:** use the isoform-inclusive generation as the broader prospective
reference, preserving ambiguity; do not duplicate one peptide observation into
independent alternative-parent labels. The542components (largest175subset
peptides) constrain future splitting but are not homology clusters or proof of
independent specimens. Three contrast peptides remain unmatched;26have resolved
leading-coordinate conflicts. Shared Hitlist ingestion and endpoint-specific
admission remain open. Merged PR749 now supplies the common external-source mechanism
and PXD057588; MAPP is still a contract candidate. Full biological availability,
APC context and concentration objective remains active.

## 2026-10-22_1914 — Shared PXD057588 mouse sequence and precursor support

- **Date/agent:** October 22, 2026; Codex / GPT-6.
- **Directory:** [2026-10-22_1914_codex_pxd057588-shared-sequence-admission](2026-10-22_1914_codex_pxd057588-shared-sequence-admission/README.md).
- **Exact data:** shared Hitlist 1.82.22 generation daff7930…, all 87 files verified
  natively and independently, then rechecked after mapping. All 35,772 primary
  PXD057588 rows (10,662 MHC-bound; 25,110 digest), 30,587 distinct sequences.
  Original complete-case selection/quantities retained. No measurement duplicated
  across parent alternatives; no inferred specimen pairing or biological zeros.
- **Reference:** official reviewed mouse release2026_03 including isoforms,
  25,758 entries; SHA25641bc0fa6…. Literal overlapping spans only; no I/L or
  alias/suffix transformations. Separate model and original search references.
- **Training/eval:** none; no model/pretraining/loss/synthetic training/GPU/Modal,
  predictive split or prediction dumps. Native quantities remain observed relative
  MHC/digest signals; no intracellular concentration label or validation claim.
- **Runtime/reproduction:** native plus full package hash check10.899s, mapping
  7.591s, independent replay8.975s, local CPU. Frozen code/commands/inputs at dirty
  6cd7edc. Independent Biopython/complete character scan reproduces52,277positions,
  all35,772source mappings,71,544original worksheet cells,26,449support records,
  all table summaries;36copied input and10output hashes pass.

| Table9/11 parent-support definition | Shared keys | ≥2 ligand and ≥2 digest keys | Distinct ligand/digest rows in multi-peptide support |
|---|---:|---:|---:|
| Literal parent label |663|110|258 /1,919|
| Verified source ID plus peptide occurrence |661|110|258 /1,919|
| All exact reference identities |1,136|236|277 /2,128|
| All distinct full-sequence alternatives |1,134|236|277 /2,128|

**Decision:** 35,771rows match the mouse reference;35,747match their reported parent.
Preserve24missing source IDs, one resolved-ID/absent-peptide case, and11,725rows
with multiple full parent sequences. Prefer verified occurrence support over
literal labels; broader alternative support adds uncertainty, not independent
measurements. The5,374all-table components (largest862sequences) constrain future
splits but do not establish homology/specimen independence. Use the shared recipe
path for a prospective observable-signal comparison; exact precursor pairing and
protein aggregation remain unresolved. MAPP752 and the broader availability goal
remain open. This is shared-source admission evidence, not another registry.

## 2026-10-22_1938 — Full-protein exact versus lazy span computation

- **Date/agent:** October 22, 2026; Codex / GPT-6.
- **Directory:** [2026-10-22_1938_codex_variable-span-computation](2026-10-22_1938_codex_variable-span-computation/README.md).
- **Inputs:** eight full proteins, four predeclared length bands per human/mouse
  verified reference source, selected by sequence hash without assay labels.
  Span lengths 1–40 and 1–53; all valid positions, no MHC/nonbinder filtering.
- **Model:** new variable-length policy, fixed one-hot/prefix composition and
  boundary/length/position features; two hidden32 tanh heads, 5,536parameters in
  six tensors. Synthetic numeric context[0.25,-0.5], not biological metadata.
  Start prior uses valid-length counts; no structurally cancelled output bias.
- **Objective/training:** fixed seed42 weights; zero optimizer updates. Synthetic
  span energy only. Exact expectation sum q*(logq+energy); fresh REINFORCE with
  leave-one-out baseline, 8/64draws. No clipping, normalization, sample reuse,
  reference KL, pretraining or biological training labels. No predictive split,
  held-out assay metrics or predictive dumps apply; full mathematical references
  and sample traces are preserved.
- **Conditions/runtime:** exact, dense8/lazy8, dense64/lazy64 across16cases;
  4warmups+12measured calls, seeded interleaving, CPU4threads/interop1/float32.
  1,280calls; no Modal/GPU. Whole benchmark13.45s, family peakRSS0.719GiB;
  independent replay1.141s. Exact source/config/model/checkpoint hashes and
  executed recipes frozen at dirty24f1d2b.

| Human2,240residues, lengths1–40 | Median forward/backward | Conditional-network rows | Cost queries |
|---|---:|---:|---:|
| Exact expected cost |31.371ms|89,600|88,820|
| Dense REINFORCE8 |17.296ms|89,600|8|
| Lazy REINFORCE8 |2.374ms|320|8|
| Dense REINFORCE64 |17.309ms|89,600|64|
| Lazy REINFORCE64 |3.364ms|2,513.3mean|64|

All conditions also score2,240starts. Actual hook counts verify network work;
query counts alone do not establish savings. Across16cases, dense/lazy runtime
ratios are2.06–8.79×(8draws),1.33–6.91×(64draws). All512paired sample traces agree;
max selected-gradient difference1.77e-8. Independent NumPy MLP/normalization and
analytic backprop reproduce678,190span probabilities and all16exact gradients,
source selection and raw timing/work summaries. Fourteen unit tests, Ruff and
strict docs pass. Scope is this local fixed-weight workload, not convergence or
a hardware ranking.

Mean-gradient probes,128fresh estimates each: human relative L2error0.341/0.106
for8/64draws; mouse0.279/0.104. Corresponding64-draw cosines0.9946/0.9949. Sampling
still introduces estimator noise even though dense/lazy selected gradients agree.

**Decision:** retain exact enumeration as reference and use lazy conditional
scoring for sampled comparisons. It reduces actual network work without changing
the policy/objective. The O(L) start calculation remains, and chunked enumeration
does not automatically bound autograd storage. The primitive is not yet wired
into the canonical mixed-assay trainer or biologically fitted; precursor yield,
clearance, MS response, APC context and calibrated concentration remain separate
requirements. Shared recipe739/MAPP752 and the broader availability goal stay open.

## 2026-10-22_2212 — Shared MAPP source ingestion

- **Date/agent:** October22,2026,22:12America/New_York; Codex / GPT-6.
- **Directory:** [shared-mapp-import](2026-10-22_2212_codex_shared-mapp-import/README.md).
- **Source:** eleven pinned PXD009092 original tables/publisher workbooks and
  PRIDE project/file snapshots; shared Hitlist source adapter with existing
  PXD057588 and verified supporting predecessor. Proteasome-associated HEK293
  fragments, not free concentration, flux or MHC-selected ligands.
- **Curation:** preserve every raw field/row, flags, quarantined group IDs,
  actual six conditions, unresolved specimens, parent claims, confidence conflict,
  search6–40scope and publisher imputation conflict. No negative labels from
  missing/filtered zeros; no inferred precursor denominator or gene knockouts.
- **Training contract:** no model, pretraining, optimizer, synthetic biological
  labels, assay heads, loss or predictive split; held-out predictive metrics do
  not apply. This is a source-integration experiment, not a fit.
- **Runtime:** producer43.29s, independent replay9.46s, local CPU; no Modal/GPU.
  Frozen dirtyHitlistca0c729/Presto80d5f9c, exact commands and source snapshots.
  First completed package is development1.82.24; pending source PR reserves1.82.25.

| Verified item | Count |
|---|---:|
| Original TSV records |162,767|
| Peptides / evidence / source runs |10,673 /149,959 /18|
| Quantity representations |362,882|
| Derivation edges |540,034|
| Workbook XML cells |36,207|
| Missing evidence PEP NaN values |74,573|
| Filtered zeros with positive evidence |9,276|
| Reproduced publisher numeric UT /TNF+IFN |4,983 /5,311|

Independent original TSV/XML and relation replay verifies111package files and
all listed values/edges. All160,095filtered values reproduce within frozen
relative1e-4/absolute1rounding tolerance; maximumrelative8.16e-5. Two failed
attempts are preserved (launcher path type, explicit source PEPNaN), followed by
successful source generation.16focused tests and format/lint pass; full normal
suite passes3,224tests in356.83s (10warnings). Missing parallel-proteome run metadata
and independent precursor estimates are tracked in Hitlist757. Reference join/query/deployment and biological fitting
remain open in Hitlist752/739. No model winner applies; decision is to carry the
verified shared importer forward in [Hitlist draftPR758](https://github.com/pirl-unc/hitlist/pull/758), commitd012527, retaining uncertainty and the full goal.

## 2026-10-22_2251 — Shared MAPP reference alternatives

- **Date/agent:** October22,2026,22:51America/New_York; Codex / GPT-6.
- **Directory:** [shared-mapp-references](2026-10-22_2251_codex_shared-mapp-references/README.md).
- **Data/curation:** thirteen pinned inputs: eleven original MAPP assets plus
  historical reviewed-human canonical DAT and modern2026_03isoform-inclusive
  FASTA. Original2016search reference unavailable; later references explicitly
  distinguished. Shared adapter schema2 preserves exact literal sequences,
  reference-local aliases, all occurrences, source claims, negative candidates,
  coordinates/flanks and coverage. Quantity observations remain unduplicated.
- **Assay/label contract:** captured proteasome-associated HEK293 fragments,
  search6–40, original contexts/confidence/missingness/filtering preserved. No
  affinity filter or biological negatives from missing/filtered quantities.
  No precursor denominator, origin assignment or free concentration claim.
- **Training:** no model, pretraining, optimizer, synthetic biological labels,
  loss terms, assay heads or predictive split. Held-out regression/binding
  metrics and prediction dumps do not apply to this source-conservation audit.
- **Runtime/reproduction:** producer54.80s; independent replay20.22s,
  peakRSS0.987GiB; localCPU, no Modal/GPU. Frozen code/commands/input receipts and
  dirtyGit state: Hitlistfa54fbf, Presto396f771. The completed producer's outer
  timing wrapper exited1 on a sandbox-denied kernel resource query; independent
  verification exits0. Full producer memory profile unavailable, recorded as such.

| Reference condition | Full proteins | Mapped source peptides | All exact positions | Resolved candidates lacking peptide |
|---|---:|---:|---:|---:|
| Historical canonical, recorded secondary aliases |20,422|8,753|14,258|9|
| Modern isoform-inclusive, primary FASTA IDs |42,562|8,790|23,463|5|

Independent BioPython parsing, complete Aho-Corasick scanning and graph traversal
verify all62,984proteinrecords/133,414accessionclaims/37,721positions,
64,538originalsourceclaims,26,992candidate resolutions and21,346coverage rows.
All120packagefiles verify. All eleven original MAPP typed tables and PXD057588
source artifacts are byte-identical to the previous independently audited
source generation. Unmatched singletons are explicit components; exact sharing
establishes neither homology nor specimen independence. Historical alias support
and modern isoform coverage are different contracts, not a model-winner comparison.

**Decision:** use the shared reference relations for the next declared endpoint
recipe; preserve ambiguity without multiplying measurements.21focused tests and
format/lint pass. The initial full-suite preflight refused0.77GiBavailable; a
normal one-worker retry admitted3.89GiB and passed3,236tests/10warnings in320.81s.
Indexed-corpus integration CI remains separate.
Release/query/endpoint admission and biological fitting remain open under
Hitlist752/739; independent precursor/pairing gap is in757. The full availability
goal remains active. This is an external-source generation, not a rebuilt later
T-cell/numeric/query export or a trained concentration model.

Shared reference implementation committed as1ad9aae in Hitlist draft758.
Presto formatting/lint and strict docs pass. Final publication does not close
source/recipe issues or the persistent biological availability objective.

## 2026-10-22_2330 — First biological MAPP exact-versus-sampled fits

- **Date/agent:** Registered October 22, fitted October 23, 2026; Codex / GPT-6.
- **Directory:** [mapp-biological-comparison](2026-10-22_2330_codex_mapp-biological-comparison/README.md).
- **Source/curation:** Verified shared Hitlist MAPP package, manifest `55f7a3d5…`; recipe `mapp_captured_composition_min2_v1`, manifest `df78d02f…`. Human HEK293 PXD009092 UT/TNF+IFN, original nonimputed median of three filtered runs including zeros; canonical20, no reverse/contaminant, one source group/full reference sequence, replicated positive quantities, direct MS/MS, peptide PEP ≤0.01, source-leading coordinates/flanks. Original global-FDR/reference/specimen limitations remain. Duplicate aliases/positions do not multiply labels.
- **Admission/splits:** Original ≥3-peptide recipe failed prospective support gate at 51/7/7 parents; no fit. Pre-fit ≥2-peptide amendment preserves all old partitions/targets and source graph. Admitted 616 peptides / 101 parents, 72/15/14 train/validation/test, 144/30/28 condition distributions. Fixed all-source full-sequence/protein-group components, including excluded bridges; no homology/specimen-independence claim. Both immutable cohorts and every exclusion archived.
- **Label/output:** Within-parent observed captured-intensity composition → full-parent start/length policy, marginalized over each peptide's positions and normalized among admitted candidates. Conditional signal only; peptide-specific MS/capture response and outside-set mass unidentified. No unreported negatives, concentration, precursor denominator, MHC-affinity filter, binding-threshold metric or synthetic biological labels.
- **Training:** Hidden32 two-tanh-MLP policy, support1–40, optional treatment0/1. Three seeds42/43/44, Adam lr0.003/no decay, batch8,40epochs/720updates; no pretraining/dropout. Unit-weight mean reverse KL per parent/condition. Exact and fresh REINFORCE64 share initial weights/order/objective; leave-one-out baseline, no clipping/std/reuse/reference penalty. Separate no-context architecture and uniform controls. Best validation KL selects; test once after selection, no refit.
- **Runtime/reproduction:** Local CPU4/interop1, no Modal/GPU. Nine fits54.98s, peakRSS0.466GiB; training commit8242200 with dirty status and exact overrides frozen in reproduce/training.json. Raw artifacts, all-split prediction dumps, initial/selected/final states, histories and masks archived in results/training-artifacts.tar.gz. Shared source and recipes pinned, frozen commands/sources in reproduce/.

| Method | Validation KL | Test KL | Test TV | Test log10 RMSE | Test Spearman | Condition-change RMSE | Seconds/fit |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact-context | 0.1668 ± 0.0076 | 0.3039 ± 0.0078 | 0.2759 ± 0.0059 | 0.3341 ± 0.0062 | 0.0090 ± 0.0412 | 0.1565 ± 0.0001 | 6.09 |
| reinforce64-context | 0.1673 ± 0.0082 | 0.3119 ± 0.0034 | 0.2804 ± 0.0033 | 0.3394 ± 0.0028 | -0.0011 ± 0.0395 | 0.1565 ± 0.0001 | 6.18 |
| exact-no-context | 0.1645 ± 0.0037 | 0.3079 ± 0.0082 | 0.2783 ± 0.0054 | 0.3362 ± 0.0052 | -0.0182 ± 0.0106 | 0.1566 ± 0.0000 | 6.00 |
| Uniform | 0.2031 | 0.2890 | 0.2705 | 0.3213 | undefined | 0.1566 | unfitted |

Means ± population SD across seeds, not biological confidence intervals. Validation15/test14 parents. Full held-out Pearson/Spearman/RMSE, forward/reverse KL, TV, spread, condition contrasts and two-versus-larger-set strata are in the linked report/JSON/CSV. Uniform correlations are undefined; raw near-zero centering artifacts are preserved but not interpreted as ranks.

**Winner/decision:** uniform remains the held-out reference. Every fitted model improves validation initially but fails test reverse KL and RMSE; selected epochs2–3, later validation worsens. Context response is nearly flat (predicted changeSD≈0.001 versus target0.176). Nonconstant composition and all six tensors updating do not establish useful biology:704parameter elements have no nonzero gradients, and the conditional objective leaves total selected mass unidentifiable. No conclusion about earlier IC50-collapse causes.

Independent original-source replay verifies all10,673dispositions/1,232contributors and old/new partition conservation. Independent NumPy forward/metric/checkpoint/work replay checks11,088prediction values, maximum logq error1.35e-6. Exact score-function expectation error5.55e-17; position-marginalization gradient error8.33e-17. Each fit scores36,720selected candidates; REINFORCE draws368,640terms and saves no network scores here. This is different from the earlier synthetic full-oracle and lazy full-protein compute experiments.

Shared recipe [Hitlist760](https://github.com/pirl-unc/hitlist/pull/760) passes3,243normal tests/10warnings in327.51s,14focused tests,format/lint and strict docs; hosted recipe CI remains separate. [Hitlist759](https://github.com/pirl-unc/hitlist/issues/759) requests controlled proteasome-inhibition/reporter panels; precursor/specimen mapping remains in757. Next: training-only capacity/interaction and replicate-variation diagnostics, followed by untouched biological evaluation. Existing test is now observed. This completed negative family advances, but does not complete, the full availability/concentration goal.

Posthoc training-only saved-checkpoint replay (registered before execution, no fit/reselection/additional test checkpoints): initial trainKL0.329–0.338 falls to final0.0227–0.0304, training Spearman0.729–0.784. This policy can fit the training objective while validation deteriorates; generalization failure is established for this family, but its biological/statistical cause and historical IC50 behavior remain unresolved. All diagnostic predictions and exact source/invocation are retained.

## 2026-10-23_0037 — MAPP repeatability and parsimonious sequence scores

- **Date/agent:** October 23, 2026, 00:37 America/New_York; Codex / GPT-6.
- **Directory:** [mapp-run-repeatability](2026-10-23_0037_codex_mapp-run-repeatability/README.md).
- **Data/curation:** shared PXD009092 human HEK293 UT/TNF+IFN MAPP generation, manifest `55f7a3d…`. Original min2 recipe unchanged (`df78d02f…`): canonical20, no reverse/contaminant, one full reference sequence/source group, replicated positive quantities/direct evidence, PEP≤0.01, source-leading coordinates/flanks, ≥2 peptides per parent. Fixed all-source sequence/group components, no homology/specimen-independence claim. No MHC affinity filter or synthetic biological labels.
- **Stage A:** new shared `mapp_complete_run_composition_v1` preserves all 3,696 run records and source metadata from 616 base peptides. Complete positive six-run profiles and ≥2 complete peptides leave 369 peptides / 71 parents (55/7/9 inherited strata). Keep 50 background-filtered and 234 no-positive-evidence zeros; no pseudocount/imputation or biological-negative labels. Both old eight-file recipe generations replay byte-identically. Manifest `23bcfeb4…`.
- **Observation endpoint:** centered log10 captured-intensity composition, held-out acquisition predicted from the other runs; condition differences use all nine UT/TNF run combinations without asserting specimen pairing. No model fitting in Stage A. Macro parent MSE; pooled correlations/spread. Previously observed validation/test strata are descriptive/development data, not a new biological test.

| Scope | Parents | Composition RMSE: other runs / uniform | Composition Pearson | Change RMSE: other runs / zero | Change Pearson |
|---|---:|---:|---:|---:|---:|
| all | 71 | 0.1553 / 0.3547 | 0.9136 | 0.2197 / 0.1916 | 0.1719 |
| train | 55 | 0.1461 / 0.3538 | 0.9230 | 0.2066 / 0.1759 | 0.1881 |
| validation | 7 | 0.2030 / 0.3399 | 0.8327 | 0.2871 / 0.2500 | 0.1746 |
| test | 9 | 0.1670 / 0.3716 | 0.9072 | 0.2362 / 0.2283 | 0.1263 |

Baseline composition repeats well (Pearson 0.914, Spearman 0.909); unshrunk condition differences repeat weakly (0.172/0.178) and lose to zero-change RMSE. Macro condition-mean RMS change 0.1236 versus estimated sampling-noise RMS 0.1036; adjusted mean square +0.004554, with negative parent estimates retained. This is not a calibrated biological noise ceiling. Within-parent normalization also removes genuine parent-wide processing-yield changes.

- **Stage B data/training:** the original 616-peptide / 101-parent median cohort, 72/15/14 parents, not the smaller complete-case subset. Captured-intensity composition → observed-set linear softmax score. Length-only (1 parameter), sequence (92 train-supported directions) and explicit treatment×sequence interactions (184). Training-only centering/RMS/QR. Original 105 features are length, composition, termini, adjacent residues/boundary and relative position. Every final/selected parameter changes; omitted contrast directions remain unidentifiable.
- **Fixed comparison:** 15 deterministic exact fits (three representations × five L2 weights 0/0.001/0.01/0.1/1) and 15 interaction REINFORCE64 fits (same weights × seeds42/43/44). Zero initialization, Adam lr0.03, 1,000 full-batch updates, no decay/dropout/pretraining. Unit-weight mean reverse KL plus L2 sum of squared weights; fresh replacement draws/leave-one-out, no clipping/std/reuse/reference penalty. Validate at 0 and every25 updates; choose regularizer/checkpoint by validation only. Evaluate six selected models on development-test; unselected fits intentionally preserve validation dumps only.

| Model | Parameters | L2 / selected update | Validation KL | Development-test KL | Test log10 RMSE | Test Spearman | Change RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| length / exact | 1 | 0 / 25 | 0.1919 | 0.2972 | 0.3258 | -0.0663 | 0.1566 |
| sequence / exact | 92 | 1 / 875 | 0.1694 | 0.2928 | 0.3246 | -0.0225 | 0.1566 |
| interaction / exact | 184 | 1 / 25 | 0.1646 | 0.2908 | 0.3241 | 0.0500 | 0.1712 |
| interaction / reinforce64 / seed42 | 184 | 1 / 500 | 0.1601 | 0.2971 | 0.3290 | 0.0524 | 0.1749 |
| interaction / reinforce64 / seed43 | 184 | 1 / 475 | 0.1594 | 0.2940 | 0.3269 | -0.0042 | 0.1717 |
| interaction / reinforce64 / seed44 | 184 | 1 / 625 | 0.1533 | 0.3095 | 0.3358 | -0.0422 | 0.1768 |
| Uniform | 0 | — | 0.2031 | 0.2890 | 0.3213 | undefined | 0.1566 |

**Winner/decision:** interaction exact is best exact architecture by validation; uniform remains the development-test reference. Every model fails test KL/RMSE. Explicit interactions increase condition-change SD to0.062–0.074 but worsen change RMSE to0.171–0.177 versus zero-change0.1566. No model promotion or further hyperparameter tuning. 144/202 development-test and12/112 validation centered feature rows lie outside the training contrast span, versus zero train rows. All allocated weights training does not establish coverage of absent contrasts.

**Verification/runtime:** independent source replay verifies all3,696 run quantities/metadata and10,673 dispositions; scalar replay checks1,065 fold records/5,535 prediction values (max difference1.78e-15). Independent sequence-feature/QR/NumPy-score/metric/checkpoint replay checks30 fits and10,080 prediction values (max logq error4.14e-7). Analytic gradient error1.25e-16; finite differences7.97e-12; mean128 sampled gradients relative error0.01374. An initial verifier compared differently padded validation batches; corrected checks match each metric to its actual dump and retain the measured ≤2.39e-7 logq discrepancy, without rerunning fits.

Local CPU only, no Modal/GPU: source materialization plus old recipe replay27.18s/2.13GiB; independent source replay3.66s/3.49GiB; analysis0.097s/148MiB; 30 actual fits25.81s/427MiB, 30,000 optimizer updates. Exact/sampled each score918,000 valid candidates per fit; sampled target terms9,216,000 versus918,000 exact, no score saving. These differ from the earlier nonlinear model's batch/thread/training budget and are contextual, not a controlled causal representation comparison.

**Artifacts/reproduction:** stage-specific plans, exact commands/env/dirty Git receipts, source snapshots, all run predictions, all initial/selected/final model states/optimizers/masks/histories and selected train/validation/test dumps in the experiment directory; portable complete-run recipe and linear-artifact archives with hashes. Execution commits Presto `dccafe1`, Hitlist `b94289d`. nM/binder metrics do not apply. Full Pearson/Spearman/loss/RMSE/TV/dispersion and condition contrasts are retained.

Twenty focused Hitlist tests, format/lint/strict docs pass. Normal local full tests were refused twice by the unchanged memory guard before workers launched; no local full-suite pass is claimed. Python3.10–3.14 unit CI passes; retrieved3.11 log:3,247passed/2optional-data-package skips/397warnings in217.11s. Indexed-corpus integration remains separate. Shared recipe stays in Hitlist760/739; precursor/yield metadata in757 and controlled perturbation/reporter data in759. Next: explicitly separate MS-response transfer and processing/yield supervision, broaden source coverage and preserve untouched evaluation. Full concentration/availability goal remains active.

Current-family closure: Hitlist b94289d integration/coverage and all Python unit, lint and docs checks pass. All 186 linear-fitting artifact files and complete-run recipe archive members independently match their original files/hashes. Independent MS-response calibration is tracked in [Hitlist #762](https://github.com/pirl-unc/hitlist/issues/762), separate from precursor #757 and processing-panel #759.

Publication follow-up: report/artifacts published and read back at `6845eb9`. Concurrent Hitlist PR761 merged (`0dfa170`) and closed739. The pinned MAPP experiment remains on its recorded generation; draft source758/recipe760 require integration with the released EvidenceStore/canonical recipe interface before merge. No release verification or migration is implied by this experiment.

## 2026-10-23_0149_codex_mapp-v2-integration — canonical shared source/recipe conservation

- Date:2026-10-23; agent/model:Codex / GPT-6. [Experiment report](2026-10-23_0149_codex_mapp-v2-integration/README.md), [exact reproduction](2026-10-23_0149_codex_mapp-v2-integration/reproduce/launch.sh).
- Contract: append historical verified MAPP PXD009092/reference data to the actual Hitlist2.0 schema7 reference package, preserving all MHC/context/bulk/T-cell/numeric/mouse data. Hitlist execution1442c03; source manifests and environment/dirty state frozen. New schema8 manifest `83a21da963d7d938c68a9570d7b868e578107302ba466ac55df09c839309b693`. Unknown specimens, FDR/search-reference conflicts and capture/response remain explicit.
- No training/pretraining/synthetic training data, optimizer or loss; no new validation/test split or predictive dumps/metrics. Endpoint is exact source/API/target conservation, not biological accuracy. The previous biological development result remains negative. nM/binder metrics do not apply.
- All124 predecessor files (including manifest) preserved with separate inodes;33 MAPP files unchanged.158 total generation files.71 materialized evidence types,18 MAPP relations; independent replay matches all1,829,008 queried native rows and32 previous baselines.
- Three recipe conditions: original min3=544peptides/65parents/130distributions; min2=616/101/202; complete-run=369/71/142. All original target/disposition/contributor/split files byte-identical, including all-run records for the complete recipe.3,058 overlapping-cohort target values independently recomputed; new canonical source/producer envelope recorded separately. Preferred interface: EvidenceStore/materialize/verify-recipe; removed the separate draft recipe exporter.
- Runtime: local CPU only;64.26s source build,21.14s open/catalog/audit,8.88s full query,26.92s recipes;3.02GiB producer peakRSS. Independent replay12.86s/607MiB. No Modal GPU requested/observed.
-55 focused Hitlist checks, format/lint/strict docs pass. Normal full local suite refused by intact memory guard (0.67GiB/2.5required).159 Presto archival/legacy-v4 regression tests pass against actual2.0 after explicit source-module import repair. Integrated-head CI/review/release still required.
- Takeaway: existing source/target contract now works through the current canonical API without dropping newer modalities. This adds no concentration label, calibrated MS response or new machinery contrast. Next owners: Hitlist762/757/759; broader goal open. Portable complete recipe archive, source manifest, all query fingerprints, independent receipts and logs linked from the report.

Integration publication review: source execution remains pinned to1442c03. Published Hitlist6e275fe reconciles merged snapshot PR763 and adds only an import-order correction to executable code; frozen sources remain exact. Final55 focused tests, format/lint and both strict-docs builds pass. Source758 ancestry is verified in the integrated branch. Full integrated-head CI/review/release remain separate gates.

## 2026-10-23_0216_codex_ms-calibration-admission — known-load MS-response source admission

- Date/agent: October23,2026; Codex / GPT-6. [Full report](2026-10-23_0216_codex_ms-calibration-admission/README.md), [shared Hitlist admission contract](2026-10-23_0216_codex_ms-calibration-admission/reproduce/hitlist-admission-contract.md). Presto7020b62 / Hitlist6e275fe; dirty registration state and frozen stage commands/source hashes recorded.
- Data: original PXD004886 Supplementary Data1/2/4/8, public Europe PMC archive with byte-identical publisher Data2/8. Original hashes, labels, cells, formula caches, missingness and failed retrieval/replay attempts retained. No repaired source values. Human HEK293 digest is background context for synthetic standards, not an APC processing label.
- Independent audit:12,941 values/formulas;30 nominal sequences/five groups;31 reporter forms;25 exact threefold design amounts. Raw S4 has2,372 positive responses among3,069 planned slots; normalized has2,398. No numeric zero. Source form multiplicity, missing run/day links and nominal-amount versus molecular-form uncertainty remain explicit.
- Comparisons:26 normalized-only positive cells;39/97 paired columns do not follow a single shared ratio at1e-6. All162 source Avg/SD/CV checks match (relative error≤1.96e-14). Data8 self-label differs from publication label; positional curve-grid/design differences include111 versus100fmol.1,914/3,410 positions differ>5%, but some are missing observations or rounding-sensitive low amounts, and the positional join is not a supplied sample identity.
- Runtime: local CPU, no Modal; initial partial inventory1.17s/85.8MB;59.9MB public archive GET174.37s; final independent replay0.181s. Four complete original files plus full cells/source receipts remain available.
- No training/pretraining/synthetic labels/loss/held-out predictive split or prediction dump: source admission only. No predictive accuracy or binding metrics apply. Fixed decision: no calibrated curve training until source joins/forms/normalization are reconciled; implement source-faithful typed relations and consider bounded raw S4 admission through Hitlist762. This does not identify the cause of MAPP's negative fits or complete the availability/concentration goal.


## 2026-10-23 — Shared MS-calibration source and protocol conservation

- **Agent/model:** Codex / GPT-6; [registered experiment/reproduction](2026-10-23_0316_codex_shared-ms-calibration/README.md). Presto3f1ae1b registration with existing dirty task notes; corrected shared source clean Hitlist04f7369 (2.0.3 candidate, not released). Exact commands/env/source snapshots retained.
- **Dataset/curation:** PXD004886 original Data1/2/4/8/13/14/15 and article XML,8 pinned assets;30 selected synthetic standards,25 nominal fmol-on-column design loads,32 distinct cross-asset reporter labels,6,138 raw/normalized S4 slots,3,410 curve positions and520 protocol parameters. No form-molarity, run/day, curve-sample, selection/censor or APC labels invented. Source title/aggregation/normalization conflicts retained.
- **Training/assays:** source/API conservation only. No pretraining/training, synthetic labels, loss terms/weights, assay heads, new held-out evaluation or predictions. MS response is not processing/yield/concentration; regression/binder metrics are inapplicable. All prior MAPP biological targets and previously exposed splits unchanged.
- **Conditions:** first source import f7ebc08 fails on exact cross-table label equality; retained failed attempt. Corrected04f7369 preserves Data4 SMD2 versus Data8 SMD2.sum4 as different reporters. Full source/API and independent XML/Parquet/recipe audit pass.
- **Results:**13,561 source cells/formulas verified;162 source-summary arithmetic checks;157 prior files byte-identical with independent inodes;84 materialized types;31 full queried relations/1,853,831 rows;3 original MAPP recipes conserved,28 archived files verified.26 normalized-only values and39 nonuniform columns remain unresolved. No predictive improvement claimed.
- **Resources:** local CPU, no Modal/GPU. Build65.72s,open/audit22.04s,queries8.90s,recipes27.77s;producer peak3.22GiB. Independent audit13.66s/0.64GiB. Format/lint,56 focused tests and strict docs pass; normal full-suite memory guard refuses0.65/2.5GiB before workers.
- **Preferred contract/decision:** separate reporter identities with explicit unresolved aggregation equivalence; shared ingestion is ready for review, training admission remains pending. Hitlist762 updated with source-backed acceptance criteria and recovered reference-protocol volume/site/gradient/window context. Next is a bounded canonical raw-S4 recipe and local response diagnostic; concentration goal and earlier MAPP/IC50 failures remain unresolved.


## 2026-10-23 — Raw-S4 conditional MS-response admission and nested peptide CV

- **Agent/model:** Codex / GPT-6; [full registered family and exact reproduction](2026-10-23_0359_codex_raw-s4-response/README.md). Presto50dbfad registration; clean executed Hitlist96b9232, unreleased2.0.3. Local CPU only, no Modal/GPU/environment changes.
- **Dataset/curation:** canonical Hitlist raw-S4 nominal-response recipe from pinned PXD004886 source. All3,069 raw slots retain dispositions;1,974 positives form224 peptide×site summaries for22 sequences. One of23 form-eligible sequences has no positive raw response. Exclude multiple forms, explicit aggregation and isotope mismatches. Normalized rows/curves are outside target scope; missing/censored/absent are not zeros or negatives. Nominal fmol remains a covariate, not certified form concentration.
- **Training/assays:** synthetic standards after background digestion; no processing/MHC/concentration labels, synthetic training examples or pretraining. Five fixed canonical-sequence folds; nested fourfold inner selection of ridge alpha[1e-4,1e-3,.01,.1,1,10,100]. Weighted MSE + alpha coefficient L2, training-only scaling, equal total peptide weight. Median log10 raw peak area maps directly to one regression output. No neural/full Presto or RL training.
- **Held-out results:** nested outer diagnostic CV, no untouched external biological test. Mean RMSE1.1796; nominal-load/site ridge**0.8084** (Pearson.8805,Spearman.9013); added length/AA fractions1.2762 (Pearson−.0277,Spearman−.0814). All inner-validation and outer-test losses/predictions retained. Binding thresholds/IC50 metrics inapplicable. Sequence-minus-context RMSE+.4677,paired22-peptide bootstrap95%[.1533,.7886].
- **Conditions/decision:** mean control,context ridge,sequence-augmented ridge; context preferred for this selected source contract. Larger jointly selected regularization shrinks context too in the sequence arm; failure is not proof of absent sequence signal. Methionine column unexposed and removed. No calibration transfer or concentration claim.
- **Verification/resources:** independent original-XML admission passes; independent augmented least-squares replay verifies all295 fits,12,544 validation and672 test predictions,group isolation/alpha choices/metrics (maxprediction discrepancy7.93e-11). Fitting.742s/106MiB; independent replay.852s/91MiB. Admission18.60s/744MiB; independent source replay.082s/66MiB. Full8-file recipe archive byte-identical. Format/lint61 focused tests and strict docs pass; full local runner refuses0.30/2.5GiB before workers without bypass.
- **Next:** retain context baseline, qualify broader response data under Hitlist762 and investigate controlled-processing/known-substrate panels under759 for the next exact-versus-sampled availability comparison. Biological MAPP/IC50 failures and full concentration goal remain unresolved.


## 2026-10-23 — Controlled MAPP source recovery and kinetic-data request

- **Agent/model:** Codex / GPT-6; [report and reproduction](2026-10-23_0434_codex_mapp-controlled-admission/README.md). Presto9f8e4e0 with recorded unrelated dirty work; clean Hitliste186dd7; canonical source manifest bce500f3….
- **Dataset/curation:** inventoried all seven original MAPP projects; fully retrieved/checksum-verified PXD009508 controls and PXD007263 time-course processed archives. Source counts:6,060/3,146 peptide rows,19,581/27,196 evidence rows,18/12 exact run assignments. Controls cross alpha6/isotype/GAPDH labels with UT/epoxomicin; time-course labels are0/1/2/6 with three numbered replicates. Preserve literal labels, unresolved specimens, source confidence/search/MBR and missingness; no binding filter, inferred missing negatives or precursor/concentration labels.
- **Training/evaluation:** source arithmetic/admission audit only; no model, pretraining, synthetic data, losses/weights, assay-head mapping or requested/observed Modal GPU. No held-out split or validation/test predictions because source qualification precedes fitting. IC50/binder metrics inapplicable; no new model winner.
- **Conditions/results:** literal-join attempt stopped on punctuation differences; explicit unique candidate aliases retain original names. All109,080 controls and37,752 time-course quantity comparisons agree with linked evidence within fixed rounding tolerance or are both zero. Positive agreeing cells15,484/20,940; no source-zero/positive-evidence mismatch. All146,832 comparisons and ten original table files independently replay via BSD tar/pandas grouping; zero sum difference. Reporter/Velcade assets remain unresolved, not proven absent from all raw data.
- **Resources/reproducibility:** failed146-second full transfers preserved;143 verified HTTP ranges recovered both archives in1,290.53s. Inventory3.28s, arithmetic1.35s, independent verification7.63s/397MiB. Full source hashes, stage configurations/frozen scripts, source/member metadata, row-level quantity dumps and first failed attempt retained. Local CPU only, unchanged environment. Both starting commits' hosted CI passes; earlier local full-suite memory refusal remains recorded.
- **Decision/ownership:** prioritize shared Hitlist admission of the recovered inhibitor panel under#759, then repeated-observation signal and frozen biological comparison. Filed [#770](https://github.com/pirl-unc/hitlist/issues/770) for quantitative full-protein digestion time courses (MSV000092631), with actual505-file inventory and source selection/background requirements. ProteasomeDB identification-only fields are not concentration targets. Exact and sampled kinetic baselines are specified prospectively with the same span support; no new fit or completed cellular availability model is claimed.


## 2026-10-23 — Adjudications and shared-source conservation

- **Agent/model:** Codex / GPT-6; [registered reproduction and report](2026-10-23_0551_codex_adjudication-source-conservation/README.md). Clean executed Hitlist `0219827` (unreleased2.0.5), Presto `7d4c7a5` plus recorded dirty registration. Exact commands, thread overrides, source snapshots and original failed verifier retained.
- **Dataset/curation:** pinned reviewed schema8 predecessor plus unchanged PXD009092/PXD004886 original assets. New schema9/catalog12 preserves all126 predecessor files,9 review claims,33 MAPP and23 calibration files:183 files total. No measured row, source scope, confidence/censor/filter policy or assay admission changes.
- **Training/evaluation:** none; no model/pretraining/synthetic data/loss or assay-output mapping change. No new train/validation/test split or predictive dumps: this is source/query/recipe conservation. All four earlier recipe targets/contributors/partitions remain byte-identical, including616-peptide/101-parent MAPP and224-summary/22-peptide rawS4 cohorts. IC50/binder metrics inapplicable.
- **Conditions/results:** actual historical adjudication and source schema8 readers both pass; combined schema9 exposes85 materialized types. Independent hashing/direct Parquet replay checks all183files and1,853,840mixed-grain rows across32queries. Canonical and earlier standalone MAPP payloads both match. Initial verifier selected standalone baselines lackingcoverage.json; amended baseline contract and initial failure retained, no producer/target rerun.
- **Resources:** local CPU, no Modal/requested or observed GPU. Producer148.01s/3.28GiBpeakRSS; independent verifier12.83s/0.64GiB. Format/lint/docs/70focused checks pass; normal local suite3,317passed/10warnings in321.51s with unchanged memoryguard andoneworker. Hosted CI separate.
- **Preferred condition/decision:** combined schema9 preserves current reviewed evidence and both source drafts. Fix pushed to Hitlist767, issue771. No new biological-performance or concentration claim; continue controls admission759 and the registered local biological comparison. Complete manifests, query hashes, recipe-export archive and logs are in the experiment directory.


## 2026-10-23 — Controlled MAPP admission and selected training response signal

- **Agent/model:** Codex / GPT-6; [report and reproduction](2026-10-23_0645_codex_controlled-mapp-signal/README.md). Presto registration a52157c with recorded dirty state; clean executed Hitlist1da88e6, candidate2.0.6/schema9/catalog13. Local CPU only; no Modal/GPU or environment changes.
- **Data/curation:** original PXD009508 and PXD007263, 9,206 peptides/46,777 evidence rows/30 acquisitions/146,832 reported quantities. Shared `mapp_controlled_run_intensity_v1`: canonical20, nonreverse/noncontaminant, peptide PEP<=.01, one source group, reported/replayed positives. All zeros, missing intensities, exclusions, contributors, label aliases, capture/treatment/scoped species/cell context and reference alternatives retained. No affinity/ligand-length filter or biological-negative inference.21,607 positive observations; train/val/test14,862/4,072/2,673, outcome-independent sequence/parent components.
- **Training/assays:** no model/pretraining/synthetic data. Training alpha6 subset requires three positives per arm and direct MS/MS in both arms:59 peptides/40 components, only2.56% of2,301 training peptides with any qualifying signal. Nine leave-one-run-per-arm-out empirical log10 differences versus a common shift; equal-weight squared error. Unresolved specimen pairing is not manufactured. Held-out14/14 peptide eligibility counts only; no held-out response statistics or predictions, intentionally reserved for the frozen next fit. IC50/binder metrics do not apply.

| Training-only endpoint | Common-shift RMSE | Peptide-specific RMSE | MSE improvement | Component-bootstrap95% interval |
|---|---:|---:|---:|---:|
| Captured signal shift |0.90314|0.67608|0.35858|[0.18812,0.55524]|
| Fixed-support composition shift |0.85018|0.60351|0.35858|[0.19023,0.54767]|

- **Decision:** descriptive residual-repeatability gate passes in all9 comparisons; no predictive winner yet. Identical average MSE improvement under normalization is algebraic, not independent confirmation. Bootstrap1000 component draws/seed20261023 is conditional on these acquisitions, not causal biological confidence. Fixed-support total signal epoxo/UT0.71187 versus0.23338 across all qualifying observed training signals emphasizes selection/missingness. Neither is intracellular concentration.
- **Verification/resources:** independent raw-table/hash/scanner replay verifies218files, all183 old files unchanged,17 canonical queries/1,098,915 mixed-grain rows, all quantities/193,609 contributor edges,40,539 exact reference occurrences and every recipe target/disposition/split. Independent scalar recomputation verifies all1,062 diagnostic predictions and bootstrap draws. Admission135.712s/5.031GiB; independent27.962s/2.521GiB; signal0.207s/.261GiB. Full reports, per-peptide observations/predictions, stage source/configuration hashes and raw artifact paths retained. A stdlib-shadowing import failure was preserved and corrected by renaming the entrypoint before reading data; scientific contract unchanged.
- **Checks/next:**78 focused tests, format/lint/strict docs pass; normal full local runner refused at2.10 and2.47GiB versus unchanged2.5GiB guard, not a suite pass. Upstream stack/version reconciliation, review/current CI and release remain distinct. [Next frozen experiment](agents/codex/plans/controlled-mapp-exact-sampled-fit.md): four sequence coefficients plus captured-signal scale, conditional on observed UT baseline, exact versus REINFORCE32/128 and mean/ridge controls. Not launched. Requests remain Hitlist759/757/762/770; full concentration goal stays active.

## 2026-10-23 — Controlled inhibitor response: exact and sampled fits

- **Agent/model:** Codex / GPT-6; [full report and reproduction](2026-10-23_0721_codex_controlled-mapp-fit/README.md). Registered before fitting on Presto80ffa7e with dirty state and executed source hashes retained; clean shared Hitlist0be964d/candidate2.0.6. Local CPU only, no Modal/GPU/environment changes.
- **Data/curation:** PXD009508 human HEK293 alpha6 untreated/epoxomicin, shared `mapp_controlled_complete_alpha6_response_v1`; standard20, nonreverse/noncontaminant, PEP<=.01, one source group, positive reported/replayed quantities, all three positive runs and direct MS/MS in each arm. All run and peptide exclusions, raw quantities, contexts and contributor IDs remain. No MHC/length filter, imputation or negative labels. Same-study sequence/parent components; train59peptides/40components, validation14/5, test14/9; six source values per peptide. Only2.56% of training positive support survives both-arm selection. Exact source/recipe hashes in the report; all eight prior run-export files remain byte-identical.
- **Model/training:** observed untreated mean intensity plus four train-standardized sequence features (log length, AVILMFYW/KR/DE fractions) → conditional softmax composition and one training log-total-ratio scalar → predicted treated captured-MS intensity. Four rank-supported policy coefficients and the scale are all fitted. Exact vs REINFORCE32/128, three sampled seeds20261023/24/25, Adam.03,2,000updates,float64,one CPU thread, final checkpoint, no validation tuning. Matched reverse KL +.1*||w||²/2; fresh replacement sampling/LOO baseline, no clip/std/reuse/reference KL. No pretraining or synthetic biological labels. Separate unchanged/common-shift controls and ridge MSE+.1*sum(slopes²), unpenalized intercept. IC50/binder metrics inapplicable.

| Condition | Val response RMSE | Test response RMSE | Val reverse KL | Test reverse KL |
|---|---:|---:|---:|---:|
| No change |1.16807|.74509|1.33880|.33326|
| Common shift |1.07057|.63536|1.33880|.33326|
| Exact |1.03605|.64789|1.62046|.30000|
| REINFORCE32, three-seed mean |1.03276|.64603|1.63194|.30133|
| REINFORCE128, three-seed mean |1.03141|.64421|1.61507|.29804|
| Ridge |1.09002|.62374|1.26770|.38587|

- **Decision:** no fitted condition passes the frozen requirement to improve both validation and test response RMSE with nonconstant predictions. All sampled seeds fail individually. Common shift remains the response reference. Exact improves test composition KL but worsens response RMSE; ridge has opposite trade-offs and worse validation RMSE. All fitted-model test paired-MSE component-bootstrap95% intervals cross zero (nine components,1,000draws/seed20261023). Full train/val/test losses, Pearson/Spearman/RMSE/TV/spread and870 per-example predictions retained. No model promotion or test-directed extension.
- **Learning:** exact final gradient norm4.92e-16; train KL.82458→.53192 while train response RMSE.71048→.81946. Posthoc saved-state analysis finds top-five peptides carry65.3% of treated signal, effective support6.81/59. Convergence of this distribution objective does not guarantee equal-peptide response accuracy. K128 reduces gradient noise but gives no consistent predictive benefit. This does not establish absence of sequence biology or identify a single cause of the broader MAPP failure.
- **Verification/resources:** pre-fit finite differences8.21e-11 and exhaustive K2 score-gradient expectation4.16e-17;4,096 K32 groups all within.57 Monte Carlo standard errors. Independent NumPy Adam/sampling replay verifies14,000updates,28checkpoints, optimizer/RNG states,870predictions,30metric sets, ridge least squares and bootstrap/gates; maxweight discrepancy8.60e-16. Fits/eval2.866s/.331GiB; independent.480s/.205GiB. Optimizer time exact.218s, REINFORCE32/128 means.246/.250s; all score118,000training policy rows, target terms118,000/64,000/256,000. Corrected accounting adds four exact-probe reference calls omitted from original counters; no fit or metric changed. Shared materialization64.04s/3.303GiB, independent source replay1.319s.
- **Checks/next:** format/lint,50 focused Hitlist tests and strict docs pass. Normal full local suite refused.64/2.5GiB before collection with guard unchanged; source/current hosted checks and review/release are separate. All source/config/checkpoints/history/probe/prediction artifacts retained. As prespecified, advance known-substrate kinetic source qualification under Hitlist770, preserving fragment yield and MS observation as distinct quantities. Full intracellular availability goal remains active.


## 2026-10-23_0800 — Known-substrate digestion source audit

- **Date/agent:** October23,2026; Codex / GPT-6. [Report](2026-10-23_0800_codex_digestion-source-audit/README.md).
- **Contract:** original MSV000092631/PXD044463 processed archive, identifier workbook, deposited methods and15 exact FASTAs;18 files pinned by hashes/URLs. No training admission, synthetic labels, MHC binding filter or qualifier imputation. Preserve raw/filtered/corrected/selected quantity semantics and unknowns.
- **Coverage:**17 condition archives/391 entries; all outer/inner CRCs pass.388 acquisition identities,5,792,969 Skyline rows,545,664 raw peptide/run quantities,53,144 quantified final peptide/biological-replicate rows plus269 annotation-only rows. All17 final substrate sequences exactly match deposited constructs.
- **Assay/output:** purified proteasome digestion MS response over time, not cellular concentration. No model, training/pretraining, losses, held-out splits, metrics or prediction dumps; source audit only.
- **Evidence:** all raw sources join metadata; RF1 has2,278 rows whose quantity time1h conflicts with metadata/filename0h. Six contaminant-digest enzyme labels conflict. Final vectors contain62,860 zeros,4,782NaNs and9,716rows with repeated time labels; source processing forces zero-time corrected values to zero.
- **Independent replay:** H4 only:27,432 raw quantities and13,284 final vector entries match to<=5.04e-15 relative error, including444missing outcomes and1,455background-censored contributors. Synthesis-error filtering and other substrates' arithmetic remain pending; full archive integrity does not prove those derivations.
- **Runtime:** local CPU, no Modal/GPU/install. Full963,763,954byte download258.56s; archive audit32.69s; H4 replay1.98s. Memory not recorded. Presto1110d49 plus frozen dirty-state/source bundle. Source tag aSPIRE1.0/c3d6c9112b2fd88d7e9453b1f23dc338f52c172b is a reference, not an asserted deposited execution version.
- **Decision:** source supports a richer kinetic experiment after shared Hitlist#770 admission; quarantine/reconcile RF1 time conflict, preserve control/censor provenance and group all Ffh ratios. No predictive winner; previous common-shift and uniform controls remain. Follow the frozen exact bond-cut/discrete-partition design only after a coherent observation likelihood and source recipe exist. Full availability/concentration goal remains active.
- **Reproduction/artifacts:** `reproduce/launch.sh`, source hashes/config and retained failure logs; raw `/private/tmp/presto-digestion-assets`, source-assets.json/member inventory; report/results linked above.


## 2026-10-23_1018 — Shared purified-proteasome digestion source

- Date/agent: October23,2026; Codex / GPT-6. [Report](2026-10-23_1018_codex_shared-digestion-source/README.md), [exact source/launch contract](2026-10-23_1018_codex_shared-digestion-source/reproduce/launch.json).
- Source: all18 pinned MSV000092631/PXD044463 originals,17 condition archives/15 literal constructs,388 acquisitions. Shared Hitlist candidate2.0.8/catalog14; producer b1c5de4 plus frozen adapter/catalog changes. All218 previous source payloads retained byte-identically;250 package files verify and33 baseline audits match.
- Curation: no training selection. Separate raw, filtered and final layers; full original fields, modifications, splice/position claims, enzyme/substrate species, physical protocol, controls and replicate labels. Preserve2,278 RF1 time conflicts, six enzyme-label conflicts,269 annotation-only rows,62,860 forced zeros and4,782 final missing slots. No implicit negatives or concentration conversion.
- Training/pretraining/synthetic labels/loss weights/assay mapping: none. No predictive split or validation/test predictions because this is source qualification. Binding thresholds do not apply.
- Condition/outcome: shared source generation and all11 canonical relation queries pass. Independent no-Hitlist verifier replays every selected original CSV/workbook/FASTA/RData record and typed numeric state against canonical query hashes. This is typing/source conservation, not independent all-substrate preprocessing arithmetic.
- Runtime: local CPU, no Modal/GPU;379.52s build/open/query,3.884GiB process peak. Independent replay183.15s/1.001GiB. R child memory not separately measured. Exact environment/git/dirty/source receipts and verification commands retained.
- Software: Hitlist977adc2;37 focused tests and the normal guarded suite3,341 tests pass, with format/lint and strict docs. Source bytes remain exact to the frozen execution.
- Decision: source support qualified for review; no training-ready recipe yet. H4-only arithmetic proof remains narrower than the source. Next replay admitted quantitative transformations, exclude/reconcile RF1, inspect temporal repeatability and freeze canonical recipe/splits/observation likelihood. Hitlist770 and full APC concentration objective remain open.

## 2026-10-23_1021 — Independent-cut exact versus sampled kinetic qualification

- Date/agent: October23,2026; Codex / GPT-6. [Report](2026-10-23_1021_codex_independent-cut-qualification/README.md), [prospective plan](2026-10-23_1021_codex_independent-cut-qualification/reproduce/plan.md), [launch/source/environment contract](2026-10-23_1021_codex_independent-cut-qualification/reproduce/launch.json).
- Synthetic data: one six-residue substrate/five irreversible independent cuts, rates0.08/0.18/0.45/0.9/1.4 per hour. All21 contiguous products, complete fragment partitions, conserved residue mass, unit MS response, no noise/missingness/selection. Outputs are expected copies per initial molecule. No biological assay labels or binding-threshold metrics.
- Training: five directly learned log rates (no sequence encoder); matched seeds42/43/44, Adam lr0.03,1,000 full-time updates, no pretraining/regularization/GRPO modifications. Exact MSE of expected amounts versus unbiased two-independent-group score gradients with leave-one-out baselines. One-rate exact control shares same budget. Fixed final checkpoint; no validation/test selection.
- Splits: train times0.2/0.7/1.5/4/12, validation0.4/1/2.5/8, test0.1/0.6/2/6/20 hours. Temporal holdouts on one known synthetic substrate, not biological generalization. All2,940 per-span train/validation/test predictions and targets preserved.

| Variant | Median relative rate error | Median validation RMSE | Median test RMSE | Median fit seconds |
| --- | ---: | ---: | ---: | ---: |
| exact | 1.542e-16 | 4.7256e-17 | 5.0678e-17 | 0.1745 |
| sampled32 | 0.03431 | 0.0071473 | 0.0063282 | 0.2809 |
| sampled128 | 0.016633 | 0.0031173 | 0.0027702 | 0.4626 |
| constant | 0.72036 | 0.16516 | 0.14176 | 0.1685 |

- Mathematics: exact span expectations versus all32 cut states error1.11e-16, residue mass error0, two-group gradient expectation error1.82e-17. Gradient RMS relative error0.788/0.344/0.170 for groups8/32/128 at fixed initial probe. All stored probes/SE retained.
- Verification: independent NumPy state sums/analytic derivatives/Adam replay all10,000 updates and match30 noninitial checkpoints (max3.46e-14 log-rate error),2,940 prediction values and metrics (max1.93e-15). Torch used only for uniform RNG stream. Preserved initial verifier failure reveals analytically tied control ranks; explicit mathematical-class correction changes no fit or gate.
- Winner/decision: exact preferred reference while enumerable. Both sampled sizes pass frozen recovery criteria; median relative rate errors3.43%/1.66%. Synthetic success does not resolve biological production/MS identifiability, learn sequence-to-rate transfer or establish APC concentrations.
- Runtime/work: local CPU float64/one thread, no Modal/GPU. Complete checks/fits4.26s/338.2MiB; first exact fit includes startup1.068s versus0.174s later, so runtime medians are descriptive. Per-run exact105,000 span expectations versus320,000/1,280,000 sampled partitions; all25,000 bond-probability evaluations. No compute-saving claim. Code, checkpoints, predictions, raw gradient probes, both verifier attempts, source snapshots and original launch are in the family directory.

Publication follow-up: both reports and frozen artifacts are pushed at Presto1503d67. All77 closure artifacts match tracked hashes and remote README blob hashes match local files. Hitlist draft773 and issue770 retain the source/recipe distinction; Presto131 remains draft. Local3,341 Hitlist tests pass; hosted checks remain in progress. The independent-cut recovery is synthetic and does not promote any biological availability baseline.


## 2026-10-23 — All-substrate digestion arithmetic

- Agent/model: Codex / GPT-6; [report/reproduction](2026-10-23_1057_codex_digestion-arithmetic/README.md). Registered before replay, Presto5a10094 plus recorded dirty state/source hashes. Local CPU only, OMP/OPENBLAS1; no Modal/GPU.
- Data: original Soh2024 MSV000092631/PXD04446317-condition archive, source package b3cfffc4, original CSV and independently decoded RData. Source faithfulness, no predictive split or synthetic labels; no fitting/pretraining/losses/assay heads or nM/binding metrics. Exact recipe and aSPIRE reference pinned in launch.json.
- All17 conditions pass:545,664 raw quantities,1,187,416 distinct precursor rows,280,664 final slots. Maximum relative error9.30e-15 across raw/technical/biological/final layers; all4,782 missing slots and62,860 zeros match. Complete per-condition CSV/JSON and empty discrepancy ledger retained.
- New admission-relevant finding:10,746 mock zero rows inserted for CaM(b1,b2), IF2(b1), UBE2S(2), RF1(b1,b2). This means missing filtered baselines, not proof that original raw zero controls are absent. RF1 time conflict remains despite arithmetic agreement. Search/synthesis filtering not independently qualified.
- Verifier checks prior independent H4 invariants and detects deliberate in-memory raw/final perturbations plus downstream mean errors; originals unchanged. Runtime58.247s/1.332GiB peakRSS. No model winner; decision is actual-baseline-aware shared admission, preserving exclusions and short/nonbinding products. Full concentration objective stays open.

## 2026-10-23 — Shared contiguous digestion time-course admission

- Agent/model: Codex / GPT-6; [report/reproduction](2026-10-23_1123_codex_digestion-timecourse-admission/README.md). Frozen initial recipe/source and reviewed3e0b999 full archive retained. Local CPU only; no training, pretraining, synthetic labels, predictive metrics or binding thresholds because this is source/eligibility verification, not a model evaluation.
- Data: original source PCP/no splice alternative, canonical20, unique I/L-equivalent contiguous span, actual filtered and metadata-agreeing raw zero evidence, no RF1 time conflict, at least three positive nonzero times and repeated source biological labels sharing three times. Preserve raw arbitrary MS targets, all slots/aliases/reasons; no normalization or negative imputation. No MHC affinity/length cut. Original selection remains a limitation.
- Fixed components over all source final peptides and identical constructs before eligibility; all Ffh conditions grouped. Sorted component hashes round-robin5; folds0/1=test/val, others train. No response-based resplit.15 source groups;11 admitted parents.

| Partition | Parents | Conditions | Curves | Positive measurements |
|---|---:|---:|---:|---:|
| Train |6|7|23,054|91,226|
| Validation |3|3|3,832|15,037|
| Test |2|2|3,680|19,160|

- Independent Hitlist-free replay checks all50,372 curve dispositions,280,664 source slots and125,423 targets, grouping, spans and zero provenance. Admission50.335s/3.235GiB; verification7.867s/2.001GiB. Canonical original/reviewed derived exports archived, full raw paths/hash receipts explicit.
- Reviewed source migration retains every quantitative/mapping column;17 context JSON strings add exactly nine inventory counts. Original failed over-strict comparison preserved. Verification-only amendment confirms unchanged numerical targets/eligibility/partitions; explicit unit/scale and source slot statuses added. Reviewed manifest42cda75e. Subsequent raw-zero cardinality strengthening changes one already-excluded curve reason only; admitted examples/components byte-identical, receipt retained.
-35 initial focused tests; immutable3e0b999 normal guarded suite3,350passed/10warnings/176.08s, two workers, no memory override; not an integration-suite or later-head claim. Source773 merged; later recipe review/deployment remains separate. No biological predictive winner. Next train-only repeatability then frozen kinetics. Upstream review independently inspected held-out descriptive repeatability; record exposure, preserve splits and require external validation.

## 2026-10-23 — Training-only digestion temporal repeatability

- Agent/model: Codex / GPT-6; [report/reproduction](2026-10-23_1138_codex_digestion-temporal-signal/README.md). Presto5a10094/dirty state, exact source/code/configuration hashes and commands retained. Shared reviewed recipe42cda75e; no extra selection. Local CPU only,1.347s/0.936GiB, one-thread overrides, no Modal/GPU.
- Endpoint: log10 positive MS signal ratio to each peptide/condition's earliest common positive time. Matched other source biological-label response predicts each label in both directions; no-change predicts0. No learned parameter, optimizer/pretraining/synthetic biological label or nM/binding metrics. These are supplied-measurement comparisons, not zero-shot sequence predictions. No validation/test responses evaluated by this family; all67,576 train predictions are retained.
- Weight times within curve, curves within condition, conditions within parent, parents equally. Training23,054curves/11,527pairs/seven conditions/six parents. Gate frozen before outcomes: lower equal-parent MSE and improvement in at least half the parents.

| Condition | Equal-parent log10 RMSE | Pooled log10 RMSE | Pooled Pearson | Pooled Spearman |
|---|---:|---:|---:|---:|
| No change |0.98579|1.08604|undefined|undefined|
| Other biological label |0.67267|0.61799|0.81557|0.75792|

- Gate passes5/6 parents; LEDGF worsens and remains included. Ffh ratios count as one parent. Pooled prediction/target SD1.01755 are equal by the bidirectional design, not independent confirmation. Independent scalar log-ratio/source/ID and hierarchical-statistic replay checks every prediction and gate. Plot/CSV expose each parent.
- Decision: proceed to a small ordered sequence-to-cut-rate model, fixed exact expectation reference and explicitly qualified sampled/hybrid gradients. Original selection/background processing, reference noise, only six training parents and missing MS calibration remain. Subsequent upstream held-out descriptive repeatability was seen; no held-out kinetic model fit exists, and future transfer is exploratory same-study evidence. Full APC concentration objective remains active. [Next model specification](agents/codex/plans/biological-independent-cut-fit.md).


## 2026-10-23 — Biological sequence-to-cut-rate exact and sampled-hybrid fits

- Agent/model: Codex / GPT-6. [Report and reproduction](2026-10-23_1202_codex_biological-cut-rates/README.md). Registered before implementation/fitting at Prestofc4d24a plus recorded dirty state; source hashes and original commands in reproduce/launch.json and phase receipts. Local macOS CPU float64, one Torch/BLAS thread; no Modal/GPU or environment changes.
- Data: Soh2024 purified human standard20S, MSV000092631/PXD044463; shared Hitlist2.0.9 digestion_contiguous_timecourses_v1, manifest5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615. Original source-component split unchanged:6/3/2 parents;30,566 curves/125,423 positive source values. Shared contiguous/positive/baseline/time admission and original exclusions preserved. No zero or missing value becomes a negative; no MHC/affinity length selection.
- Endpoint: log10 MS signal ratio to each peptide's earliest common positive time,67,576/10,990/15,168 train/val/test targets. Original-unit predictions require measured per-peptide reference signal on every split. No calibrated MS gain or absolute concentration. Previously exposed held-out descriptive repeatability is documented; future same-cohort work is development evidence.
- Model: independent irreversible bond cuts with exact mass-conserving fragment expectations; log-linear rate variants have1/39/40 coefficients (shared rate; ordered additive left/right residue effects; plus supplied log enzyme-concentration proxy). Every allocated coefficient has gradient/update support. No pretraining, synthetic biological labels, APC/MHC/TCR/transport/clearance or detectability module. Small synthetic fixtures only verify mathematical correctness.
- Training: equal-parent hierarchical log10-response MSE +0.01/2 times squared nonintercept coefficients; Adam0.03, betas0.9/0.999, eps1e-8,2,000 full-batch updates/fixed final checkpoint, initial0.01/hour. No validation selection. No-change and shared-time quadratic weighted-LS controls retained. Assay mapping is measured temporal MS response to predicted fragment log-ratio; nM/binder thresholds do not apply.

| Condition | Train RMSE | Validation RMSE | Test RMSE | Test Pearson | Test Spearman |
|---|---:|---:|---:|---:|---:|
| No change |0.98579|1.16704|1.44253|undefined|undefined|
| Shared time |0.88014|1.10936|1.15749|-0.0551|0.1294|
| One cut rate |0.99001|1.18114|1.01798|0.2573|0.3537|
| Sequence exact |0.90327|1.09796|1.02205|0.2797|0.3843|
| Sequence + protocol exact |0.90257|1.09556|1.02057|0.2832|0.3869|
| Same model, integrated-presence sampled hybrid512 |0.91048|1.07721|1.04386|0.2337|0.3396|

- Comparison: actual sampled fit uses one seed42, fresh512 complete partitions per condition/time, same model/loss/initialization/Adam/budget, with exact residuals/probabilities retained. Integrating irrelevant cut scores reduces observed probe variance, but rare events remain noisy; naive sampled log-ratios are undefined for94.3–99.8% of weighted targets at tested budgets. No clipping/group standardization/reuse/referenceKL or compute-saving claim.
- Runtime: exact variant fits19.5–20.7s; entire exact stage72.208s/1.123GiB. Sampled training323.040s,325.157s with terminal predictions/1.124GiB, roughly16.6times exact fitting.28.672million partitions/9.417billion bond draws/47.215billion presence tests. Detailed work/timing receipts retained.
- Verification: independent NumPy derivatives/Adam replay all8,000 updates and16 noninitial checkpoints, maximum coefficient discrepancy5.72e-15; all562,404 predictions independently reproduced. All source IDs, references, hierarchical metrics and outcomes checked. Raw numerical tied-rank artifacts retained alongside canonical analytic Spearman reporting; no fit/RMSE/gate changed.18 focused tests pass after AST-identical formatting.
- Learning: outputs are nonconstant (exact test SD0.7005 versus target0.8844). Exact stationarity is reached, but no sequence model beats all controls on both held-out splits. Training-only analytic shape diagnostics show repeated envelope/concavity violations; observation effects versus process mismatch are unresolved. No biological model promoted; exact remains the tractable estimator reference. Source-linked raw/baseline observation diagnosis is next; full APC concentration goal stays active.
- Artifacts: all raw/canonical train/validation/test prediction dumps, coefficients/checkpoint histories, raw gradient estimates, comparison/by-parent tables, figures, frozen source and failure logs are in the linked family. Prior admission contains original/reviewed source and derived archives; external paths/manifests are explicit in the report.


## 2026-10-23 — Training-only digestion observation effects

- Agent/model: Codex / GPT-6. [Report/reproduction](2026-10-23_1257_codex_digestion-observation-effects/README.md). Registered before analysis, Prestoaa4c809 plus dirty-state/source hashes. Shared Hitlist2.0.9 digestion_contiguous_timecourses_v1, manifest5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615; no new curation/import. Full source_tables already expose required contributors.
- Data: original six training parents/seven purified human standard20S conditions;23,054 admitted curves/91,226 positive points/67,576 fixed log10 temporal targets. Original common times/per-peptide references/admission preserved. No validation/test evaluation by design: this is a training-only source-transformation diagnostic, not a generalization estimate. All three layers'202,728 ratio predictions and133,566 triplets retained.
- Conditions: original corrected means; restored pre-subtraction means over identical retained contributors; pre-subtraction means over all filtered contributors. No relabeling as concentration, no missing-as-negative policy, no fitting/pretraining/synthetic biological labels/loss/assay-head change. Other measured biological label predicts each label bidirectionally; no change predicts zero. nM/binder thresholds do not apply. Prior model used only for frozen training-residual associations.

| Layer | Other-label equal-parent RMSE | Pooled growth violations | Equal-parent growth violations | Pooled concavity violations |
|---|---:|---:|---:|---:|
| Corrected |0.67267|7.4745%|6.4629%|9.5863%|
| Restored |0.61972|6.9344%|5.7829%|9.5301%|
| Before residual censoring |0.62868|7.1608%|5.9906%|9.5863%|

- Findings: baseline restoration leaves4,684/5,051 growth and4,215/4,268 concavity violations.94.73% of hierarchical weight has <25% reference baseline subtraction; fixed model RMSE there0.8608 versus other-label0.6418. Changing retained source technical-label signatures marks6.13% of weight and higher disagreement, without proving physical acquisition pairing or causality. Changes in layer/dynamic range mean smaller diagnostic RMSE is not a trained-model improvement.
- Reporting correction: prior kinetic prose incorrectly called pooled7.47%/9.59% shape fractions weighted. Original numerical artifacts, models and gates are unchanged; corrected README and explicit dual reductions retained.
- Verification: independent original-field Pandas replay checks all91,226 points/195,586 contributor occurrences, ratios/triplets/weights, stored-model residuals, aggregate/per-parent statistics and shape/stratum reductions. Maximum layer relative error5.11e-15; metric discrepancy zero. Actual baseline/contributor perturbations detected. Read-only-array verifier failure preserved; outputs unchanged.
- Runtime: local CPU, OMP/OPENBLAS/MKL1, no GPU/Modal/env change. Layer replay10.915s/1.318GiB; analysis10.543s/1.087GiB; independent verification10.931s/1.857GiB. Exact commands/source hashes, full prediction/provenance dumps, figures and logs in the family.
- Decision: no model winner/promotion. Baseline subtraction alone is insufficient to account for the shape discrepancy; other measurement explanations remain. Next qualify a parsimonious substrate-entry delay, explicitly verifying which restrictions it relaxes and retaining exact/sampled references. [Next specification](agents/codex/plans/substrate-entry-comparison.md). Full APC concentration objective remains active.


## 2026-10-23_1324 — Substrate-entry differentiable versus sampled comparison

- **Date/agent:** October 23, 2026, registered13:24 America/New_York; Codex / GPT-6. [Report, all artifacts and frozen phases](2026-10-23_1324_codex_substrate-entry/README.md).
- **Source/curation:** unchanged canonical Hitlist2.0.9 digestion recipe, manifest5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615; original positive corrected MS signal, source/contributor/reference identities, contiguous construct mapping and exclusions.6training parents/7conditions/67,576ratios;3development parents/10,990ratios. Previous2test parents excluded before target construction; no new test metrics/dump because known test outcomes must not drive development.
- **Model/training:** exponential substrate-entry event followed by the complete independent-cut process, every product retained and residue mass conserved.40sequence/protocol coefficients+1entry rate, versus2-parameter one-rate+entry and saved immediate controls. Same fixed2,000full-batch Adam.03updates, float64, original initializer, no pretraining. Hierarchical log10 temporal-ratio MSE+.01/2 shared nonintercept coefficient penalty; entry unpenalized. Assay mapping: measured same-peptide time/reference MS signal ratio maps to expected fragment amount ratio. No binding/elution/TCR/stability or synthetic biological training labels; binder/nM metrics inapplicable.

| Condition | Parameters | Train RMSE | Development RMSE | Development Pearson / Spearman | Prediction SD | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| Immediate one-rate |1|0.990008|1.181136|.31469/.28869|.46325|8.49|
| Entry one-rate |2|0.990313|1.181390|.31478/.28861|.46345|238.85|
| Immediate sequence/context |40|0.902566|1.095559|.42877/.36165|.45384|8.68|
| Entry sequence/context |41|0.902838|1.095745|.42881/.36162|.45408|239.46|
| Entry sampled hybrid K512/seed42 |41|0.949424|1.113748|.40369/.34697|.44399|1295.28|

**Decision:** shared entry delay earns no gain; retain the immediate sequence model. Fitted mean delays6.10/5.92/2.85seconds approach immediate entry against hour-scale measurements. The sampled final loss worsens; exact final gradient norm.3233, no convergence or earlier-checkpoint substitution. All allocated coefficients train, but finite delay is not identified. No promotion, APC concentration or cross-peptide MS calibration.

**Qualification/verification:** original fixed-age fit stops at500 on its numerical guard; preserved source/config/log/checkpoint. Registered positive fast-entry quadrature amendment changes integration only, qualifies32nodes throughlambda1e6/hour and all actual checkpoints; initial probe values/gradients bitwise unchanged. Entry relaxes early growth power but preserves log-concavity. Initial integrated/full-score RMS relative gradient errors.35946/.47690. The sampled estimator uses continuous entry/partition draws with irrelevant-bond scores integrated out and quadrature residuals/denominators, with no clipping/std/reuse/reference penalty or score-saving claim. Actual work28.67million partitions/9.42billion bond draws/46.40billion presence terms.

Independent source/NumPy/SciPy replay verifies392,830predictions(max3.11e-15error), final objectives/gradients, all2,000sampled Adam arithmetic updates and five independently rebuilt sampled gradients(max3.99e-15). Exact-entry optimizer trajectory and every sampled gradient were not independently rerun. Raw and structural-tie-adjusted metrics retained.22focused tests pass; figure inspected. Local CPU only, unchanged environment, OMP/OPENBLAS/MKL=1; no Modal/GPU. Sampled fit overlaps read-only/capacity work, so observed wall times are not an isolated hardware benchmark. Expanded qualification64.56s/2.61GB; independent closure9.89s/1.81GB.10,000successful updates plus500preserved failed updates. Execution base311744e with frozen dirty-source snapshots and exact phase commands. Next: training-only capacity/optimization decomposition, retaining full APC availability goal.

## 2026-10-23_1401 — Training-only kinetic capacity decomposition

- **Date/agent:** October23,2026, registered14:01 America/New_York; Codex / GPT-6. [Report and reproducibility bundle](2026-10-23_1401_codex_kinetic-capacity/README.md).
- **Data/curation:** same pinned shared digestion manifest5f616170…; only6training parents/7conditions/11,527condition-span groups/67,576ratios. Both previous held-out strata are filtered before target construction. Original corrected positive values, source labels, reference times and hierarchical weights retained. No missing negatives, relabeling, binding/TCR/elution/synthetic labels or pretraining. No held-out metrics/dumps by design: this is a capacity diagnostic, not predictive evaluation. Every training prediction is preserved.
- **Fixed comparisons:** saved regularized40-feature reference; same40features without penalty; free supported condition-specific bond-rate groups; per-curve necessary-shape quadratic programs and unconstrained replicate/time means. Assay mapping is unchanged log10 fragment/time-reference ratio versus observed same-peptide MS ratio. Free groups are diagnostic identities, not a transferable model.

| Condition | Kind | Parameters | Training MSE | Training RMSE |
|---|---|---:|---:|---:|
| Saved regularized sequence |Fixed reference|40|.814625|.902566|
| No-penalty sequence |Nonstationary capacity fit|40|.779949|.883147|
| Free condition rates |Nonstationary capacity fit|2,121|.635750|.797339|
| Immediate necessary shapes |Per-curve lower bound|—|.197476|.444383|
| Entry necessary shapes |Per-curve lower bound|—|.166156|.407622|
| Replicate/time means |Disagreement floor|—|.113122|.336336|

**Training/support:** same saved initial rate field; no-penalty hierarchical MSE;2,000Adam.01updates per fit then L-BFGS-B with explicit500-evaluation cap. Both fits exhaust that cap above gradient tolerance; final states are not proven optima. All40/2,121allocated parameters change and receive gradients. Among2,299bonds,15unsupported remain fixed; identical incidence groups preserve initial observed hazards. Additional nonlinear equivalences remain possible. Numerical preflight key mismatch is preserved; correction pins the older metrics artifact, with no optimizer updates in the failed attempt.

**Shape evidence:** common per-curve log response must have nonincreasing physical-time slopes and growth power≤b(immediate) or≤b+1(entry). These necessary constraints are not a sufficient characterization of any globally consistent rate process. Their excess MSE over source-label/time means is.08435/.05303; neither means nor projections are a biological noise ceiling or predictive concentration model.50independent SLSQP checks agree to4.45e-16.

**Verification/runtime:** independent original-field replay verifies23,054KKTcertificates and67,576predictions(max3.81e-15KKT,1.78e-15prediction); actual per-curve perturbations are rejected. Separate direct-sum verification checks135,152fitted predictions(max1.68e-13), all incidence allocations and finite differences of all40sequence/48free coordinates(max1.71e-10/6.24e-12). No complete optimizer replay/global-optimum claim. Stable rank predictions/raw artifacts retained. Shape projection3.36s/1.05GB; verification2.64s/.94GB; fits24.15/24.20s; fit verification9.39s/1.05GB.4,000Adamupdates and1,000polishing evaluations. Local CPU one thread, unchanged environment, no Modal/GPU; overlaps prior sampled fit, not isolated runtime comparison. Base311744e plus exact source/config snapshots.

**Decision:** regularization and sequence/context sharing account for part of the observed training gap. Neither unconverged fit proves the remaining error is caused by process misspecification. No promotion. Next [rate-space optimization audit](agents/codex/plans/digestion-rate-space-optimization.md) examines conditioning and rate-boundary limits before expanding the biological model. APC machinery, independent MS response, transfer and full concentration validation remain unresolved.


## 2026-10-23_1438 — Direct-rate optimization and abundance degeneracy

- **Agent/model:** Codex / GPT-6; [complete report/reproduction](2026-10-23_1438_codex_rate-space-audit/README.md). Presto e1f7403 with recorded dirty state; exact stage commands/environment/source hashes retained. Local CPU, no Modal/GPU/environment changes.
- **Data/curation:** same canonical Hitlist digestion manifest5f616170…; six training substrates/seven conditions/67,576 original positive log10 time/reference MS ratios and fixed hierarchical weights. All original source/contributor identities, censor/exclusion rules and2,121 incidence groups retained. Both previous held-out strata filtered before targets; no validation/test output because this is training-only optimization/identifiability diagnosis. No imputed or synthetic biological labels.
- **Model/training:** unchanged independent-cut process, per-condition free-rate capacity diagnostic. One saved warm start per block, direct nonnegative rate coordinates, sparse analytic TRF/LSMR, Jacobian scaling, max300 evaluations/180seconds each, ftol=xtol1e-12/gtol1e-10. No pretraining/statistical penalty. Original log10 temporal signal ratio maps to predicted fragment ratio; no affinity/stability/TCR/elution heads or nM/binder metrics. Zero-rate ratio closure explicitly does not define positive fragment yield.

| Condition | Training RMSE | Interpretation |
|---|---:|---|
| Previous regularized shared40-rate features |0.902566|Fixed reference|
| Previous unregularized shared40 features |0.883147|Budget-limited|
| Previous free2,121 log-rate groups |0.797339|Budget-limited|
| Direct-rate sparse solve |0.797292|Six ftol exits; IL37b300-evaluation cap; no global optimum proof|

- **Metrics/outcome:** direct pooled Pearson0.603785, raw Spearman0.496366, SD0.785654; all675 evaluations50.159s, peak1.005GiB. Per-condition metrics/evaluations/optimality, full traces and all predictions in report. MSE improves0.01185%, while1,288rates fall below1e-10/hour. Reference amounts below1e-20copies/parent increase from0.186% to53.732% of rows (51.799% of weight); median log10 amount drops14.28. Ratio fit does not identify concentrations.
- **Verification:** high-precision boundary/derivative, sparse-JVP and constrained-toy checks pass; independent raw-source/direct-sum replay verifies67,576predictions and all2,121gradients (max2.61e-14/5.79e-16), all target/weight values and saved final trace membership. StageA2.530s/1.012GiB; verification3.153s. Original Spearman-agreement failure retained; amended report preserves both near-tie raw ranks differing1.53e-7 with no fit/output replacement. No optimizer trajectory replay/global proof claim.
- **Decision:** no transferable model promotion; stop local solver-budget expansion. Next frozen shared-observation comparison retains cross-peptide signal, profiles only condition-wide scale and qualifies exact/sampled gradients before fitting. Canonical run/instrument contributor projection belongs to770, broader known-load response to762;775 stays closed. Full APC concentration goal remains active.

Publication closure: report/artifacts committed and pushed at `eb96219`; all59 recorded artifact hashes match committed blobs and immutable remote README bytes match. Hitlist770/762 and Presto131 updates have exact body/state readbacks; draft131 and closed775 preserved. Parent e1f7403CI passed; current experiment CI40007342075 is in progress. All local fitting/verification/publication processes are terminal. No biological model is promoted.


## 2026-10-23_1501 — Shared MS observation and conditional sampled kinetics

- **Agent/model:** Codex / GPT-6; [full report and artifacts](2026-10-23_1501_codex_shared-observation/README.md). Prestoce86bdb with unrelated dirty work recorded; clean Hitlist61baed9 unchanged. Local CPUfloat64/one thread only, no Modal/GPU/environment changes.
- **Source/curation:** unchanged canonical digestion recipe5f616170/sourcead62414d.6training parents/7conditions/23,054curves/90,630common-positive signals;3development parents/3,832curves/14,822signals. Original corrected MS1 arbitrary units, source slots, positive selection, missingness and exclusions preserved; no MHC/affinity filter or negative imputation. Previous test parents excluded before collection; no new test loss/dump because outcomes are already exposed.
- **Context:** existing EvidenceStore raw-quantity/acquisition join suffices;365,001rows and105,452aggregate memberships independently replay. All candidate runs agree on instrument;21,718observations have multiple candidate forms. Exact technical/form contribution remains unresolved. No new Hitlist adapter needed; receipt metadata-variable defect documented separately.
- **Training:** same40sequence/protocol kinetic coefficients; add38training-supported instrument/length/composition response directions. One analytically profiled offset per condition, no peptide IDs or per-peptide calibration. Hierarchical log10-signal MSE+.01/2 squared nonintercept/response weights. Original ratio checkpoint warm start,zero response; Adam.03,2,000full-batch updates,fixed final checkpoint. No pretraining/new synthetic biological labels. Only purified-digestion signals train; no binding/elution/TCR/stability/APC labels or nM/binder metrics.

| Variant | Train shape RMSE | Development shape RMSE | Development Pearson / Spearman | Development temporal-ratio RMSE |
|---|---:|---:|---:|---:|
| Condition scale only | 1.21471 | 1.28060 | undefined | 1.16704 |
| MS-response features only | 1.17668 | 1.27936 | 0.0961 / 0.1040 | 1.16704 |
| Shared time quadratic | 1.19993 | 1.27383 | 0.0876 / 0.0835 | 1.10889 |
| Old ratio-trained kinetics, fixed | 1.24799 | 1.29302 | 0.2151 / 0.1970 | 1.09556 |
| Old kinetics fixed + fitted response | 1.20344 | 1.29793 | 0.2187 / 0.2046 | 1.09556 |
| Refitted kinetics, constant response | 1.19004 | 1.26809 | 0.2363 / 0.2273 | 1.17697 |
| Exact kinetics + sequence response | 1.13843 | 1.26510 | 0.2561 / 0.2452 | 1.16391 |
| Conditional sampled hybrid + response | 1.13906 | 1.26279 | 0.2572 / 0.2464 | 1.15644 |

- **Estimator qualification:** unconditionalK512 fails its frozen mean-gradient gate (uniform-state meanerror.20195;72–73%observed span/time events unseen). Separately prospective conditional complete-partition importance proposal passes stricter mean/RMS gates: .0444/.1476 and.0410/.0856. Enumerated32-state density/score proof and32,768draw-law check pass. K32/128 and uniform proposals retained. The hybrid retains exact forward/response/conditional scores and currently computes the exact backward before replacing kinetic gradients; no pure simulation or score-saving claim.
- **Learning:** exact development MSE gain2.406%versus constant and.471%versus kinetics-only; H4 remains worse than constant. Old temporal-ratio error worsens1.09556→1.16391. Conditional1.26279 versus exact1.26510 is one seed, not a robust RL winner. All40/78/78allocated coefficients train. Exact sequence-response/sampled final gradient norms.0180/.0481 are nonstationary. Response can span2.90log10 units; absolute calibration/transfer unresolved. Fixed-response rate-scale probes constrain one direction, not joint identification.
- **Verification/runtime:** all843,616final predictions,6,000Adam arithmetic updates, final gradients,15selected exact/same-RNGsampled gradients, ridge/feature support and15sensitivity probes independently replay. Maximum prediction error1.07e-14,gradient7.44e-15,Adam8.88e-16; no raw rank discrepancy. Every sampled gradient/form contribution is not independently verified. Initial1.18e-12dense-LS qualification failure and compensated-scalar amendment preserved. Exact combined fit28.96s,sampled440.89s; whole exactfamily70.70s/1.17GiB,sampled444.34s/1.06GiB; overlapping verification means no isolated benchmark.28.672million sampled partitions; no network score saving.
- **Winner/decision:** exact is preferred for this tractable expectation problem; conditional hybrid is a qualified stochastic reference. No biological/concentration model promotion or stable-baseline replacement. Canonical report includes every source/hash/command, checkpoints,predictions,diagnostics and plots. Next prospective [single fragment-state length effect](agents/codex/plans/fragment-state-length-qualification.md) must pass exact/discrete numerical qualification before biological fitting. Hitlist770/762 retain form/calibration scope;775 stays closed. Full APC availability/concentration objective remains active.


## 2026-10-23_1816 — Complete-fragment state dependence and runtime qualification

- **Date/agent:** October23,2026; Codex / GPT-6; [family](2026-10-23_1816_codex_fragment-state-qualification/README.md).
- **Model:** effective bond hazard multiplied by `(fragment_length/20)^eta`, eta=-1/0/+1. All spans1..L retained; binary cuts conserve residues. Exact O(L²) moment action/positive uniformization versus complete continuous-time branching paths with event, waiting and terminal-survival scores. Eta0 recovers independent cuts.
- **Source/curation:** input-only projection of unchanged canonical digestion recipe manifest5f616170… and known40-coefficient initializer2400f214…. Six training parents/seven conditions, lengths123–533; no quantities/new labels loaded. No source eligibility, censor or assay-label change.
- **Training/evaluation:** no fitting/pretraining/optimizer or synthetic biological labels. Numerical fixture five rates [.02,.07,.11,.03,.15], eta=-1/0/+1, times0/.5/2/8; arbitrary linear readout. No held-out validation/test split because this is numerical/input-runtime qualification. Biological Spearman/Pearson/IC50/binder metrics inapplicable; full numerical arrays/statistics retained. No pMHC/TCR/MS head trained.
- **Numerical result:** all12 expectation/gradient cases and9 primary K65536 Monte Carlo gates pass.10,764,288 paths; independent32-state generator/Frechet replay maximum count/gradient discrepancy1.39e-15/1.18e-14; all432 groups and288 complete event histories replay. Primary mean-relative-gradient error0.000390–0.00552 and RMS0.00630–0.0235. SmallerK1024/8192 diagnostics retained.

| Actual-parent condition | Exact completed | Sampled completed | Decision |
|---|---:|---:|---|
| eta=-1, seven conditions |7/7|7/7|numerically executable|
| eta=0, seven conditions |7/7|7/7|independent-cut amounts/gradients replay|
| eta=+1, seven conditions |3/7|7/7|bothFfh,LEDGF,hTau exact60s timeouts|

- **Precision/runtime:** K512 one-group tagged-residue TV0.146–0.744 and40-feature/eta gradient relative error0.0346–0.185 across17 matched cases; artificial readout, no biological variance guarantee. All38 successful full outputs replay. Qualification10.565s/424MB;42-worker suite332.458s; independent verification0.634s/199MB. CPUfloat64/complex-free reference, one thread, no Modal/GPU or environment mutation.
- **Reproduction:** HEAD0361e0d plus explicitly frozen dirty module/test sources; launch/runtime/verify scripts and hash configs under reproduce/, all worker arrays and failures under results/ and launch_logs/. Nine focused production tests pass1.59s; no full-suite claim.
- **Winner/decision:** mathematical formulations qualify, but differentiable runtime gate fails. No biological model promoted or new fit admitted. Preserve full support and qualify a structured triangular shifted solve/CRAM exponential under the next frozen plan. Sampling speed does not establish distribution fidelity. The full APC concentration goal remains active; no new Hitlist data gap arose from this numerical experiment.


## 2026-10-23_1846 — Structured full-fragment rational solver

- **Date/agent:** October23,2026; Codex / GPT-6; [family](2026-10-23_1846_codex_fragment-rational-solver/README.md).
- **Model/source:** same complete-fragment length-dependent cut model and canonical frozen training-parent inputs74f615a7…; manifest5f616170…/40-coefficient initializer2400f214…. All lengths1..L, eta=-1/0/+1, no new measured quantities, source recipes or labels. The O(L²) decreasing-length shifted solve feeds fixed IPF CRAM16/48; positive uniformization and discrete full-path references remain unchanged.
- **Training/evaluation:** no fitting/pretraining/optimizer/biological loss or synthetic biological labels; arbitrary numerical count readout only. No validation/test split or assay metrics because this is input-only numerical/runtime qualification. No binding/stability/TCR/MS-response head trained. Full270 numerical predicted/reference count/gradient records and50 full-parent outputs retained.
- **Numerical gates:** all27 shifted-solve value/gradient probes and270 dense exponential/Frechet cases pass accuracy. CRAM16/48 maximum count error1.73e-13/5.47e-15, gradient2.44e-12/2.70e-13. Strict positivity fails24/19 numerical cases, minimum−9.53e-15/−9.13e-45. Values are not clipped or floored.

| Actual-parent variant | Completed | Seconds/case | PeakRSS | Cases with negatives |
|---|---:|---:|---:|---:|
|CRAM16, one step|21/21|0.182–1.363|554MB|13|
|CRAM48, one step|21/21|0.552–4.156|1,180MB|10|
|CRAM16, two steps on prior timeouts|4/4|1.960–2.757|874MB|4|
|CRAM48, two steps on prior timeouts|4/4|6.082–8.458|2,100MB|4|

- **Verification/runtime:** 50-worker suite138.279s; qualification4.244s/256MB; independent replay23.178s/169MB. All50 full arrays/readouts and negativity records replay. CRAM48 matches17 preceding uniformization count/gradient arrays to3.78e-15/2.39e-15; eta0 independent formulas pass. Three independent FD directions on16 formerly-timed-out variant cases agree within1.11e-10. Those directions and cross-order agreement are not a universal true-error certificate. All original failures preserved.
- **Reproduction:** HEAD2903e11 plus frozen dirty experiment sources, full stage commands/configs/hashes/source copies. Float64/complex128 one-thread local CPU, no Modal/GPU or environment mutation. Production fragment module unchanged; separate prior nine-test evidence remains. Original family165 closure hashes and immutable remote README verified at2903e11.
- **Decision:** CRAM48 is the more accurate numerical candidate and fixes the observed runtime limitation, but neither method is admitted to biological log training under the frozen positivity rule. Next: positive boundary-conditioned full-path importance means/scores, including rare-product log and finite-sample bias/variance qualification. No new Hitlist data gap; machinery/observation/clearance and the full APC concentration objective remain open.

Publication closure: report/code/artifacts at556e2ee; all186 closure hashes and immutable remote README match. PR131 body-only update reads back exactly and preserves open/draft state. Prior2903e11CI succeeds;556e2eeCI runs40037397437/40037390050 are in progress. All local experiment handles terminal. Full goal remains active; next is the prospectively frozen boundary-importance qualification.

## 2026-10-23_1909 — Positive boundary-conditioned importance sampling

- **Date/agent:** October 23, 2026; Codex / GPT-6. [Full report](2026-10-23_1909_codex_boundary-importance/README.md).
- **Data/curation:** Fixed six-residue numerical system, five bond rates, eta=-1/0/+1, three times/five target spans and three prescribed rare stresses. No biological targets, source changes or new curation. Full fragment support, including monomers and nonbinding products, remains intact.
- **Model/estimator:** Original fragment-length hazard; independently uniform forced target-boundary times with all other allowed cuts sampled under original hazards. Exact complete path density ratio restores forbidden-event survival. No floor/clipping; positive log-space count estimates and original-path score gradients.
- **Training/assays:** Zero optimization or pretraining, no assay labels/loss weights or synthetic biological labels. Numerical reference probabilities/gradients only. No held-out validation/test split because this is estimator qualification; complete group/raw/path numerical dumps replace predictive assay dumps. Binding/nM metrics do not apply.
- **Contract/reproduction:** Same registered45 ordinary/3 rare states, K256/2048/16384,16groups, fixed seeds and K16384 gates. Source702d66b with dirty status, code hashes, frozen source and launchers in reproduce/. CPU float64 one thread, OMP/OpenBLAS/MKL=1; no Modal/GPU or environment changes.

| Primary scope | Passed | Worst probability relative error | Worst absolute mean log bias | Worst RMS gradient error |
|---|---:|---:|---:|---:|
| Ordinary |45/45|0.002009|0.002004|0.244641|
| Rare |2/3|0.066063|0.053024|0.100319|

- **Runtime/verification:** 14,352,384paths/2,304groups,24.81s/318MB. Independent21-fragment generator/Frechet and closed rare-eta reference verify all groups,48raw primary groups and384full event histories; max trace error9.10e-13. Three focused tests pass; initial import-path collection failure and corrected launcher retained.
- **Decision:** Uniform boundary proposal fails fast-boundary stress; ESS34.55/16384. Log estimates remain accurate near log probability-1682 despite all linear weights underflowing. No full-parent/biological fit admitted. Existing unconditional zero-hit reference retained at its different K; no matched-cost claim. The separately registered truncated-time proposal is the next comparison; full APC availability/concentration goal stays open.

## 2026-10-23_1920 — Truncated boundary proposal, score control and full-parent failure

- **Date/agent:** October23,2026; Codex / GPT-6. [Full report and tables](2026-10-23_1920_codex_boundary-exponential/README.md).
- **Data/curation:** Same numerical matrix plus frozen canonical training-input projection74f615a7… from shared purified human20S digestion package5f616170… and original40-coefficient initializer2400f214…. Seven conditions/six full constructs,123–533residues; no quantity/development/test tables read, no new source/exclusions or biological fitting.
- **Model/estimator:** Same complete branching process. Required boundary times use initial-fragment truncated exponentials; exact original/proposal density scores retained. Primary leave-one-out proposal-score control has unbiased unnormalized numerator, but normalized log-gradient estimates retain finite-K bias. Plain scores preserved; no clipping or probability floor.
- **Training/assays:** Zero new optimizer/pretraining steps; no measured assay losses/labels or synthetic biological training labels. No held-out predictive validation/test split: fixed-state numerical qualification using training-parent inputs only. Numerical arrays/path/group dumps retained; regression/binder metrics inapplicable. No APC, MS-response or pMHC validation implied.
- **Reproduction:** Source702d66b/recorded dirty state, exact launch/source hashes and commands in reproduce/. CPU float64 one thread, OMP/OpenBLAS/MKL=1; no Modal/GPU or environment mutation. Small qualification14,352,384paths/18.25s/320MB; all48primary states pass. Fast-boundary controlled/plain RMS gradient error0.000496/0.005915; ESS16384. All48raw groups/384saved histories independently replay.
- **Full-parent contract:** eta=-1/0/+1,24h, centered lengths1/9/40 selected from parent length, fixed signed log readout and all41parameter gradients, K64/512×8groups; primary512. CRAM48 two/fourstep target references positive and agreeing within1e-5; full-output tiny-tail negativity remains unresolved. Every case retains its original status.

| eta | Primary passed / 7 | Worst mean absolute log error | Worst mean gradient error | Worst RMS gradient error | Minimum ESS / 512 |
|---|---:|---:|---:|---:|---:|
| -1 |0/7|0.185814|0.339250|0.749008|17.11|
| 0 |7/7|5.77e-15|0.011553|0.022798|512|
| +1 |1/7|0.529441|0.594019|1.358499|2.76|

- **Verification/runtime:** All21workers complete,290,304paths/467.45s/maximum1.98GiB. Independent full-array replay,63directional finite differences (max1.70e-7), all group algebra,63raw groups and504saved full histories pass in85.20s. This is not every-component or every-unsaved-path verification. Three focused tests pass.
- **Reporting correction:** Worker used absolute signed mean log error; registered plan required mean absolute log error. Independent closure enforces original criterion without changing outputs/seeds/thresholds. Three primary decisions change11/21→8/21; five change across both K. Both gates and reductions retained.
- **Decision:** No full-parent/biological admission. Initial-rate proposal mismatch under changing fragment lengths is a hypothesis; larger states/lower K also differ from toy checks. One [prospective state-adaptive proposal](agents/codex/plans/fragment-boundary-adaptive.md) is frozen with unchanged physical law/cases/counts/gates and a defined biological next-stage gate. No new Hitlist data gap arose; original MAPP failures, independent observation calibration and full APC concentration objective remain unresolved.

## 2026-10-23_2000 — State-adaptive boundary proposal

- **Date/agent:** October23,2026; Codex / GPT-6. [Full report](2026-10-23_2000_codex_boundary-adaptive/README.md).
- **Source/curation:** Same fixed numerical fixture and canonical seven-training-condition/six-construct projection74f615a7…, package5f616170… and original initializer2400f214…. All required target definitions/counts/seeds/gates unchanged; no source quantities, development/test observations, new curation or fitted biological labels.
- **Model/estimator:** Same full-support fragment-length-dependent branching law. Only proposal changes: recompute each required boundary's truncated-exponential hazard after every cut, including exact adaptive waiting/event densities and proposal scores. All outside fragments retained; no clipping/floors. Same leave-one-out proposal-score control; normalized logs/gradients retain finite-K bias.
- **Training/evaluation:** Zero optimization/pretraining, no assay loss or synthetic biological labels. No held-out predictive split: fixed mathematical/reference qualification and training-input-only study. Numerical counts/gradients/raw paths substitute for assay prediction dumps; nM/binder metrics inapplicable. No APC-concentration or MS-response validation.
- **Reproduction:** Source2c6cb97/recorded dirty state, exact source/launch/test/verifier snapshots in reproduce/. CPUfloat64 one thread, OMP/OpenBLAS/MKL=1; no Modal/GPU or environment changes. Six analytical/path tests plus the explicit mean-absolute-error cancellation test pass. Small matrix14,352,384paths/23.21s/317MB; all48primary cases pass, worst ordinary RMS gradient error0.012494. All2,304groups/48raw groups/384saved histories independently replay.

| eta | Previous proposal passes | Adaptive passes | Worst mean absolute log error | Worst mean gradient error | Worst RMS gradient error | Minimum ESS / 512 |
|---|---:|---:|---:|---:|---:|---:|
| -1 |0/7|0/7|0.175523|0.359927|0.815162|7.96|
| 0 |7/7|7/7|1.57e-14|0.008050|0.021198|512|
| +1 |1/7|0/7|0.095046|0.084908|0.228954|61.37|

- **Full-parent runtime/verification:** All21workers complete;290,304conditional paths,333.98s total/308.55s inside sampler calls,323MB maximum worker RSS. Exact references are reused by verified hash; prior totals included construction, so no matched end-to-end speedup claim. Independent replay verifies63raw groups,504full saved histories, all group algebra and original mean-absolute gate; max path discrepancy8.67e-13,3.238s. No new reference finite differences or unsaved-path replay is claimed.
- **Decision:** Positive-eta worst gradient RMS improves1.3585→0.2290 and worst logMAE0.5294→0.0950, but all length-dependent cases still fail atK512; overall7/21 versus prior8/21. Stop this proposal extension without sample tuning. The [next full-observation admission specification](agents/codex/plans/fragment-observation-domain.md) checks every actual training span/time before a bounded differentiable biological comparison, keeping the sampled failure and prior controls visible. It explicitly limits numerical admission to required outputs without retroactively passing old blanket positivity gates. Full APC goal remains active; no new Hitlist gap arose.


## 2026-10-23_2029 — Complete training-observation fragment numerical admission

- **Date/agent:** October23,2026; Codex / GPT-6. [Full report](2026-10-23_2029_codex_observation-domain/README.md).
- **Source/curation:** Unchanged canonical Hitlist digestion recipe manifest5f616170… and original40-coordinate initializer. Every90,630training observation is retained:45,315unique span/time queries,seven conditions/six constructs,123–533residues,1/2/4/24h. Complete source JSON is parsed, including excluded records, but measured quantities are never consulted by projection or numerical decisions. Support identities and equal-parent/condition/curve/time weights independently reconstruct. No new exclusions, negative labels, ambiguous-origin assignments or source curation.
- **Model:** Complete1..L fragment branching law `exp(x@theta)*(length/20)**eta`, with40ordered sequence/protocol coefficients and one global eta. Expected copies per initial parent, without observed-set normalization, floors or clipping. Eta fixed−1/0/+1; no fitted biological parameter or MS-response update.
- **Training/evaluation:** Zero new optimizer/pretraining updates or biological assay loss. No predictive held-out validation/test evaluation because this is numerical admission; original test parents stay excluded and development is not a new blind test. All135,945required span/eta predictions and full arrays are retained. Assay correlations, nM/binder metrics and synthetic biological labels do not apply.

| eta | Admitted condition/time cases | Unresolved required queries |
|---|---:|---:|
| −1 |28/28|0|
| 0 |28/28|0|
| +1 |28/28|0|

- **Precision:** Every required value is positive and finite. Maximum two/four/eight-step natural-log disagreement 1.5099e-14 against1e-5criterion. All252directional checks agree within registered1e-5+1e-6*absolute-FD tolerance; maximum discrepancy2.00716e-10. Independent NumPy eight-step recomputation of all84full outputs gives maximum full-array error9.4369e-16 and required-log error1.24345e-14. This is fixed-state empirical qualification, not a universal gradient/error proof.
- **Scope limitation:** Full arrays still contain tiny negative unobserved tails (maximum41,297entries/call), all preserved. This prospectively registered required-output gate does not retroactively pass earlier blanket positivity or qualify arbitrary unseen queries. Maximum absolute residue-mass discrepancy5.11591e-12. Every future selected training block and final full observation domain must be checked again.
- **Verification/runtime:** All84workers terminal; suite1464.95s, maximum2.055GiB. Three focused tests pass, including dense-generator/variable-time and analytical-refinement counterexamples. Independent full-reference work83.26s; source/array/gate closure4.40s. Worker eta0 scalar compares eight steps; closure enforces analytical agreement across all three, changing0decisions. Original import-path preflight failure preserved; corrected family PYTHONPATH invocation passes.
- **Reproduction:** HEAD6347987636c2de32a4f20001c356fb1d24febbc7 with recorded dirty state, source hashes, exact support projection, frozen scripts/launches/configs in reproduce/. CPUfloat64/complex128, OMP/OpenBLAS/MKL=1; no Modal/GPU or environment changes. Hardware sysctl query was sandbox-denied and is recorded rather than guessed. All168primary/independent raw array files are byte-verified archives with restore helper.
- **Decision:** Differentiable solver is numerically admitted on the complete required training domain at the frozen states. The sampled adaptive estimator's prior7/21full-parent result remains unchanged; no sampled biological fit is admitted. Next: [qualify matched offset/minibatch objective](agents/codex/plans/fragment-minibatch-objective.md), freeze bounded biological execution and test whether the one added length effect improves real development performance. No new Hitlist gap; live reads confirm770/762/757open and739closed. Independent response calibration, APC machinery/transport/clearance and cellular concentration validation remain open.


## 2026-10-23_2108 — Real biological fragment-length fit: small gain, no new best model

- **Date/agent:** October 23, 2026 local registration, October 24 UTC execution; Codex / GPT-6. [Full report and artifacts](2026-10-23_2108_codex_fragment-biological-fit/README.md).
- **Source/curation:** Unchanged canonical Hitlist recipe manifest `5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615`, MSV000092631/PXD044463 purified human standard 20S digestion. Original positive background-corrected MS signals; 90,630 training measurements from six parent constructs/seven conditions and 14,822 development measurements from three parents. Existing complete-case eligibility, contributor/modified-form ambiguity, instruments and equal-parent/condition/curve/time weights preserved. No extra filtering, new negatives or local curator. Only this digestion-MS endpoint trains; IC50/pMHC/TCR/APC machinery and concentration outputs are absent. No censor inequalities or synthetic biological labels.
- **Models:** Analytical independent-cut expectation versus full fragment-branching expectation with one added bounded scalar, `eta=tanh(rho)`, hazard `exp(x@theta)*(current_length/20)^eta`. Both retain all fragment lengths 1..L, including monomers/nonbinders, and predict expected copies per initial precursor. Shared 40 rate coefficients, 38 instrument/length/composition response coefficients and seven condition-scale offsets; fragment adds the 86th parameter. Original corrected log10 MS signal maps to log10 expected count + relative response + scale. MS response is jointly estimated, not independently calibrated detectability.
- **Training:** Same original 40-coordinate ratio-trained initializer, zero response, optimal initial offsets, 200 identical seed-42 condition/time-block draws (all 28 visited), Adam lr .03 / betas .9,.999 / eps 1e-8, no clipping/decay/dropout. Probability-weighted block MSE has the full hierarchical objective as its expectation. Add .005 squared nonintercept rate/response coefficients and .005 physical eta squared; offsets unpenalized. Final checkpoint only, no development-selected extension/retry or new pretraining. Both arms completed; every 85/86 allocated coordinate has material gradients and updates. No new RL fit: the earlier length-dependent sampler remains unqualified.

| Condition | Train shape RMSE | Development shape RMSE | Development Pearson / Spearman | Development temporal-ratio RMSE | Seconds / peak GiB |
|---|---:|---:|---:|---:|---:|
| Analytical, 200 joint-offset block updates |1.18192|1.28894|.2320 / .2283|1.23295|14.86 / 1.222|
| Fragment length effect, same 200 updates |1.16259|1.27630|.2401 / .2354|1.20013|1692.91 / 2.262|
| Prior constant scale |1.21471|1.28060|undefined|1.16704|prior family|
| Prior time quadratic |1.19993|1.27383|.0876 / .0835|1.10889|prior family|
| Prior exact + response, 2,000 full batches/profiled offsets |1.13843|1.26510|.2561 / .2452|1.16391|prior family|
| Prior sampled hybrid + response, 2,000 full batches/profiled offsets |1.13906|1.26279|.2572 / .2464|1.15644|prior family|

- **Evaluation limits:** Shape metrics profile one condition scale using development labels; they evaluate relative MS signal shape, not absolute signal/concentrations. MSE is hierarchically weighted; correlation and SD are pooled/unweighted. Fragment development prediction SD .6111 versus target 1.2524. Development parents were previously inspected; original test parents remain deliberately excluded to keep this optimizer/mechanism diagnostic from consuming test data. Full train/development prediction dumps are retained; no new test prediction/performance claim or binder/nM metrics apply. Older longer fits are contextual, not matched optimization budgets.
- **Optimization finding:** Final total training objectives 1.620957 analytical / 1.508486 fragment versus common initial 1.595535. Independently verified nonoptimal-offset MSE gaps .196090 / .129245. Final fragment eta .197406. This establishes an incomplete nuisance optimization problem, not the cause of weak generalization or a causal length effect.
- **Verification/resources:** Both 200-step optimizer trajectories independently replay; every analytical gradient, fixed fragment gradients at steps 1/100/200 (nine rate/eta directions plus 45 linear coordinates per state), all 105,452 predictions per arm, and all global metrics independently verify. Separate NumPy eight-step final physical expectations agree; maximum fragment natural-log discrepancy 1.23e-14. Every selected block before/after update and final query passes required positive finite 2/4-step checks; tiny unobserved negative tails remain unqualified. CPU float64/complex128, one thread, no Modal/GPU/environment changes; full suite 1712.01 s. Descriptive runtimes include guards/artifacts/evaluation and limited verifier overlap, not isolated benchmarks.
- **Reproduction:** Source HEAD `120d55ac26b1906c0d22c465254c4ab0051470da`, recorded dirty state; exact external hashes, source snapshots, original qualification launch and separately frozen fitting/verification receipts in `reproduce/`. Initial `launch.json` remains the zero-update qualification receipt; `fitting.json`/`fit.sh` and completed receipts record the subsequent 400 updates. All original states, moments, checkpoints and reference arrays are byte-verified archives; prediction gzip files are direct. Original outputs are not replaced.
- **Decision:** Fragment beats its matched control and barely beats constant shape RMSE, but loses to the older exact/hybrid and even time-quadratic controls. It does not improve temporal-ratio RMSE over constant. **No new best biological model and no validated APC concentration model.** Next [profile nuisance offsets exactly](agents/codex/plans/fragment-profiled-minibatches.md) at each current parameter state, holding biology/data/schedule fixed. No new Hitlist limitation arose; existing independent calibration/precursor gaps remain. Full availability goal stays active.


## 2026-10-23_2201 — Profiled offsets improve digestion fits, with weak biological prediction

- **Agent/model/source:** Codex / GPT-6, source a6c6559 with recorded dirty state; [completed report and reproduction](2026-10-23_2201_codex_profiled-fragment-fit/README.md). Registered before implementation and updates. Exact executed fitting/source/data hashes retained.
- **Data:** Unchanged MSV000092631/PXD044463 canonical Hitlist digestion manifest5f616170…; 90,630 training/14,822 development positive corrected MS observations. Same source forms/contributors/instruments, partitions and hierarchical weights as prior 2108 family. Only digestion-MS targets train; no IC50/MHC/TCR/APC machinery labels, inequalities, synthetic biological targets or new negatives. Original test parents excluded from this optimizer diagnostic; previously inspected development is not a blind test. Full prediction dumps retained per completed arm.
- **Hypothesis/change:** Replace seven learned condition offsets with exact weighted optima at every current parameter state, detached for sampled-block gradients. Same one-scale-per-condition objective, rates/response/length effect, original ratio-trained initializer, 200 seed42 draws, Adam .03/.9/.999/1e-8 and .005 squared nonintercept/response/eta penalties. Allocate 78/79 parameters; no new pretraining, clipping, dropout or development-based selection. No new RL arm is admitted. All-fragment expected copies + relative response + condition scale predicts original log10 MS signal, not concentrations.
- **Pre-fit qualification:** Toy counterexample rejects per-time profiling; all 28 actual blocks at two predetermined states agree with full profiled objective and independent gradients (<3e-14 error). Actual eta-zero common-gradient parity passes. Longest-parent target-free cost26.09s sets a prospective 8,400s/4GiB cap. Local CPU float64/complex128, one thread, no Modal/GPU/environment change.

| Condition | Train shape RMSE | Development shape RMSE | Development Pearson / Spearman | Development temporal-ratio RMSE |
|---|---:|---:|---:|---:|
| Analytical, exact offsets, 200 draws |1.164345|1.264868|.254680 / .245122|1.144171|
| Fragment, exact offsets, 200 draws |1.156478|1.262718|.253056 / .242653|1.130980|
| Prior analytical, learned offsets, same 200 draws |1.181919|1.288941|.232033 / .228289|1.232951|
| Prior fragment, learned offsets, same 200 draws |1.162592|1.276300|.2401 / .2354|1.200130|
| Constant condition scale |1.21471|1.28060|undefined|1.16704|
| Earlier exact, 2,000 full batches |1.13843|1.26510|.2561 / .2452|1.16391|
| Earlier hybrid, 2,000 full batches |1.13906|1.26279|.2572 / .2464|1.15644|

- **Analytical evidence:** All 78 coordinates materially train; all 200 gradients/updates and 105,452 source-linked predictions/metrics independently replay. Final training objective1.378363 versus initial1.595535; residual offset gap1.95e-29MSE. Runtime7.76s, peak1.224GiB. Development prediction SD.60448 versus target1.25239. Shape metric profiles scales using development labels, not absolute signal; MSE weighted, correlations pooled/unweighted. No binder/nM metrics apply.
- **Fragment evidence:** All 79 coordinates materially train; final eta .317001. All 200 Adam updates and 105,452 source-linked final predictions/metrics independently verify; nine fixed rate/eta directional checks at steps 1/100/200 plus all 38 response components at those states pass. Other fragment gradients are not independently recomputed. Maximum final physical natural-log difference 1.33e-14. Final training objective1.360604 versus initial1.595535; residual offset gap2.50e-29MSE. Runtime3051.87s, peak2.364GiB; full two-arm suite3061.99s. Prediction SD.57841 versus target1.25239. Numerical guards pass for required outputs; tiny negative unobserved tails remain unqualified.
- **Decision:** The nuisance optimization fix improves both matched models. Fragment beats analytical by only .00215 development RMSE, with slightly worse rank correlation; its1.26272 is effectively tied with the earlier sampled hybrid1.26279 on this small, already inspected cohort. The gain over constant is about1.4% RMSE. No robust biological superiority or APC concentration validation is established. All fitting/verification handles are terminal, every train/development prediction is retained, and raw checkpoints/optimizer/reference arrays are byte-verified archives. The paired result does not admit a new length-dependent RL sampler. Target persistent cross-peptide signal error and independent MS-response support next, before further sampler expansion; existing MAPP/calibration/precursor limits and full availability goal remain open.


## 2026-10-23_2214 — Persistent versus temporal training residuals

- **Agent/model/source:** Codex / GPT-6, a6c6559 with recorded dirty state; [report, plot, per-curve evidence and reproduction](2026-10-23_2214_codex_residual-decomposition/README.md). Prospective arithmetic plan and all input hashes frozen before calculation. This is a diagnostic of fixed predictions, not another fit.
- **Data/target:** Same canonical Hitlist digestion manifest5f616170…, MSV000092631/PXD044463; all90,630 training corrected-positive log10 MS observations,23,054 original source curves, seven conditions/six constructs. Preserve every source identity/value/weight and original parent/condition/curve/time reduction. No new negatives, censor reinterpretation, synthetic biological labels or source curator. Full input files include development records, which are parsed but skipped before their quantities enter the diagnostic. No new validation/test metrics or predictions because this is training-only residual arithmetic; original test excluded. Existing full hash-pinned predictions remain per-example evidence. Binding/nM metrics do not apply.
- **Contract:** Zero optimizer/pretraining updates or new assay loss. For original condition-centered residuals, split weighted MSE exactly into squared curve-mean residual and residual variation within that curve. Source curves include biological-label and unresolved contributor/form scope; these are not independent unique molecular peptides or pure detector/processing terms. Compare five completed fits and a constant condition-scale baseline; do not inspect the live profiled-fragment checkpoints.

| Fixed model | Total training MSE | Persistent curve-mean MSE | Temporal MSE | Persistent share |
|---|---:|---:|---:|---:|
| Constant condition scale |1.475509|1.150526|.324983|77.97%|
| Joint-offset analytical,200 draws |1.396931|1.038436|.358496|74.34%|
| Profiled-offset analytical,same200 draws |1.355700|1.042360|.313340|76.89%|
| Joint-offset fragment,200 draws |1.351619|1.023909|.327710|75.75%|
| Earlier exact,2,000 full batches |1.296025|.986178|.309847|76.09%|
| Earlier hybrid,2,000 full batches |1.297461|.986934|.310527|76.07%|

- **Finding:** Exact condition-offset solving improves temporal MSE by .045156 while persistent MSE worsens .003925. The remaining persistent share is71.77–81.06% across all six training constructs, not one exceptional parent. Older long exact/hybrid fits retain the same broad pattern. This locates residual error but does not identify its cause or prove a noise ceiling.
- **Verification/resources:** Independent NumPy grouping from every original input prediction verifies138,324 curve records, all source identities/targets/weights, per-condition/construct reductions and fractions; max component error8.88e-16. Nonuniform toy checks the identity and rejects unweighted-mean substitution. Producer8.56s/0.968GiB; verifier6.40s. Local CPU, one-thread numerical settings, no Modal/GPU/environment change. Producer remains within120s/4GiB. Original arrays need no separate archive; compressed per-curve CSV and full source prediction pointers are retained.
- **Decision:** Finish the live paired biological fit unchanged, then target persistent cross-curve signal error as well as temporal kinetics. Do not grow a sampler merely because it is numerically qualified or add per-peptide offsets that erase abundance information. No model promotion, calibrated detector response or APC concentration conclusion. Read-only native bulk source review confirms an already tracked peptide-intensity omission in the inspected bundled CSV: Hitlist95open, with existing scale/lineage/missingness requirements; no duplicate issue. Known-load support762 and precursor757 remain open. Full availability goal remains active.


## 2026-10-23_2358 — Wider cleavage context improves digestion prediction; sampled hybrid adds no gain

- **Agent/model:** Codex / GPT-6; source010be23818400d47a55d6ce0f386b95c222ce8ec, recorded dirty state. [Family and frozen prospective specification](2026-10-23_2358_codex_sequence-context/README.md). Prior-head CI40056708162 passes docs, lint, tests and distribution checks.
- **Source/curation:** Unchanged canonical Hitlist digestion manifest5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615; purified human standard20S MSV000092631/PXD044463. Same90,630training observations from six constructs/seven conditions and14,822development observations from three parents; positive corrected log10MS signals and original parent/condition/curve/time weights. No new negatives, source importer, exclusions, synthetic labels or censor reinterpretation. Original test parents excluded; already inspected development is not a blind test. Full train/development predictions retained, no applicable nM/binder metrics.
- **Hypothesis/models:** 2x2 factorial of original adjacent-residue rates versus six-residue bond context and original instrument/length/composition response versus added peptide first/last residues. Original40rate/38response coordinates remain; training-only residualized/eigen-supported additions allocate80rate and72response coefficients. Independent-cut expected copies across all1..Lfragments plus response and one exact condition scale predicts MS signal. No peptide-identity offset. All source forms/contributors remain ambiguous as previously documented. No APC KO/species/transport/clearance, MHC or TCR supervision; no independent detector calibration.
- **Training:** Same original ratio-trained40rate initializer, zero additions/response,2,000full-batch Adam updates per arm at.03/.9/.999/1e-8. Weighted profiled MSE plus.005squared coefficients except rate intercept; final checkpoint only. No new pretraining, dropout, clipping, seed/penalty search or development selection. The prospectively named fifth arm uses both additions with K512conditional complete-partition sampled kinetic gradients, seed42,90%gradient-norm/10%uniform proposal, exact probabilities/residuals/conditional scores/response gradients. It is a hybrid, not simulator-only RL or a compute-saving claim.

| Model | Parameters | Train RMSE | Development shape RMSE | Development Pearson / Spearman | Development ratio RMSE |
|---|---:|---:|---:|---:|---:|
| base |78|1.13843|1.26510|0.2561 / 0.2452|1.16391|
| rate_context |158|1.06433|1.22853|0.3555 / 0.3487|1.09848|
| response_termini |150|1.12157|1.27193|0.2538 / 0.2442|1.16216|
| both |230|1.04616|1.24659|0.3467 / 0.3426|1.09710|
| both_hybrid |230|1.04802|1.24860|0.3456 / 0.3401|1.10118|

- **Biological interpretation:** Wider cleavage context improves development shape and temporal error; terminal response additions improve training error but worsen development shape error. Wider rates alone improve all three development-parent RMSEs, with most gain in one. This does not yet qualify general cellular concentrations or establish a causal response/processing separation. Constant shapeRMSE1.28060; old exact1.26510/hybrid1.26279; new base reproduces old exact.
- **Qualification/runtime:** Independent feature construction, zero-extension/old-objective parity, two fixed NumPy objective/Jacobian states and enumerated conditional-mixture toy checks pass. K512mean relative gradient errors.03864/.03093 and RMS.15391/.13643 pass both predetermined gates. Prospective local caps600sexact/1800shybrid,4GiB each; CPUfloat64one thread, no Modal/GPU/environment change. Source-freeze invocation-path failure occurred before qualification/fitting and is preserved; corrected invocation changes no code/data/settings.
- **Final comparison:** Conditional sampled hybrid developmentRMSE1.248605 versusmatchedboth-exact1.246593, Spearman.340120 versus.342622; no accuracy advantage in this single seed. Wider-rate-only exact is the preferred representation hypothesis (1.228529,.348720), improving matched base RMSE about2.9% and constant about4.1%. Extra response features fit training better but generalize worse. Do not promote them based on extra flexibility. Wider rates improve AnnexinA1 RMSE1.035409→1.024307, H3 1.317240→1.316613, H4 1.412173→1.321052; most gain is H4. No independent study robustness or calibrated cellular concentration is established.
- **Verification:** All five fits and their independent verifiers complete. Every10,000Adam arithmetic updates,527,260source-linked final predictions, global/per-parent metrics and persistent/temporal residual components verify. Four fixed gradient states per arm replay; hybrid uses independent NumPy conditional score/importance arithmetic and preserved PyTorch RNG draws. Other gradients are not independently replayed. All78/158/150/230/230allocated coefficients materially train. New raw features independently reconstruct; source loader remains the unchanged pinned canonical consumer. Exact final check scope and errors are retained in each arm.
- **Resources:** Local CPU/float64, one thread. Limited fit/verifier overlap means these are descriptive end-to-end times, not isolated implementation benchmarks.

| Arm | Seconds | Peak GiB |
|---|---:|---:|
| base |31.82|1.463|
| rate_context |35.23|1.454|
| response_termini |52.27|1.346|
| both |51.46|1.351|
| both_hybrid |1104.58|1.466|

- **Artifacts/decision:** Full report, comparison plot, CSV, source/config snapshots, original failure/correction logs, source-linked train/development prediction dumps and byte-verified optimizer/checkpoint archives are retained. No original test evaluation, sampled retry or outcome-driven extension. [Next prospective transfer plan](agents/codex/plans/digestion-to-mapp-transfer.md) freezes the learned cleavage prior and tests a minimal input/removal bridge on existing cellular MAPP, with its observation limits explicit. Separate literature discovery filed [Hitlist788](https://github.com/pirl-unc/hitlist/issues/788) for native intracellular peptidome perturbation panels with censored ratios and cross-paper reuse; no such workbook is admitted or fitted yet. Full APC availability/concentration objective remains active.

## 2026-10-24_0043_codex_digestion-cellular-transfer — Frozen digestion rates tested on cellular captured-peptide composition

- **Agent/model:** Codex / GPT-6; source5b888cccf97796d205e519f6371c7556076cf4f2, registered dirty state. [Report, source snapshots and reproduction](./2026-10-24_0043_codex_digestion-cellular-transfer/README.md).
- **Source/curation:** PXD009092 human HEK293 proteasome-associated fragments; unchanged canonical Hitlist MAPPmin2 manifestdf78d02f8200809c659e43adfae29866c48b712122b740379106994622f4ac80.72 train parents / 144 examples / 918 measurements;15 historically inspected validation parents / 30 examples / 112 measurements. Original14testparents excluded; no new test metrics/dumps by prospective design. No source curator, new negatives/imputation/censor conversion or synthetic biological labels. Original source/run/parent/material limitations retained. No applicable nM/binder metrics.
- **Model/training:** Continuous precursor input plus independent cleavage/common removal, F_s=int C_s(t)exp(-delta*t)dt over all 1..Lspans. Occurrence-summed centered log10F predicts centered log10MS. Uniform rates versus frozen 40adjacent or120wider digestion coefficients; no transfer of MS-response terms. Only log(delta) learns, initialized log(.01),2,000 full-batch Adam .03/.9/.999/1e-8, equalparent/condition/candidateMSE, no penalty/clipping/selection/restarts/new pretraining. No treatment/KO/species/TCR/pMHC/transport branch trains. Uncalibratedcapture/MSresponse and imported time scale preclude concentration/half-life claims.

| Model | Train RMSE | Validation RMSE | Validation Pearson / Spearman | Prediction SD | Delta | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| Uniform observed composition | 0.35609 | 0.28603 | undefined / undefined | 0.00000 | undefined | 0.00 |
| Uniform cleavage | 0.41394 | 0.36972 | 0.4416 / 0.3723 | 0.43871 | 0.105202 | 138.46 |
| Frozen adjacent rates | 0.44012 | 0.39307 | 0.4067 / 0.3456 | 0.43028 | 0.167398 | 135.09 |
| Frozen wider rates | 0.57514 | 0.55627 | 0.1685 / 0.1258 | 0.57088 | 0.175558 | 131.31 |


- **Decision:** Neither frozen sequence prior improves both no-sequence controls on this development endpoint. These are already inspected validation parents, not a newly blind result. No APC concentration model is promoted. All physical fits have varying outputs; wider-context validation prediction SD exceeds the target SD. The failure is not constant-output collapse. These numbers use validation, not the older14parenttestcontract. ForwardKL(target||prediction) is secondary; older reverseKL-trained models are contextual, not matched objective controls. All train/validation predictions and global/per-parent/exposure metrics retained.
- **Sampled comparison:** Conditional-age complete-partition estimator passes enumerated toy and16 K512 groups at 3 fixed states (meanrelative<=.00118,RMS<=.00639). It retains exact forward/conditional scores and is hybrid, not simulator-onlyRL. Original target-independent projection8598s exceeds7200 s ceiling; sampled training is not launched. This family supplies no fitted RL accuracy comparison. No prospective budget extension.
- **Exposure/verification:** No digestion-training construct/peptide overlap or unsupported algebraic bond directions in either MAPPsplit; prior digestion development overlaps separately reported. No homology/specimenindependenceclaim. Independent feature-transform/source/physical/metric checks verify4120predictionrows,6000Adamupdates and4fixedgradients perfit. Every allocated scalar receives updates; frozen imported sequence coefficients intentionally remain frozen. Allsmall-parentresolvent/integration/finite-difference/massconservation checks pass.
- **Resources/reproducibility:** Local CPU float64, one thread, noModal/GPU/environmentchanges. Exactcaps690/517/525s,4GiB; raw optimizer/checkpointJSON,predictionGZ,source/configreceipts directly stored. Initial NumPy boolserialization failure preserved; same seed/settings resume with ORIGINALcostcaps. No biologicalfitretry. Correctedqualification61.58s,1.08GiB. Runtimes include artifacts and limited verifieroverlap, not isolatedbenchmarks. Independentcalibration, precursor/clearance, nativecellularsource788 and APCvalidation remain open; fullgoalactive.

## 2026-10-24_0106_codex_cellular-amplitude — Calibration helps the non-sequence control; stop the MAPP transfer bridge

- **Date / agent:** 2026-10-24T01:06:11.535884-04:00; Codex / GPT-6; source `ebb70e3f4a86fd6ba33f9e2422dd0df63d56479b`, recorded dirty state. [Report and frozen reproduction bundle](2026-10-24_0106_codex_cellular-amplitude/README.md).
- **Source / curation:** Unchanged canonical MAPPmin2, PXD009092 human HEK293 captured proteasome fragments; manifest `df78d02f8200809c659e43adfae29866c48b712122b740379106994622f4ac80`. Verified 0043 predictions: 72 train parents/144 examples/918 rows, 15 historically inspected validation parents/30 examples/112 rows. Original 14 test parents explicitly excluded: no new test metrics/dumps. No new curator, filtering, negatives, synthetic labels, censor reinterpretation or source independence claim. No applicable affinity/binder metrics.
- **Training / outputs:** All physical rates and prior removal parameters frozen. Fit one nonnegative scalar beta per physical arm by exact training-weighted least squares, no optimizer/pretraining/penalty/search or validation calibration. `beta * centered log10 residence amount` maps to original centered log10 captured-MS intensity. Three new scalar parameters; no physical/APC/KO/treatment/species/MHC/TCR/transport training. Original unscaled and uniform composition controls retained; no sampled fit in this analytic diagnostic.

| Model | Beta | Train RMSE | Validation RMSE | Validation Pearson / Spearman | Validation prediction SD |
|---|---:|---:|---:|---:|---:|
| Uniform composition | 1.000000 | 0.35609 | 0.28603 | undefined | 0.00000 |
| Uniform cleavage | 1.000000 | 0.41394 | 0.36972 | 0.4416 / 0.3723 | 0.43871 |
| Uniform cleavage + calibration | 0.302123 | 0.34137 | 0.26711 | 0.4416 / 0.3723 | 0.13255 |
| Adjacent sequence rates | 1.000000 | 0.44012 | 0.39307 | 0.4067 / 0.3456 | 0.43028 |
| Adjacent sequence rates + calibration | 0.265903 | 0.34161 | 0.27590 | 0.4067 / 0.3456 | 0.11441 |
| Wider sequence rates | 1.000000 | 0.57514 | 0.55627 | 0.1685 / 0.1258 | 0.57088 |
| Wider sequence rates + calibration | 0.113527 | 0.35129 | 0.28697 | 0.1685 / 0.1258 | 0.06481 |


- **Decision:** The calibrated uniform-cleavage control wins. Neither imported sequence prior beats both controls, so the prospective stop rule is triggered: stop extending this bridge on the same MAPP development set. Uniform calibration improves validation RMSE about 6.6% versus uniform composition and helps 10/15 parents (4 worse,1 tie); this is modest geometry-based captured-signal prediction. Adjacent loses to calibrated uniform on 9 parents, wider on 11. Positive beta preserves ranking; no proof of detector calibration, physical abundance or APC concentrations. Validation selection history remains explicit. Keep the best control provisional for this development contract only, not the older test contract.
- **Metrics:** Hierarchical-weighted MSE/RMSE/R2, pooled Pearson/Spearman/SD, forward KL(target||prediction), per-parent/exposure strata and all train/validation predictions retained. Calibrated uniform validation R2 .12793 and predictionSD .13255 versus targetSD .35130. No new untouched test.
- **Verification / runtime:** Independently reconstruct 4,120 input source/physical predictions using the pinned prior reference; weighted least-squares/KKT verify all3 beta values; all7,210 output rows, metrics/strata and stop decision replay (max prediction error1.78e-15). Producer 0.935s/104.14MiB, verifier 9.196s; declared60s/2GiB. Local CPU one-thread, no Modal/GPU/environment change or failed-run retry. Source/config receipts, coefficients/cross moments, plot/CSV and compressed predictions directly stored.
- **Next:** Enforce the stop rule and prioritize canonical native cellular source788, independent response95/762 and precursor/machinery evidence. Full APC availability/concentration goal remains active; no further tuning of this bridge on the same MAPP development set.

## 2026-10-24_0117_codex_native-peptidome-admission — Native source replay succeeds; semantic errors defer training

- **Agent/model/date:** Codex / GPT-6; October24,2026. Registered before retrieval/schema arithmetic at132404d1d0a2b714dcff9804e7e052b909f28f6d with recorded dirty task state. [Report, complete cell evidence and reproduction](2026-10-24_0117_codex_native-peptidome-admission/README.md).
- **Exact source/curation:** Public Dasgupta2014/Gelman2013 native human HEK293T/SH-SY5Y extract peptidomics. Thirteen PLOS assets retrieved; six OOXML workbooks audited, including derived heatmap/dose/time tables. Original2012 epoxomicin XML retrieval failsHTTP500; primary-source lineage remains unresolved. No author outreach. Original cells/qualifiers/unknown peaks/annotations remain unchanged; no predictive recipe admitted.
- **Shared implementation:** Hitlist790 merged during audit. Its pinnedf4d3a6d public writer/EvidenceStore produces a schema15/catalog20 package, code2.0.15; PyPI publication not verified here. No new source curator or shared environment changes. Exact canonical manifestb15de0b80c9fff1aad5b60f66e4e5b083c7f585ea01e723209a482182e26c32b; input paths/hashes, execution receipts and source snapshots preserved.
- **Training/assays/splits:** No model, optimizer, pretraining, synthetic biological labels, loss weights or assay-head mapping. Relative isotope-channel ratios are observed endpoints, not concentration or MHC labels. No validation/test split/predictions or regression/binder metrics because this is source qualification before independence/identity admission. No winner/checkpoint; MAPP transfer stop remains in force.

| Audit condition | Outcome |
|---|---|
| Independent XML versus openpyxl |271,778explicit cells across6workbooks;0disagreements|
| Canonical source-row replay |10,564rows;328,880values including blanks, all agree|
| Canonical query and independent quantity parsing |61,908slots, all source values/statuses/operators replay|
| Candidate cross-paper reuse |830fingerprint groups/1,660rows versus0canonical matches;552groups also share normalized experiment labels,278need label adjudication|
| Identification eligibility |16,807numeric exact slots have no sequence;213have explicitly tentative outer parentheses|
| Projected heatmap identity |173Gelman rows use adjacent-residue column instead of peptide column|

- **Quantity policy:**34,763reported numeric exact slots;15,753bounds;4,113empty;7,217nd;60po;2spreadsheet errors. Three numeric values are nonpositive. These are measurement slots, including controls/derived averages, not independent targets. Parenthesis-containing annotations have219exact slots including one uncertain-modification form;540slots containMoxwhose modification is lost by the normalized parser.
- **Findings/decision:** Source bytes replay, but semantic admission fails. Reuse hash includespaperID; parenthesized acetylation can add a residue; modifications and tentative evidence are lost; Gelman heatmap column is wrong; acquisition assignsG1universally despite2014G2Methods. Treatment/harvest intervals need source scope:2014keeps inhibitor during washing, so35min total remains supported. File and read back [Hitlist791](https://github.com/pirl-unc/hitlist/issues/791); retain corrected-source and canonical-recipe work before fitting. No APC concentration claim.
- **Resources/reproduction:** Local CPU, one-thread environment, no Modal/GPU. Workbook audit14.64s/332.33MiB; canonical build/query/replay14.60s/997.98MiB; semantic verification1.02s. Allcaps pass; no failed audit execution/retry. All raw cell/layout/candidate evidence, receipts and source snapshots retained; original assets/package referenced underdata/native_peptidome_source_audit. Python explicitly linted outside normal experiment exclusions. Presto132404dCI docs/lint/tests/distribution jobs pass.
- **Next:** Correct shared semantics under791, qualify a native-response recipe and preregister the bounded biological exact-versus-sampled comparison. The full APC availability/concentration goal remains active.


## 2026-10-24_0156_codex_tmab-semantic-replay — Corrected native peptide semantics preserve sources; recipe still pending

- **Agent/model/source:** Codex / GPT-6; Presto b298ea9 with recorded task dirt, Hitlist f4d3a6d plus frozen correction diff/files. [Full report and reproduction](2026-10-24_0156_codex_tmab-semantic-replay/README.md). Execution version 2.0.16, revision v2/schema16/catalog21; delivery status separate.
- **Dataset:** Same four pinned public Dasgupta2014/Gelman2013 primary/heatmap workbooks as0117. Native human HEK293T/SH-SY5Y extract isotope-channel ratios, not APC concentrations or MHC measurements. Old manifestb15de0b8…; corrected manifest `f84bd8e738a2c14d446956c719cfeaba91bc867a95aebd62dac1c91d3e3a1666`. Source quantities/qualifiers/missingness/overlap/errors, row arrays and measurement IDs unchanged. No new curator or environment changes.
- **Conditions/results:** Shared public writer/EvidenceStore versus original package and independent source grammar/memberships:10,564rows/61,908measurement slots preserved;830possible cross-paper row groups/1,660rows,552with matching normalized experiment labels and278different. Structured modifications, tentative/unresolved identification,173Gelman heatmap projections, acquisition scope and source timeline examples verify. Source candidates:7,933exact and4,662censored treated/control ratios. Controls/derived means remain deferred; these are not independent targets.
- **Failure/correction:** Original execution failed1,242context-label comparisons because its verifier ignored existing whitespace trimming in projected labels. Raw rows retain original bytes/values. All other assertions passed. Separate frozen closure verifies every discrepancy against both versions and original cells, with0unexplained differences; initial code/log/results/package remain immutable. No production change or outcome-driven biological retry.
- **Training/evaluation:** No model/pretraining/optimizer/loss/assay head/synthetic biological labels. No predictive validation/test split or prediction dumps because this is source admission. Full semantic-change/query/membership dumps are retained; nM/binder metrics inapplicable. No predictive gain or concentration qualification.
- **Resources/verification:** Local CPU one thread, no Modal/GPU. Original replay38.99s/1152.34MiB; closure2.49s/554.45MiB, within registered caps.43focused tests pass;14fail/1pass on pinned old code. Full ./test.sh passes:3,403tests/723.92s; hosted/release gates remain pending. Exact commands/env/source snapshots and old/new manifests in reproduce/results.
- **Decision:** Corrected source semantics pass; native-response recipe still pending shared-control/charge/reuse grouping and parent mapping. [Hitlist791](https://github.com/pirl-unc/hitlist/issues/791) and implementation792 remain upstream owners. No more MAPP bridge tuning on the inspected development set; proceed toward real native peptide perturbation prediction. Full APC availability/concentration goal stays active.

## 2026-10-24_0259_codex_native-response-admission — Canonical source, parent alternatives and grouped response targets qualify

- **Agent/model/date:** Codex / GPT-6, October24,2026; [report and frozen reproduction](2026-10-24_0259_codex_native-response-admission/README.md). Presto593e3b5 with recorded task dirt; Hitlist frozen merge of ddc6581/7649ffe, now committed/pushed as b3f1079 in PR793. Sourcev3/schema17/catalog22/version2.0.17; release not yet verified. The original0156v2 replay is not evidence for concurrentPR794's differentv2 implementation.
- **Source/curation:** Same four public Dasgupta2014/Gelman2013 workbooks, native HEK293T/SH-SY5Y peptide extracts. All10,564rows/61,908slots and1,736,376original-field invariants preserved. Possible reuse includes830identified and553unidentified groups; no specimen identity invented. Modern UniProt2026_03human reviewed isoform reference42,562records, all1,155exact occurrences/345used records retained; no original-search or unique-origin claim.
- **Recipe:** Positive reported supported-identity exact/bounded treated/control ratios; defer controls/means/unknown/tentative/unsupported/nonquantifiable/conflicting copies. No generated labels, negatives, MHC length filter or quantity averaging. Components use all source sequences, possible reuse, parent/gene assertions and all reference alternatives before filtering. Weights average condition/component/form/observation. Source manifest7f80844b…, recipe manifestc72980c0…; complete hashes/paths in report.

| Split | Groups | Components with targets | Effective observations | Exact | Bounded | Full-parent-supported groups |
|---|---:|---:|---:|---:|---:|---:|
| Train |1,861|86|6,735|4,301|2,434|1,855|
| Validation |245|19|932|504|428|245|
| Test |631|16|2,447|1,278|1,169|631|

- **Verification/resources:** Public materialization54.87s/1.415GiB; independent grammar/Aho–Corasick/BFS/target/weight replay10.59s/1.069GiB. All source, reference, component, disposition and observation checks pass. Initial independent verifier incorrectly required a space after taxonomy; valid end-of-header token correction is preserved with original failure, without source/split/gate changes. Hitlist format/lint and3,414normal local tests pass630.18s. Local CPU/one thread; no Modal/GPU/environment changes. Frozen exact invocations/source/files/results retained.
- **Training/evaluation/decision:** No model, pretraining, loss optimization or predictive validation/test evaluation in admission; no prediction dumps or nM/binder metrics applicable. The prospective split is retained for0314biological fitting. Thirty-three control proxies cross splits; specimen/denominator/homology/external-study independence unresolved. Native relative ratios, drugs and source parent alternatives do not identify APC concentrations, KO activity or unique precursor origin. Admission closes; full goal remains active.

## 2026-10-24_0314_codex_native-response-baselines — Real native response prediction: condition effects transfer, direct sequence models overfit

- **Agent/model/source:** Codex / GPT-6, October24,2026; Presto593e3b5 with recorded dirt. [Full report, plot, fitted models, predictions and frozen plan](2026-10-24_0314_codex_native-response-baselines/README.md). Canonical native response recipec72980c0… after independent0259qualification; no Presto-only source curator.
- **Exact data/assays:** Native HEK293T/SH-SY5Y chemical treated/control isotope-channel ratios. Train1,861groups/86components/28conditions/6,735effective observations; validation245groups/19components/27conditions/932observations. Train4,301exact/2,434bounded; validation504exact/428bounded. Source qualifiers, contributors and condition/component/form/observation weights unchanged. No target clipping/averaging, generated negatives, synthetic biological labels or affinity-length filter. Other affinity/elution/TCR/APC assays excluded. No500nMmetric applies.
- **Training/features:** Six prospectively fixed arms: zero, weighted exact training mean, exact/censored optimal constant, condition-only, shared sequence, compound-dependent sequence. Sequence=loglength/20AAfractions/first+last residues/acetylation/oxidation; literal compound interactions only. Training-only scaling and rank-support projection;28/83/582allocated linear parameters all change, with zero unseen validation condition/compound categories. No pretraining. Weighted exact squared error plus one-sided bound violation; .01coefficient penalty except globalintercept. Zero-init float64L-BFGS-B maxiter1000/gtol1e-9/ftol1e-12, final checkpoint only, no tuning/restarts. Iterations10/37/215; all converge by stated solver rules, latter two by objective-change tolerance.

| Model | Parameters | Train loss | Validation loss | Validation exact RMSE | Validation Pearson / Spearman | Weighted bound satisfaction |
|---|---:|---:|---:|---:|---:|---:|
| No change |0|.225013|.186297|.257019|undefined|0%|
| Exact train mean |1|.235781|.208923|.254223|undefined|0%|
| Censored constant |1|.224752|.183884|.260793|undefined|0%|
| Condition only |28|.130494|**.073495**|**.247726**|.4518 / .4279|0%|
| Shared sequence |83|.088709|.129189|.372097|.2833 / .2297|22.45%|
| Compound-dependent sequence |582|**.045652**|.153178|.364586|.2980 / .2881|45.88%|

- **Decision:** Condition-only is the provisional reference, reducing validation primary loss60.5% versus nochange but exactRMSE only3.6%; it still satisfies zero bounds. Sequence variants worsen primary loss75.8%/108.4% versus condition-only despite training gains. Shared sequence loses on21/27conditions and11/19components; compound sequence on20/27conditions and12/19components. Neither passes the biological screen; no proof that sequence is irrelevant or unique root cause. Do not extend this direct feature sweep on the same validation data. Next is the small full-parent mechanistic model with matched differentiable/sampled gradients, frozen in [native-processing-comparison.md](agents/codex/plans/native-processing-comparison.md).
- **Test policy:**631testgroups remain reserved for the final predeclared physical-model comparison. No test metrics/predictions here by design, no validation selection. Full train/validation per-observation dumps permit metric recomputation. These are seen-treatment, held-out-component results, with shared studies/control proxies and unresolved specimen/homology independence; not APC concentration validation.
- **Verification/resources:** Six arms4.639s/418,185,216bytes, local CPU/one numerical thread/noModal/GPU/envchanges. Independent checks verify46,002source-linked predictions, training transforms/support/final objectives/gradients,12global and960stratified metrics. Exact input tie correction changes149correlation records, maxpredictionchange3.28e-15; losses/RMSE/decisions unchanged. Original outputs retained; `results/qualified-v2` are canonical reported predictions. Scalar verifier completed mathematical checks but failed180s budget; vectorizing only coordinate products passes6.956s/458,981,376bytes and agrees with saved scalar reference1.17e-15. No fit retry or changed gate. Numerical toy and finite-difference directions pass; optimizer history is not independently replayed. Full APC availability/concentration goal remains active.


## 2026-10-24_0410_codex_native-processing — Small real held-out processing gains; sampled gradients add cost

- **Date / agent / source:** October24,2026; Codex / GPT-6. Source2cb7db0fff076180e3235ea11ef9d30b6cb2872e with recorded dirt and frozen new module/launcher. [Full report, plot, frozen plan and artifacts](2026-10-24_0410_codex_native-processing/README.md).
- **Source and curation:** Canonical Hitlist native-response manifestc72980c070cb1296609bbe7d296e27b6a2a22d3ff354ef846c1c99ad7959ccfc, independently admitted0259. Dasgupta2014/Gelman2013 HEK293T/SH-SY5Y native extract chemical treated/control isotope ratios. Full-parent-supported subset: train1,855groups/85components/6,707observations(4,289exact/2,418bounds), validation245/19/932(504/428), test631/16/2,447(1,278/1,169). Six unsupported train groups/28observations omitted, all source IDs retained. Original hierarchical weights renormalized once per selected split. No synthetic labels, negatives, imputation, target averaging, cutoff or chosen parent. All six0314controls refit on the matched subset.
- **Model and output:** Continuous unit parent input, independent cuts and common removal delta1; all contiguous1..Lproducts implicitly supported. Uniform rates versus38supported adjacent-residue coordinates, plus globalrate and28condition rate/28condition offset coordinates:57/95parameters. Train-only supported feature transform over5,008reachable bonds/236parents. Keep all source origin alternatives, collapse identical parent sequences, sum occurrences and average candidate-parent inputs. Predict b_condition+log10(Ftreated/Fcontrol), with a shared treatment rate multiplier; no peptide-specific offset. Distant outside bonds integrate out. Form labels retained without a PTM rate model. This is native response, not APC concentrations.
- **Training:** Four fixed arms, zero init,200full-batch float64Adamsteps,lr.02/betas.9,.999/eps1e-8, penalty.01 exceptbglobalintercept. Exact squared log-ratio error plus one-sided squared bound violations; no pretraining/selection/restarts/synthetic data. No binding/stability/elution/TCR/APC machinery or species auxiliary supervision. Direct controls use unchanged0314ridge/SVD/L-BFGS-B contract. Every57/95coordinate changes. All fits terminate before once-only test evaluation; all settings/checkpoints/constants/transforms sealed. Test is now inspected and may not be tuned against.
- **Sampling:** K512conditional origin/age draws per response group,seed42, paired treatment/control uniforms. Exact forward amounts/residuals/origin probabilities/offset gradients; conditional unnormalized event-score kinetic gradients. Hybrid, not simulator-only RL/GRPO. No clipping/group-standardization/reuse/KL. Four fixed probes pass meanrelative.00279–.00321/RMS.00912–.01142 versus.10/.25gates; zero shared-rate cancellation and finite differences pass.

| Model | Train loss | Validation loss | Test loss | Test exact RMSE | Test Spearman |
|---|---:|---:|---:|---:|---:|
| No change |0.225345|0.186297|0.202817|0.255458|undefined|
| Condition only |0.130686|0.073514|0.130557|0.250953|0.5324|
| Direct shared sequence |0.088821|0.129940|0.226896|0.331508|0.4473|
| Direct compound × sequence |0.045643|0.152802|0.372538|0.499235|0.2226|
| Uniform cleavage / exact |0.129261|0.072834|0.130125|0.249390|0.5280|
| Uniform cleavage / sampled |0.129287|0.073126|0.129744|0.249202|0.5345|
| Sequence cleavage / exact |0.124510|0.070384|0.129789|0.243207|0.5373|
| Sequence cleavage / sampled |0.124566|0.070619|0.129336|0.242887|0.5383|

- **Decision:** Exact sequence processing improves condition-only primary loss4.26%validation/0.59%test, exactRMSE0.71%/3.09%. TestPearson/Spearman.5578/.5373; weighted exactR2.0929 versuscondition.0342. ValidationSpearman declines.4279→.4221. Wins13/19validation and10/16testcomponents, only14/27 and15/28conditions. Testbound satisfaction14.92%, validation1.30%; most bounded extremes remain unsatisfied. Direct sequence regressions overfit again. Nominal primary-loss screen passes, but incremental signal is small and unreplicated. Exact is the practical provisional processing reference; no source-independent or concentration qualification.
- **Exact versus sampled:** Sequence exacttestRMSE.243207 versushybrid.242887, primaryloss.129789 versus.129336; validation slightly favors exact. Single-seed near-equivalence supplies no convincing sampling advantage. Uniform exact5.40s/hybrid146.61s; sequence exact5.55s/hybrid138.74s. Each hybrid uses379,904,000conditional event scores; all prescribed200steps complete under300s/1800s and4GiB. PeakRSS.412–.659GiB. Exact finalgradientinf8.36e-6/3.46e-5; sampled.00809/.00734. Bounded fits, not a global convergence proof. Local CPU, one requested numerical thread; no Modal/GPU/environment changes.
- **Verification and artifacts:** Nine physical/substrate tests pass (3.39s); original float32 test-reference failure retained and corrected without model/gate changes. Independent scalar physical reconstruction, source lineage/qualifiers/weights and global/stratified metric closure verify100,860predictionrows/2,180metricrecords; maxprediction/logamount error3.55e-15,86.08s/.277GiB. All checkpoints/trajectories/gradients/conditional probes, source snapshots, configs, train/validation/test predictions, tables/plot and receipts stored. Optimizer trajectories retained but not independently replayed. nM/500nMmetrics do not apply.
- **Limits and next:** Same studies/seen treatment categories; unresolved specimen, shared-control and homology independence. Chemical labels are not measured enzyme activities or KO states; same-form detector cancellation and equal candidate-parent input remain assumptions. Actual APC data, precursor-relative calibration, causal machinery effects, transport and independent MS response remain open. Do not tune this inspected test or extend sampling machinery on numerical success alone. Full availability goal remains active.

## 2026-10-24_0449_codex_apc-precursor-response — Actual APC ligand response: processing fails to beat context

- **Date / agent / source:** October 24, 2026; Codex / GPT-6. Presto f7b76a30fbbffce6f2d36752ee16d85b6b3fa1a6 with recorded task dirt and frozen new experiment code. [Full report, predictions, checkpoints and reproduction](2026-10-24_0449_codex_apc-precursor-response/README.md). Local CPU, one numerical thread, no Modal/GPU or environment changes.
- **Dataset/curation:** PXD057588 Table9 macrophage MHC-I treated/control log2 fold changes, explicitly converted to log10 ratio; Table11 digest peptides supply condition-level precursor surrogates. Current Hitlist2.0.17 consumer b3f1079 against verified schema7/2.0.0 source package manifest d99f67e6…; canonical response recipe93ef0b23…. Source quantities/derivations/context/all origin alternatives retained. Require ≥2 distinct digest sequences for every candidate parent; mean per-sequence channel log-ratios then median across digest sequences. Not exact aliquot matching:8/9channel mappings differ between tables. Digest normalization differs from MHC; Table10KO is tumor-partner, not macrophage. No source curator added to Presto.
- **Split/weights:** All-source canonical dependency components hashed with fixed salt before filtering. Train492peptide rows/984groups/361components/2,952observations; val96/192/69/576; test129/258/82/774. All717peptides have CT2A and GL261 conditions;920candidate parent sequences/1,840parent-condition surrogates. Weights average condition/component/peptide/channel. All admitted targets exact, no imputation/clipping/target averaging/negatives/synthetic biological labels. Other assay families excluded; nM/binder metrics inapplicable. Shared study/conditions/controls and unresolved specimen/homology independence remain.
- **Model/training:** Continuous unit parent input, independent bond cleavage, common removal1; all1..Lspans implicit. Query sequence/boundaries and parent-specific precursor ratio produce conditionoffset+log10(sum treated amounts/sum control amounts). Uniform,38supported adjacent-residue coordinates, or frozen0410human-native rate prior. Train-only scaling/SVD/equal-parent weighting;2condition coordinates. Four fixed controls use zero or precursor-only prediction, or2parameter L-BFGS-B condition offsets. Five physical arms use200zero-init full-batch float64Adamsteps,lr.02/betas.9,.999/eps1e-8,seed42. Weighted MSE+.01coefficient penalty except globaloffset. All5/43/4allocated coordinates change. Prior rates frozen, only condition scales/offsets fitted; zero exact peptide overlap with priortraining, no held-out exact full-parent overlap. No affinity/stability/TCR/KO/species or independent detector module is trained.
- **Sampling contract:** K512conditional origin/age scores with precursor-weighted exact origin probabilities, paired treatment/control uniforms, exact forward/residual/offset gradients. Hybrid, not simulator-only RL/GRPO. No clipping/std/reuse/KL. Four fixed probes pass meanrelative.000716–.001628/RMS.004194–.007620 against.10/.25; finite differences and no-input cancellation pass. No biological fit retry or held-out selection; all attempts sealed before once-only evaluation.

| Model | Validation MSE | Test MSE | Validation RMSE | Test RMSE | Test Spearman |
|---|---:|---:|---:|---:|---:|
| No change |.017453|.014547|.132111|.120611|undefined|
| Condition only |.017035|.014403|.130517|.120011|.2512|
| Precursor only |.026993|.019088|.164295|.138158|.2241|
| Condition + precursor |.027396|.019394|.165518|.139264|.2642|
| Uniform processing + precursor |.027390|.019398|.165498|.139278|.2639|
| Sequence processing + precursor |.027389|.019398|.165496|.139278|.2641|
| Sampled sequence + precursor |.027398|.019404|.165524|.139298|.2643|
| Sequence without precursor |.017032|.014402|.130508|.120008|.2323|
| Frozen native prior + precursor |.027407|.019375|.165551|.139193|.2646|

- **Decision / preferred reference:** Negative. Keep condition-only for this contract. No-precursor sequence testMSE gain only.0052%, weightedR2.00136 versus.00131. Its within-condition testpredictionSD.000278/.0000759 versus target.1372/.0972 is nearly constant. Precursor input produces variation but worsens error; processing does not rescue it. Native prior gives no useful transfer. Stronger ranking with precursor does not offset its substantially worse MSE. This does not identify whether normalization, noisy precursor surrogates, fixed proportional scaling, regularization, restricted kinetics or changing selection/measurement dominates. Test now inspected; no tuning against it.
- **Runtime:** Exactuniform2.58s,sequence1.92s,noinput1.90s,prior1.34s; hybrid67.48s (~35×sequenceexact),201,523,200event scores, no gain. All200steps/300s-exact/1800s-sampled/4GiB caps pass;physical peak.416–.540GiB. Fourcontrols each<.002s aftersetup. Exactfinalgradientinf1.4–1.9e-6 versus hybrid.000591; bounded local stationarity, not global convergence proof. Numerical qualification16.73s.
- **Verification / artifacts:** Source closure1,256,555relationrows/200,718responses/1,840precursors/4,302admittedobservations passes13.23s. Separate scalar physical reconstruction and independent NumPy/SciPy metrics verify38,718predictionrows/4,743metricrecords,maxerror3.55e-15,7.21s/.319GiB. No optimizer replay or external biological validation. Original export/nullable-reference consumer failures retained and corrected before fitting; no reachable upstream sequence omission. Frozen launch/config/source/prior/dependency receipts, all checkpoints/trajectories/predictions, summaries/CSV/PNG/PDF and overlap/spread diagnostics preserved. Figure layout correction does not change numbers.
- **Next:** Address precursor calibration and kinetic identifiability on training data, then require a fresh prospective biological population for further predictive claims. Independent machinery/selection/detection and precursor-relative concentrations remain open. Exact and sampled agree on this failure; do not extend sampler machinery as its remedy. Full availability goal stays active.

## 2026-10-24_0527_codex_apc-training-diagnostic — Nonconstant APC fits overfit; fixed precursor scaling is unsupported

- **Date / agent / source:** October24,2026; Codex / GPT-6. Presto34568ca with recorded task dirt and frozen new code. [Report, predictions, checkpoints and reproduction](2026-10-24_0527_codex_apc-training-diagnostic/README.md). Registered after0449failure, before diagnostic arithmetic. Local CPU/one numerical thread; no Modal/GPU/shared-environment changes.
- **Exact source/curation:** Only original0449training subset of canonical PXD057588 recipe93ef0b23…:492peptides,984condition groups,361components,2,952exact source-channel log10responses. Table9MHC-bound targets, Table11condition-level parent surrogates; all original values/weights/origin alternatives/contributors preserved. Same sample-pairing/normalization limitations. No synthetic labels, imputation/clipping/negative generation; other assay families excluded. Original sourceJSON is filtered before modeling; originalvalidation/test outcomes are not fitted/predicted/scored.
- **Evaluation contract:** Posthoc training/development diagnostic. Fixed sha256component fivefold innerCV with86/100/125/92/89validationpeptides; feature transforms and calibration refitted within each fold. Original hierarchical weights normalized inside fitting subsets and retained for pooledCV. Each arm also fits full originaltraining. No untouched outertest or nM/binder metrics; no external concentration claim. Every inner-validation observation is preserved once pervariant with source IDs.
- **Model/training:** Four analytic controls(nochange,condition,fixedprecursor,calibratedprecursor). Single nonnegative gamma from within-condition covariance/variance; two unpenalized condition offsets. Physical factorial shared versus condition-specific adjacent-residue treatment preferences × kinetic penalty.01/0. Fold-specific gamma frozen and applied to each parent's input exponent. 38train-supported residue coordinates;41/117kinetic parameters, all update; two offsets profiled exactly at each objective.400zero-init full-batch float64Adamsteps,lr.02/betas.9,.999/eps1e-8,seed42,finalcheckpoint. Weighted MSE+registeredkineticpenalty; no pretraining/selection/restarts. This different offset/calibration contract is not a controlled repeat of0449.

| Model | Full-training RMSE | Inner-CV RMSE | Inner-CV R2 | Inner-CV Spearman |
|---|---:|---:|---:|---:|
| No change |.116091|.116091|-.00168|undefined|
| Condition only |.114155|.114185|.03095|.1890|
| Fixed precursor coefficient1 |.136207|.136383|-.38246|.1568|
| Calibrated precursor |.114083|.114738|.02154|.2210|
| Shared preferences, penalty.01 |.114057|.114715|.02193|.2222|
| Shared preferences, no penalty |.105947|.116482|-.00844|.2039|
| Condition-specific, penalty.01 |.105242|.115730|.00454|.2577|
| Condition-specific, no penalty |.098916|.127096|-.20059|.2194|

- **Finding / decision:** Full-training gamma.05163,foldrange.00165–.14445; calibration removes most damage from forcing1 but does not beat condition-onlyCV. Initial condition-centered weighted Jacobian ranks5/41shared,81/117flexible atrelative1e-8; final full-data ranks41/41and117/117. This diagnoses local first-order sensitivity, not causal identification. Less penalty/more flexible rates fit training better but overfit components. Keep condition-only comparator; no production baseline promotion. Within-group channel variation MSE.002354 accounts for18%of conditiontrainingerror,82%between groupmeans; this is not a biological/noise decomposition.
- **RL decision:** Predesignated flexible/unpenalized exact arm must beat calibratedprecursor innerCVRMSE by≥2% with SD>.001bothconditions. Variationpasses(.0920/.0630),RMSEworsens10.77%; gatefails. No sampled fit or sampled numerical qualification ran; do not claim an exact-versus-RL result from this family. Conditional-score code retained, noGRPOclips/std/reuse/KL. Stop adding sampler complexity or expanding this source's sequence features.
- **Runtime / checks:**24physical400stepfits complete,sumrecordedfitloops/finaloutputs100.02s,individual3.50–5.25s,peak.467GiB. All300s/4GiBcaps pass. Sharedpenaltygradientinf<8e-13; flexibleunpenalized.0042–.0254,not convergenceproof. Fourfixed finite-difference probes/three directions each pass1e-7; qualification3.58s. Independent source/fold/scaling/calibration,scalarphysical/profiledoffset andNumPy/SciPyclosure verifies47,232predictions/5,824metricrecords,maxerror2.84e-14,28.70s/.861GiB. Optimizer/Jacobiantrajectories retained but not independently replayed. No failed biological run discarded.
- **Artifacts / next:** Full/foldcoefficients,transforms,calibration,loss/gradient/updatehistories,100stepoptimizer/RNGcheckpoints,source-linkedfulltrain/innerCVpredictions,metrics/CSV/PNG/PDF,numericalprobes,ranks and frozen code/config/source/commands retained. No originaloutertest dumps by explicit diagnostic design. [Hitlist798](https://github.com/pirl-unc/hitlist/issues/798) now requests canonical quantitative deVerteuil2010DC machinery-loss admission with replicate/control/concurrent-state and separateRNA/protein metadata; requestreadbackverified,sourceassetsnotadmitted. Need independent prospective biological population/measurementconstraints. Full APC availability/concentration goal active and unachieved.

## 2026-10-24_0602_codex_dc-machinery-admission — Original quantitative supplements not yet accessible

- **Date/agent:** October24,2026; Codex/GPT-6; Presto c7d538c with recorded task dirt. [Report and reproducibility bundle](2026-10-24_0602_codex_dc-machinery-admission/README.md). Hitlist fda1eff/2.0.18 inspected; unchanged source code.
- **Source contract:** deVerteuil2010 PMID20484733/PMC2938112. Quantitative admission remains pending:13 retrieval attempts,10 hashed HTTP bodies, no original quantitative asset/observation admitted. EuropePMC metadata hasSupplY/OAN; supplementary API returns explicit non-OA XML error despiteHTTP200. Current official PMC archive listing is empty/untruncated; legacy API retired. Publisher article/linkedPDF403; no access-control bypass or author messages.
- **Training/evaluation:** none. No pretraining, synthetic labels, loss, output mapping, split, validation/test prediction dumps, affinity metrics or selected model because actual source schema was not available. Published aggregate counts are not training rows. Full APC availability goal stays active.
- **Verification/runtime:** independent size/hash/receipt checks pass for every retained response, including metadata identity and explicit error-body checks. SumHTTPtime5.59s; local work, no GPU/Modal/environment change. PyPI2.0.17JSON separately returns404; release deployment still unverified. Exact commands/environment, original failures, source paths and source snapshots retained.
- **Decision:** preserve Hitlist798 access gap; require original quantities, qualifiers, source filtering, concurrent machinery state and uncertainty. Continue existingHitlist726 HAP1 quantitative panel as a machinery-conditioning dataset, not a professional-APC concentration validation. No practical biological baseline changes.

## 2026-10-24_0612_codex_hap1-quantitative-admission — Original quantities recovered; training admission pending

- **Date / agent / source:** October 24, 2026; Codex / GPT-6. Presto c7d538c with recorded task dirt; Hitlist fda1eff/2.0.18 inspected. [Report, source receipts and replay bundle](2026-10-24_0612_codex_hap1-quantitative-admission/README.md). Registration preceded retrieval; numerical replay hypotheses were written after header inventory and before comparison.
- **Dataset contract:** Shapiro 2025, PMID40113210/PMC12090245/PXD056426. Human HAP1 wildtype plus 11 machinery KO conditions; three declared biological replicates each. MHC-I elution and global proteomics, not direct free-peptide concentrations or professional APCs. Original XML, eight supplements, canonical Search.zip and SDRF retrieved; 167-file deposit inventory complete. Noncanonical processed archive and raw acquisitions only inventoried. No Presto-only curator, synthetic labels, zero imputation or reinterpretation of missing values.
- **Source arithmetic:** All 296,483 numeric canonical peptide cells reproduce normalized precursor sums averaged across all declared technical runs (maximum error 7.45e-9); raw quantities do not. All 166,028 numeric protein intensity cells reproduce MaxLFQ exactly. Field names alone would misidentify both quantities. Full per-cell comparisons and original source fields/identities remain in the bundle.
- **Open lineage:** Two of 69 DIA runs remain unmapped (40,529 rows). All 36 peptide NA cells with a deposited reconstruction have zero quantity. Protein NA cells include 42,608 deposited zeros and 23,096 positive MaxLFQ values; source-filter lineage is unresolved and these values are not silently filled. The 36 proteomics filenames have unique punctuation aliases, without proof of paired specimens. Preserve confidence fields: 2,059 mapped rows exceed .01 global Q although per-run/library Q meet .01.
- **Training/evaluation:** None. This source audit has no pretraining, model parameters, loss, assay-output mapping, train/validation/test split, predictions or binding/concentration metrics. It supplies no new predictive success or model promotion. Prior inspected biological tests remain closed.
- **Runtime / initial artifacts:** Replay 55.03s, peak RSS 0.635GiB; Search.zip retrieval 117.73s, SDRF .64s. Local CPU, no Modal/GPU/environment changes. Actual headers, sample workbooks, hashes, commands/env, original response bodies, frozen launch/replay source and comparison dumps are retained.
- **Shared implementation:** [Hitlist PR806](https://github.com/pirl-unc/hitlist/pull/806), draft commit2f4bbe0 on current base e8faa7f, adds the canonical standalone source package and ten EvidenceStore relations. Package schema20/catalog25/version2.0.20. Original build v1 failed before quantity projection on repeated SDRF column names; positional preservation and its regression fix the issue. Build v2 and complete source replay pass in379.92s/1.163GiB, manifest88cad10d7bc2a638305cb0d37ad96ac2826c3badaf7b3e0aad7928820df5c4b8.
- **Actual source closure:** 2,391,692 supplement cells including missing;558,216 deposited protein cells;607,928DIA rows;62,496 source records;40,964 FASTA entries;60,447 occurrence/unmapped records;83 sample groups/166runs/24contexts. Extra sample labels belong to DDA library construction, not additional quantitative replicates. A separate raw-format checker verifies every quantity/DIA row, source/sample/run links and each59,726 emitted exact match, with721 unmapped peptide rows;96.90s/.987GiB. It does not independently prove search exhaustiveness or specimen independence. No model fit or new predictive metric.
- **Query/testing:** Real unresolved-context querying exposed Hitlist805. Three null-join regressions fail before the fix; optional null rows now retain typed null metadata and cannot match null identities. Actual package queries pass. Format/lint and55 focused package/catalog/query tests pass. Full test script refused to launch on actual0.22GiB free against2.5GiB minimum; guard retained, full-suite/hosted validation and merge/release pending. All build attempts freeze source/config/diffs; original failure is retained.
- **Decision:** Complete Hitlist726 endpoint eligibility/recipe and source release, then preregister a fresh biological comparison. No Presto-only source adapter and no claim that elution data alone identify processing or intracellular concentrations. Prior biological baselines/tests remain unchanged. Full APC availability goal remains active and unachieved.

## 2026-10-24_0730_codex_hap1-response-baselines — Sequence predicts machinery response beyond controls

- **Date / agent / source:** October 24, 2026; Codex / GPT-6. Presto6045dd2 and Hitlist2f4bbe0 plus frozen recipe code, subsequently published as abbc376 in [Hitlist PR807](https://github.com/pirl-unc/hitlist/pull/807). [Report and reproduction](2026-10-24_0730_codex_hap1-response-baselines/README.md). Local CPU, float64, one numerical thread; no Modal, GPU or environment changes.
- **Source / curation:** Shapiro2025 PXD056426 human HAP1, canonical MHC-I elution plus supplemental MaxLFQ. Source88cad10d… / recipea1eec392…. One positive KO replicate minus mean log10 of at least two positive WT replicates. WT reference and B2M background only; exclude 520 B2M-positive sequences. Preserve all origins and source missingness; no imputation, inferred bounds or synthetic labels. Every candidate parent requires one unambiguous protein group and at least two positive WT/KO protein cells for the matched baseline population. Equal-parent log-ratio surrogate, without exact aliquot or molar claims. Other assay labels excluded.
- **Grouping / evaluation:** All-source peptide, claim, exact-parent and protein-group components precede eligibility filtering. Fixed 70/15/15 hash partitions. Complete-precursor training: 87,260 cells / 6,317 peptides / 1,798 components / ten KO conditions. Five fixed inner component folds; weights average condition/component/peptide/replicate. Outer validation (19,744 responses) and test (22,724) deliberately remain unscored, reserved for a frozen physical-model comparison. Same-study, seen-KO development; specimen, clone and homology independence unresolved. No applicable nM/binder metrics.
- **Training / output:** Four analytic controls; three condition-specific ridge penalties (0.001/0.01/0.1) using 61 length/composition/N/C features. Training-only transforms and closed-form weighted MSE plus L2; no optimizer, pretraining, restarts or identity parameters. Single log10 elution-response output. Nonnegative precursor gamma: full fit0.189554, folds0.121589–0.262808. Sequence slots610, supported/nonzero580, structural zeros30; ten offsets and gamma additional. No physical, detector, affinity/stability, TCR or auxiliary branch trains.

| Model | Training RMSE | Inner-CV RMSE | Inner-CV R² | Inner-CV Spearman |
|---|---:|---:|---:|---:|
| zero | 0.703614 | 0.703614 | -0.1417 | undefined |
| condition | 0.638312 | 0.638870 | 0.0588 | 0.2137 |
| fixed_precursor | 0.645376 | 0.645887 | 0.0380 | 0.2022 |
| calibrated_precursor | 0.637901 | 0.638745 | 0.0591 | 0.2215 |
| sequence_0.001 | 0.553116 | 0.576014 | 0.2349 | 0.4601 |
| sequence_0.01 | 0.553141 | 0.575716 | 0.2357 | 0.4601 |
| sequence_0.1 | 0.554784 | 0.574832 | 0.2380 | 0.4587 |

- **Decision:** Fixed sequence penalty0.1 improves inner-CV RMSE by10.006% over calibrated precursor. All ten KOs improve (4.0–13.4%); within-condition prediction SD0.1266–0.5307. The prespecified 2%-gain/0.001-spread gate passes. Useful biological response signal; no blind-test, concentration, causal-machinery or unseen-KO qualification. Preserve as the development reference for a matched physical exact/sampled comparison. No RL fit ran in this family.
- **Verification / resources:** Source and recipe424.19s/1.802GiB. Independent audit of all2,391,692 dispositions,220,271 responses,45,981 inputs and full graph114.83s/1.109GiB; maximum ratio discrepancy1.78e-15. All42 fits214.21s/1.027GiB, within caps. Independent augmented-LS replay of1,221,640 predictions/global metrics/gate52.37s/0.992GiB; prediction/coefficient discrepancies2.94e-12/1.90e-12. Detailed strata remain producer summaries backed by verified predictions. No biological retry. Hitlist43 focused and3,439 full unit tests pass; memory guard retained.
- **Artifacts / next:** Every training/inner-CV prediction, fitted cohort, coefficient/transform/support record, source/launch receipt, comparison CSV/PNG/PDF and verification log is retained. No outer-test dump by design. Shared canonical source/recipe paths are in the README. Hitlist806/807 merge/release remain separate pending work. Prior MAPP/native/macrophage tests stay closed. Full APC availability/concentration goal remains active.

## 2026-10-24_0810_codex_erap1-trimming — Correct trimming estimators, no gain beyond the biological sequence baseline

- **Date / agent / source:** October24,2026; Codex / GPT-6. Presto5699b9f plus frozen new module/launcher and recorded dirty state. [Full report, plan, predictions and reproduction](2026-10-24_0810_codex_erap1-trimming/README.md). Local CPU, float64, one numerical thread; no Modal/GPU/environment changes.
- **Dataset / curation:** Shapiro2025 PXD056426 human HAP1 canonical MHC-I elution plus supplemental MaxLFQ, source88cad10d… / registered recipea1eec392…. Only ERAP1 from the preceding complete-precursor outer-training population:8,281observations/3,304peptides/1,296components/1,514parents/3,409origins. One positive KO replicate relative to at least two positive WT cells; original B2M exclusions, all origins, unknowns and source lineage retained. No imputation, synthetic labels, target averaging, inferred bounds or source curator. Protein inputs are condition-level surrogates, not aliquot-paired concentrations. Other assay families excluded; nM/binder metrics inapplicable.
- **Model / output:** Unit full-parent input, fixed uniform cleavage/exit source compartment, then residue-dependent N-terminal trimming and clearance. All1..Lspans supported, including nonbinding/short products; monomers do not trim. Enumerate1,741,916birth paths using removed-residue counts, sum all origins, predict an offset plus log10 KO/WT amount ratio. KO rates use fitted effective sigmoid reduction; KO parent inputs use frozen fold-matched0730gamma. Augmented arm adds the physical correction to frozen sequence predictions and subtracts their precursor term. No trained TAP, selection/detector, affinity/stability, TCR or species/class branch. Elution alone does not identify rates or free-peptide concentrations.
- **Training / evaluation:** Uniform exact2parameters; residue exact/augmented exact/augmented sampled21each. Fourarms×(full+fiveinnerfolds)=24fits; zero init,150full-batchAdamsteps,lr.03/betas.9,.999/eps1e-8,seed42,finalcheckpoint. MSE+.01mean(log-rate²)+.01reduction-logit²; offsets profiled using fitting rows only. Every allocated coordinate updates; frozen baseline intentionally unchanged. Original molecular components/folds/weights retained. **Outer validation/test intentionally unscored** in this development family; previous inspected tests closed. Full per-observation train/inner-CV/fold dumps preserved; specimen/homology/external independence remains unproved.
- **Sampling:** Exact forward plus conditional score-function gradients, paired WT/KO quantiles; hybrid, not simulator-only RL/GRPO. No clipping/standardization/reuse/KL. K128 fails RMS.317–.334; predeclaredK512passes meanrelative.047–.052/RMS.127–.193 against.10/.25. Threefixedprobes/eightdraws, no regularization masking. Both attempts and numerical checks retained.

| Condition | Train RMSE | Inner-CV RMSE | CV R² | CV Spearman |
|---|---:|---:|---:|---:|
| Condition only |.686424|.686516|-.0003|-.0036|
| Calibrated precursor |.687914|.688302|-.0055|-.0100|
| Frozen sequence, penalty.1 |.581462|**.596370**|**.2452**|.5243|
| Uniform trimming / exact |.688704|.689090|-.0078|-.0451|
| Residue trimming / exact |.681335|.683779|.0077|.1389|
| Sequence + trimming / exact |.581296|.596436|.2450|.5248|
| Sequence + trimming / sampled |.581292|.596417|.2451|.5249|

- **Decision:** Keep the frozen sequence baseline. Exact augmentation is.0112%worse, failing the2%gate despite correctionSD.01023; component wins654/1296 versus642losses. Sampled is essentially tied, with no advantage over baseline. Physical-only residue model beats condition-only by only0.4%RMSE. No new causal/activity/concentration conclusion. All seven original controls, including zero/fixedprecursor/otherpenalties, are in the report and machine-readable summaries.
- **Resources / training evidence:** All24fits888.29s/2.280GiB. Summed fits:uniform377.46s,residue132.20s,augmentedexact93.46s,sampled280.82s (~3×matchedexact). Early fits were slower and brief source-integration checks overlapped; observed runtime, not an isolated hardware benchmark. Everyfit526,058,632exactpathscores; sampled adds507,494,400drawscores. Qualification57.60s/2.352GiB. Augmentedexactfinalgradientinf≤1.99e-7,sampled≤6.58e-5; local numerical stability does not establish correct biology.
- **Verification / artifacts:** Eight relevant software tests pass. Independent source/cohort/weight/fold/path checks and scalar recurrence reconstruct198,744fit predictions,maxerror9.05e-15; final offsets/penalties/losses/globalmetrics/gate checked and28,512component metric rows saved.8.34s/1.227GiB. Original raw-source audit reused; optimizer updates/gradient vectors not independently replayed. All checkpoints, histories, attempts, prediction dumps, sources/configs/logs,CSV/PNG/PDF retained.
- **Next:** Stop expanding this sampler on the basis of numerical correctness. Seek independent substrate-dependent processing and selection/detection constraints before a new prospective biological comparison. Shared Hitlist integration remains separately tracked. Full APC precursor-relative availability/concentration objective stays active.

- **Post-run source lead:** [Hitlist809](https://github.com/pirl-unc/hitlist/issues/809) requests independently measured ERAP1 substrate/product evidence with units, context and figure-versus-raw distinctions. Primary literature motivates auditing missing substrate-context effects; no such new labels enter this completed experiment.

## 2026-10-24_0924_codex_erap1-source-qualification — Deposited numbers do not provide a sequence-diverse processing cohort

- **Date/agent/source:** October24,2026; Codex/GPT-6. Presto e6ce3b2 with frozen new audit code and unrelated task dirt. [Report and reproduction](2026-10-24_0924_codex_erap1-source-qualification/README.md). Registered before workbook-cell inspection; executes the independent-source plan after the negative ERAP1 comparison. Local bundled Python3.12, no Modal/GPU/installations.
- **Exact dataset/curation:** Maben2021 DOI10.1038/s41467-021-25564-w, original Source Data ZIP a9d7bcba…/479,402bytes, two workbooks. Read-only source-field inventory, no production curator or biological-label admission. Primary article XML3776aa57… confirms assay/figure context. The2023 correction concerns an acknowledgment. Preserve missing, zero, above100 values, nonstandard probe chemistry, enzyme variants and unresolved experiment identities.
- **Conditions/evidence:** 34sheets:21structural/derived,2peptide-assay,9reporter-activity/mixed-fit,1reporter-fit and1image panel.3,726populated/3,390numeric cells, noformulas. Fig2h has70response cells/72slots for one peptide across12concentrations/three enzyme conditions. FigS8d has35/36cells for two probes/nine times. Both probes share the same N-terminal residue; no broad product-resolved substrate cohort. A duplicate concentration column inFig2i must not be read as a rate replicate. Numeric cells are not independent observation counts.
- **Training/evaluation:** none. No pretraining, synthetic labels, optimizer, loss, assay-output mapping, parameters, validation/test split, predictions or nM/binding metrics. The source audit precedes model admission; HAP1 outervalidation/test remain unscored and no prior biological baseline changes. No winner among predictive variants.
- **Verification/runtime:** ZIP/member integrity and byte preservation pass. Independent openpyxl3.1.5 reconstructs every populated cell, exact numeric floats, worksheet identity and counts. Inventory.116s, verification.044s/55.9MiB peak. Semantic classification is manual primary-source interpretation, not independently established assay calibration. Ruff and diffchecks pass; original executed/pre-format source is retained alongside final source. Every raw artifact is hash-linked.
- **Decision/next:** Fails admission as the independently anchored sequence-processing cohort sought after0810. Narrow biochemical observations could be represented later with actual units/context, but cannot train the intended specificity model alone. Update Hitlist809 with exact ranges, missingness, derived quantities and remaining metadata. Seek broader substrate/product measurements or native intracellular peptidomics from APC material. Full precursor-relative APC availability objective remains active and unachieved.

## 2026-10-24_0938_codex_native-apc-source-search — Public metadata available; capture semantics prevent native-APC admission

- **Date/agent/source:** October24,2026; Codex/GPT-6; Prestoeb4e8d8 plus frozen new audit files and unrelated task dirt. [Report and reproduction](2026-10-24_0938_codex_native-apc-source-search/README.md). Prospective source-selection plan precedes workbook retrieval. PeptidomicsDB DOI10.1093/nar/gkag1032, official public metadata exports, no per-sample peptide CSV retrieval. Local bundled Python3.12/openpyxl3.1.5, no GPU/Modal/installations.
- **Dataset/curation contract:** Original project/sample/dataset workbooks,468,673bytes total, exact URLs/hashes/receipts retained.151projects,3,992sample rows,249pairwise/14context datasets. Read-only inventory retains all source columns and identifiers.3,983unique raw-file names; sample counts reconcile but do not establish biological independence. No label generation, imputation, category normalization or source/parser admission to Presto.
- **Conditions/findings:** Fixed immune-cell lexical screen identifies29projects, all MHC-tagged;35human/mouse projects have non-MHC category labels. Manual review exposes lexical limitations and capture-versus-compartment ambiguity.20solely intracellular-tagged rows are enteroendocrine cells. JY samplePPD0170004 is both intracellular/MHC-tagged but primaryPMID34211107 uses HLA purification. GenericPPD070/PXD009801 is dynamic HLA-I/source-proteome labeling(PMID24715725), not admitted free-peptide evidence. Preserve source assertions and require primary preparation evidence. No exhaustive absence claim.
- **Training/evaluation:** none. No pretraining, optimizer, synthetic targets, loss/output mapping, validation/test split, prediction dumps or affinity/binder metrics; this is metadata qualification before a biological experiment. No new baseline or predictive winner. HAP1 outer outcomes remain unscored.
- **Verification/runtime:** Metadata inventory.255s/51.7MiB. Independent original-OOXML versus openpyxl comparison verifies all63,761nonempty values across four sheets,.559s/90.1MiB; original blank payloads stay separate fromzero, noformulas. All source hashes/CRCs pass.20exact source-row/cell locators preserve the manually reviewed examples. Checks prove fidelity, not assay/replicate semantics. Frozen code/launch/plans, candidate lists, source receipts and verification summaries retained.
- **Decision/next:** No native APC free-peptide cohort qualified in this screen. [Hitlist811](https://github.com/pirl-unc/hitlist/issues/811) requests shared source inventory and distinct specimen compartment, capture, laboratory digestion, search-enzyme, reanalysis-confidence, quantity and replicate fields with regressions for the concrete cases. Continue primary-source qualification and independent processing evidence; full APC precursor-relative concentration goal remains active and unachieved.


## 2026-10-24_1007_codex_erap1-length-response — Registered length-dependence diagnostic

- Codex / GPT-6; [prospective plan](agents/codex/plans/erap1-length-response.md), [experiment](2026-10-24_1007_codex_erap1-length-response/README.md). Same HAP1 ERAP1 outer-training cohort, grouped folds and frozen baseline as0810. Test three fixed remaining-length bands with exact and qualified sampled gradients. Explicit limited revision of the independent-numeric-anchor requirement; no new numeric source, physical-rate or concentration claim. Outer validation/test intentionally unscored. No results yet.

### HAP1 upstream publication closure (October24,14:28UTC)

Hitlist PR806/source and807/recipe are merged and deployed as2.0.22/2.0.23. Both successful manual release-build wheel/sdist artifacts match PyPI hashes; recipe artifacts also pass the clean-main release verifier atc7eed83ee58e97460fb549959bd3a6c132c87ae2. They were already published when this turn checked; no duplicate upload. [Verification receipt](../tasks/modular_model/hap1-release-verification.json). Original experimental source/recipe manifests and consumer versions remain unchanged; no new biological performance claim.


### Closure — 2026-10-24_1007_codex_erap1-length-response: length dependence adds no predictive gain

- **Date/agent/source:** October24,2026, Codex/GPT-6; Presto0248f11 with recorded dirt and frozen new code. [Report and reproduction](2026-10-24_1007_codex_erap1-length-response/README.md). Local float64CPU/one numerical thread; noModal/GPU/installations.
- **Data/curation:** unchanged Shapiro2025 PXD056426 HAP1 ERAP1 complete-precursor outer-training cohort:8,281positive replicate-response cells/3,304peptides/1,296components/1,514parents/3,409origins/1,741,916paths. Canonical source88cad10d…/recipea1eec392…, all origins/B2M exclusions/missingness/lineage retained; no imputation, inferred bounds, synthetic biology or new source curator. Target=log10 KO/WT MHC-elution response. Protein inputs are condition-level surrogates, not aliquot-matched concentration. Other assays and nM/binder metrics excluded/inapplicable.
- **Training/model:** add two remaining-length log-rate modifiers (1–8,9–16reference,>=17) to the previous N-terminal model. Uniform+length4coordinates or residue+length23; profiled fitting-only offset. Physical-only exact, residue+length exact, frozen sequence+physical exact and matched sampled arms each full+5fold fits. Same frozen sequence_0.1/gamma, hierarchical weights and component folds.150zero-initAdamsteps,lr.03,seed42,float64; MSE+.01mean(theta²)+.01mean(eta²)+.01z². No pretraining/other Presto heads. Every allocated coordinate updates. Explicit limited revision of the numeric-anchor prerequisite; fitted rates are not independently anchored.
- **Sampling/enumeration:** exact posterior/forward hybrid, K512paired conditional path draws, no clipping/std/reuse/KL. All paths and2,516,006padded states remain enumerated. Three fixed probes×eight draws pass meanrelative.0397–.0521/RMS.1374–.1652 against.10/.25gates; K2048fallback unused. Exact full-path equality<=6.22e-10, response<=9e-14. This is not simulator-onlyRL.

| Model | Train RMSE | Inner-CV RMSE | CV R² | CV Spearman |
|---|---:|---:|---:|---:|
| Old uniform |0.688704|0.689090|-0.0078|-0.0451|
| Length-only |0.688273|0.688680|-0.0066|-0.0248|
| Old residue-only |0.681335|0.683779|0.0077|0.1389|
| Residue + length |0.680819|0.683296|0.0091|0.1466|
| Frozen direct baseline |0.581462|0.596370|0.2452|0.5243|
| Baseline + old exact |0.581296|0.596436|0.2450|0.5248|
| Baseline + new exact |0.581294|0.596442|0.2450|0.5248|
| Baseline + new sampled |0.581291|0.596422|0.2450|0.5249|

- **Decision/evaluation scope:** gateFAIL; augmented exact0.0121%worse than baseline, sampled also worse. CorrectionSD.010261/.010146; nonzero variation does not establish useful prediction. Physical-only residue+length explains<1%weightedvariance. Keep direct baseline; stop this length extension without new tuning on these inner folds. Outer validation/test intentionally unscored and no test dumps; [separate prospective outer evaluation](agents/codex/plans/hap1-frozen-outer-evaluation.md) is next. No external, unseen-KO, native APC or concentration validation. Observed peptides8–14aa; shorter/nonbinding products remain unvalidated latent states.
- **Numerical failure preserved:** global-prefix attempt was stopped at11complete+1incomplete fits after a late full-path check found2.375e-7error above2e-9. Actual producer and independent diagnosis agree. Per-origin sums with bounded padding recover the unchanged mathematical model; all gates pass before the replacement fixed family. Original checkpoints/source/logs remain underattempt-global-prefix. Original1e-12gradient-floor typo was inactive; independent1e-8reduction identical, corrected before recovery. No biological settings/tolerances changed.
- **Resources/verification:** qualified24fits1319.54s/1.979GiB. Matched augmented exact290.51s vs sampled442.43s(1.52x), observed fitting times not replicated hardware benchmark. Independent scalar/source/weights/folds/offset/penalty/metric closure verifies198,744predictions,max1.52e-13;38,880component and210length-stratum rows;9.94s/1.288GiB. Initial closure checker hit a variable-name collision after reconstructing all24fits; original retained and name-only correction passes. Optimizer histories/RNG saved, not independently replayed.11tests,tracked-source lint/format and strictdocs pass; untracked historic raw-audit formatting exceptions preserved.
- **Artifacts/next:** all per-example full/fold and inner-CV predictions, checkpoints0/50/100/150, histories, source/phase receipts, tables/plots and failed-attempt outputs retained. Broader availability objective active and unachieved. [Hitlist812](https://github.com/pirl-unc/hitlist/issues/812) requests original APC-derived endolysosomal quantities/context; the linked software's example data are explicitly synthetic and not admitted.

## 2026-10-24_1503_codex_hap1-frozen-outer — Sequence prediction survives the reserved test; processing adds no gain

- **Date / agent / source:** October 24, 2026, Codex / GPT-6; Presto9a8742d with recorded task dirt and frozen new evaluation code. [Full report, predictions and reproduction](2026-10-24_1503_codex_hap1-frozen-outer/README.md). [Prospective plan](agents/codex/plans/hap1-frozen-outer-evaluation.md) predates outcome access. Local float64 CPU, one numerical thread; no Modal/GPU/environment changes.
- **Data / curation:** unchanged Shapiro2025 PXD056426 HAP1 canonical source88cad10d…/recipea1eec392…. Positive MHC-I KO replicate intensity relative to mean log10 WT intensity, with original >=2positive-WT/B2M-background/complete-precursor policies, all candidate origins and source lineage. Validation19,744 and test22,724 cells across ten KOs; ERAP1 subset1,970/2,297 cells. MaxLFQ inputs are condition-level surrogates, not paired concentrations. No missing-value imputation, synthetic labels, inferred censoring or additional source curator. Other assay families and nM/500nM metrics are inapplicable.
- **Models / training contract:** no new training, calibration, offsets, transforms, optimizer, seeds, early stopping or outer model selection. Seven0730full-training direct/control checkpoints, including three ridge penalties, and all four0810 plus four1007physical checkpoints. Direct inputs are61length/composition/terminal features, KO identity and calibrated precursor ratio; one log10 elution-response output. Physical inference sums all origins and birth paths with frozen N-terminal rates, optional remaining-length modifiers, effective KO scale, gamma and original training-only offset. Exact forward inference for both exact- and conditional-score-hybrid-trained parameters. No new RL/GRPO comparison or other Presto branch training.
- **Evaluation / weights:** both covariate-only prediction sets sealed before either outcome set loaded. Equal condition/component/peptide/replicate hierarchical weights computed within each split, ERAP1 renormalized. Weighted error/R²/SD, unweighted Pearson/Spearman. Shared study/WT controls and unresolved specimen/clone/homology independence remain. Component disjointness verified; no independent-cohort or unseen-condition test. Outer sets now inspected and closed to tuning.

| Predeclared comparison | Validation RMSE | Test RMSE | Test R² | Test Spearman |
|---|---:|---:|---:|---:|
| Ten-KO calibrated precursor |0.648257|0.631249|0.09081|0.23639|
| Ten-KO sequence_0.1 |**0.577617**|**0.573380**|**0.24986**|**0.46623**|
| ERAP1 sequence_0.1 |0.603220|**0.585327**|0.30062|0.57538|
| ERAP1 residue + length physical only |0.692245|0.698824|0.00310|0.14823|
| ERAP1 sequence + new exact physical |0.602332|0.585337|0.30060|0.57794|
| ERAP1 sequence + new sampled-trained physical |0.602393|0.585335|0.30060|0.57795|

- **Decision:** preselected direct model passes >=2%RMSE/nonconstant gates:10.897%validation and9.167%test gains. All ten KOs improve test RMSE4.25–16.61%, within-condition SD0.126–0.474. This is now held-out biological response evidence, beyond prior inner CV. Both physical gates fail: validation gain0.1471%; test0.001744%worse despite correctionSD0.010165. Keep sequence_0.1; other penalties/physical variants are fully reported descriptively, not selected from test. No demonstrated sampled-training advantage.
- **Verification / resources:** source/checkpoint hashes and677,068archived training predictions replayed before outcome access; target mutation leaves inference unchanged. Preparation15.84s/1.826GiB. Evaluation7.56s/1.490GiB, including metrics. Independent source/origin/precursor/group/weight/affine/scalar reconstruction verifies361,281held-out prediction values,max1.16e-13,and50,995stratum metric records;14.62s/1.267GiB. All caps pass. Inference runtime is not a training-throughput comparison. Five near-constant uniform/length-only validation component-stratum correlations are explicitly undefined in qualified strata; raw numerical values retained, all predictions/primary metrics/gates unchanged. Equivalent lint amendment preserved before preparation. Figure visually checked; experiment code lint passes.
- **Artifacts / limits:** full per-example validation/test source-linked predictions, frozen transforms/parameters/offsets, NPZs sealed before outcome access, complete metrics/strata, CSV/PNG/PDF, source/launch/verification receipts and raw logs retained. `model_to_beat.md` updated for the scoped HAP1 contract. HAP1 is not a professional APC; elution combines processing, selection and detection. Free-peptide amounts, short/nonbinding products, causal rates, independent detection and precursor-relative APC concentration remain unvalidated. Full availability goal active and unachieved; new independent biological constraints/cohorts are required rather than tuning these inspected sets.


## 2026-10-24_1521_codex_endolysosomal-source-qualification — Original APC-derived supplements do not supply a peptide quantity cohort

- **Date / agent / source:** October24,2026; Codex / GPT-6. Registered atbf7b3e64e484fd280ce311dffe2c15a93e6c4aca with existing task dirt. [Report, asset table and independent verification](2026-10-24_1521_codex_endolysosomal-source-qualification/README.md); reproduce/launch.json, launch.sh, source snapshots, exact source receipts and reconstructed follow-up argument vectors retained. Local CPU; no Modal/GPU/environment changes.
- **Dataset / qualification:** Öztemiz2026 DOI10.3389/falgy.2026.1936335, isolated murine JAWSII endolysosomal preparation with two plant allergens. Official article/XML, all five listed version1 PDF supplements and pinned linked software tree inspected. Preserve original source bytes, quantities/context limits and access failures. Source intensity is semi-quantitative; search-length filtering excludes short products, and reduction/alkylation affects observation. No invented peptide quantities, negatives, censor bounds or replicate identities.
- **Conditions / evidence:** S1 reporter-activity graph; S2 protein half-life graph plus eight aggregate n/mean/SD/SEM values; S3/S4 peptide span maps with color-binned intensity; S5 peptide time-course graph. Five pages and zero embedded files independently verified. Linked61-entry repository has one explicitly synthetic example workbook, excluded without download. No original peptide-by-sample numeric table recovered from these assets. Separate Figshare metadata request HTTP404 remains an access failure, not absence evidence; initial DNS failure followed by successful public retrieval is preserved.
- **Training / evaluation:** no model, pretraining, optimizer, loss, synthetic biological labels or assay-to-output mapping. No validation/test split or predictions because quantitative source admission precedes modeling. Regression/binder metrics do not apply. No existing outer outcomes reopened or model promoted.
- **Verification / runtime:**15 successful asset hashes,14,805,139 PDF bytes, five pages/no attachments, eight literal aggregate values and61 repository entries independently checked. Rendering/inventory17.61s/0.440GiB; independent Poppler/XML/hash verification0.43s/0.047GiB; retrieval receipt times sum14.97s excluding failed Figshare/discovery. All figures visually reviewed. Pypdf form-XObject truncation on S3/S4 explicitly retained; original text/log archives round-trip byte-for-byte. New verifier lint passes; frozen fetch sources retain two UP017 style findings without source mutation.
- **Decision / next:** admit no quantitative peptide cohort from the inspected public assets; this is not an exhaustive absence statement or denial of valid aggregate protein measurements. Update shared ownerHitlist812 with exact asset coverage and missing peptide/sample/replicate fields. Independent measured MS-response data underHitlist762 are the next constraint to qualify. APC concentration/availability objective stays active and unachieved.


## 2026-10-24_1546_codex_measured-ms1-source — Recover original nontryptic MS response and intended products; reject equimolar assumptions for all IDs

- **Date / agent / reproducibility:** October24,2026; Codex / GPT-6. Registered at2bf8ca1f0a5ee5ace7599920b6a43528be66e604 with task dirt and new source audit. [Report, source inventories, row dispositions and independent checks](2026-10-24_1546_codex_measured-ms1-source/README.md). Frozen launch.json/sh, executed scripts, original/corrected resource plan, actual per-asset retrievals and explicit extraction operation retained. Local CPU; no cloud/GPU/environment changes.
- **Exact source / curation:** PXD021013 experimentally synthesized nontryptic human peptides, original2021 DOI10.1038/s41467-021-23713-9; discovered through2023 reanalysis DOI10.1016/j.csbj.2023.07.027. Inventory all1,865 RAW/1,865 SEARCH references (one nominal ZIP exposed as directory), community SDRF1,008raw names, two first named HLA-I/HLA-II DDA archives, and all four original intended-sequence mapping sheets. Keep source bytes/cells, molecular forms, zeros, PEP, reverse/contaminant status and planned membership; no source repair or training adapter. Other1,863 search references remain uninspected. PXD004732/PXD010595 and PXD003472 are not quantitatively admitted by this audit.
- **Source findings:**305,331 intended sequence-to-pool rows across421 pools,610,670 populated map cells independently verify. Two processed archives have3,649 peptide/3,738 form rows: planned positive quality sequence signals759/484, corresponding unmodified positive quality forms677/470. Off-planned identifications1,101/1,197 must not inherit the intended standard's nominal amount. Planned-but-unobserved sequences140/9 are not pure detector negatives. Source uses crude products and estimated200or500fmol injection; true per-peptide purity/amount and run-specific load choice remain unresolved. SDRF material=cell, generic Fusion/invariableHCD/run1 conflict with primary synthetic/Lumos/multimodal context; retain raw assertions, do not infer specimens. Repeated modification columns preserved by position. Search enzyme is not laboratory digestion. Published reanalysis filters recurrence/CV and final tryptic subset; its accuracy does not establish nontryptic transfer.
- **Training / evaluation:** none. No model/pretraining/optimizer/loss, no fabricated labels, no assay-to-model output, no validation/test split or predictions because this is source qualification. No regression/binder metrics or new predictive improvement claimed. Existing biological held-outs remain closed.
- **Verification / runtime:**369,548 original literal tab cells independently agree via CSV/pandas,0.28s/0.224GiB. Full intended-map XML/openpyxl comparison610,670cells,26.58s/0.436GiB; combined980,218cells, all sampled row dispositions retained. Original232,332,843-byte public supplement bundle retrieved in50.03s after prospective cap amendment100→400MiB for this asset, total transfer<=1GiB; member63,188,883bytes retains original CRC/hash. Initial over-cap attempt and inaccessible individual routes preserved, no access bypass. Expanded request begins15:58:38UTC; original heading's rounded15:59caption is explicitly corrected/preserved. Initial index assertion failure retained; recovery correctly records nominal ZIP-as-directory, no data changes. New code lint/format passes; raw logs losslessly archived.
- **Decision / next:** source existence and intended-product joins pass. A canonical shared relative-MS1-response recipe is feasible; true molar detector calibration is not established. Implement shared Hitlist762 ingestion/recipe with planned product, molecular form, run/pool dependencies, all dispositions and load/context uncertainty. Then preregister sequence-versus-constant/length response prediction with grouped held-outs, retaining synthesis/handling confounding. Use direct differentiation for this continuous observer; later exact/sampled processing comparisons must share its frozen independently evaluated response term. No claim that this completes APC concentration/availability prediction.

## 2026-10-24_1635_codex_ms1-canonical-admission — Canonical original MS1 quantities and conditional response recipe

- **Date / agent / reproducibility:** October24,2026; Codex / GPT-6. [Report and all dispositions](2026-10-24_1635_codex_ms1-canonical-admission/README.md); [prospective plan](agents/codex/plans/ms1-canonical-admission.md). Presto a39d527 with recorded task dirt; Hitlist bafa4fa plus frozen source/diff, candidate2.0.26. Exact launch/env/source snapshots and separately frozen final grouping amendment retained. Local CPU, no cloud/GPU or environment installation.
- **Source / curation:** same two original PXD021013 nontryptic synthesis-pool DDA archives qualified1546. Full305,331row intended mapping, primary protocol, README and positional SDRF;117,129,479original bytes. Canonical standalone EvidenceStore source23/catalog28 and registered intended-unmodified response. Both sequence/form positive finite,PEP[0,.01),no reverse/contaminant,canonical/intended/unique variable-Unmodified form; fixed CAM explicit. Unknown purity/run load and conflicting metadata preserved. No missing negatives, synthetic biological labels, CV filtering or molar calibration.
- **Outcomes:** HLA-I677/HLA-II470admitted products from3,649peptide/3,738form rows;1,147admitted,204observed-excluded,149unobserved among1,500planned pairs. All forms retain dispositions. Independent original decoding verifies320,222literal cells and every target. Source and final recipe hashes in report. Counts independently match prior1546membership audit.
- **Dependence / corrections:**106shared observed identities are all outside both intended pools. Initial graph unnecessarily merged them; before any fit, revised grouping uses all intended products including unobserved ones. Two components; all1,147targets unchanged. Original grouping and overlap audit preserved. Also preserve two parser failures (experiment totals and repeated parameter rows), source snapshots and explicit corrections; no eligibility relaxation.
- **Training / eval:** none; no pretraining, optimizer, loss, assay-to-model output or held-out metrics. No test split or prediction dumps because this is source qualification; examples contain measured conditional log10 signal targets, not predictions. Synthetic files only test software. No model promoted and previous biological held-outs remain closed.
- **Resources / decision:** main build/replay/independent audit27.78s/1.011GiB; final grouping qualification9.89s/1.116GiB,within3GiBprocess/1GiBartifact caps. Shared canonical ingestion is ready for review; release and larger grouped observer fitting remain pending. Proceed to prospectively selected pools and sequence/length/constant response comparison, then freeze observer for matched processing methods. Actual APC availability/concentration objective remains unachieved.


## 2026-10-24_1702_codex_ms1-grouped-observer — Held-out sequence signal in measured non-MHC MS1 response

- **Date / agent / source:** October 24, 2026; Codex / GPT-6. Presto1606976 with recorded task dirt and frozen experiment-local code. [Full report, exact launch/env, predictions and checkpoints](2026-10-24_1702_codex_ms1-grouped-observer/README.md). Local float64 CPU, one Torch/BLAS thread; no Modal/GPU or shared-environment modification.
- **Dataset / curation:** original PXD021013 synthesized nontryptic peptide pools, canonical Hitlist source e373789b… and recipe0183d91b… (`pxd021013_intended_unmodified_ms1_v1`, runtimeb9b1047). Metadata-only fixed-hash selection excludes pilot overlap and enforces intended-sequence-disjoint pools. All24 original indexed DDA archives, fixedCAM/MaxQuant1.5.3.30/unspecific context. Positive finite intended canonical sequence and unique variable-Unmodified form, PEP[0,.01), no reverse/contaminant.17,517intended products:12,877admitted,1,852observed excluded,2,788unobserved unknown-cause, not negatives. Train16pools/8,712examples, validation4/2,104, test4/2,061. No synthetic numerical biology, imputation, bounds or affinity labels; nM/binder metrics inapplicable.
- **Target / information flow:** sequence features predict one centered log10 measured MS1 signal; per-run centering uses predicted scores, never target offsets. Conditional on admitted positive candidates. Unknown synthesis/purity/load/handling confound pure detector response. HLA-I/II library origin is provenance, not measured restriction. Same-study pool holdout, not independent biological/instrument transfer.
- **Training:** zero, length ridge (2coefficients), sequence ridge (42supported coefficients), alpha.001/.01/.1/1 selected by validation; weightedMSE + alpha*mean(coefficient²). Neural53features→32tanh→scalar, no output bias,1,760parameters; seeds42/43/44,400full-batch Adamsteps,lr.003,betas.9/.999,eps1e-8,penalty.001mean(parameter²), no pretraining/dropout/restarts. Validation checkpoints every25 select50/75/75; fixed ensemble. All allocated neural coordinates receive gradients and change. No RL or other Presto branches trained; continuous observation subproblem. All C-terminal main effects are unidentifiable under these pools/centering, and nine terminal categories are absent in training. Validation/test raw-range extrapolation897/8examples is retained.

| Fixed comparison | Validation RMSE | Test RMSE | Test R² | Test Spearman |
|---|---:|---:|---:|---:|
| Zero |0.534813|0.520561|0.000000|undefined|
| Selected length ridge |0.534663|0.520456|0.000405|0.078182|
| **Selected sequence ridge** |**0.470624**|**0.449828**|**0.253294**|**0.525644**|
| Fixed neural ensemble |0.463737|0.443033|0.275683|0.547499|

- **Decision:** primary passes all gates:11.98%validation/13.57%test RMSE gains, positive Pearson/Spearman in4/4test pools, prediction SD/target SD.4726/.4574. Secondary ensemble13.27%/14.88%gain; keep predeclared sequence ridge as scoped reference, not a test-selected winner. All13variants, every pool/library stratum and source-linked per-example dumps retained. Errors/R²/SD equal-pool weighted; correlations unweighted. No independent-study replication or confidence interval; all held-outs now closed.
- **Verification / runtime:**4,767,232original cells and12,877targets independently reconcile. Two math/gradient tests pass1.21s. All checkpoints and test covariate predictions sealed before reserved evaluation. Independent feature/weighted least-squares/neural inference/dump/metric reconstruction verifies167,401predictions and182metricrecords,max1.0814e-13. Source admission39.18s/1.716GiB, original replay9.18s/.778GiB, allfits5.42wall/5.48CPU seconds/.393GiB, evaluation1.00s/.267GiB, independent prediction replay.66s/.163GiB. Optimizer trajectories retained, not independently replayed. This is not a replicated throughput study.
- **Artifacts / failures:**660,751,734download bytes including partial attempts; raw+canonical1,582,596,202logical bytes, all resource gates pass. HTTPS403/S3-500 and initial FTP timeout preserved; original HLA63/HLA112 recovered via public FTP with metadata/CRC/hash checks, no cohort change. Actual remaining-download script's missing FTP fallback corrected in a new frozen launcher. New code isolated lint and tracked-source pinned lint/format pass; whole-tree lint's two preexisting frozen untracked ERAP1 lines remain unchanged. Plot visually checked. Raw/canonical paths and source/recipe manifests, all receipts/logs/checkpoints are in the report.
- **Context / next:** advances from source admission to measured-response generalization, but does not show that MS response caused MAPP/macrophage failures. Test a frozen observer on a new prospective biological cohort before combining it with processing as a validated correction. Exact and sampled methods must share it and the same objective; omitted between-pool/terminal effects and unknown load remain. APC free-peptide concentrations and short/nonbinding products are still unvalidated; full availability goal active and unachieved.


### Canonical MS1 release closure (October 24, 17:52 UTC)

Hitlist814 is merged at68dea0b and **2.0.26 is published**. The clean-main release build40115753764 passed formatting/lint,3,457unit tests (one skip because Ensembl release data were not installed),96integration tests, distribution checks and artifact provenance verification. Runtimeb9b1047 and merge68dea0b have identical source trees. PyPI wheel/sdist SHA256 hashes exactly match the tested artifact manifest. [Release receipt](../tasks/modular_model/ms1-release-verification.json), full compressed build log and release manifest are retained. The user's shared Hitlist editable remains1.82.2 at its original checkout; pipcheck passes. This closes source distribution, not biological calibration or the full availability goal.

The next [prospective admission plan](agents/codex/plans/observer-composition-transfer-admission.md) recognizes that a fixed peptide-response factor cancels from same-peptide treatment/control ratios. Candidate PXD057588 Tables7/8 may supply within-condition raw TMT compositions, but prior-component overlap, chemistry/domain transfer and comparable cross-peptide quantities must be qualified before fitting. No new biological cohort or predictive result is claimed by that plan.


## 2026-10-24_1755_codex_apc-composition-admission — Fresh macrophage composition cohort fails prospective support

- **Date / agent / reproduction:** October 24, 2026; Codex / GPT-6. Presto534ae42 with recorded task dirt; [full report and frozen invocation](2026-10-24_1755_codex_apc-composition-admission/README.md). Local CPU/one numerical thread, no Modal/GPU/environment changes.
- **Data / contract:** canonical PXD057588 Tables7/8 raw positive TMT quantities, source d99f67e6… through released Hitlist2.0.26. Canonical sequence, one full-parent sequence, at least three distinct peptides per parent/table/channel. Exclude all dependencies on prior Table9 training/validation/test components or parents. Fixed component hash split; >=100/25/25 parents required. Preserve all source cells/origins, unknowns and missingness; no imputation, invented negatives or synthetic biology. MHC-selected composition is not intracellular concentration.
- **Results:** 6,378 rows, 38,268 quantities, 2,760 eligible rows before group minimum. Only **45 train / 5 validation / 7 test parents**, in324/30/54 channel groups: all three gates fail. Repeated channels are not independent parents. No predictive winner or baseline change.
- **Training / evaluation:** none; no pretraining, optimizer, loss, assay-output mapping or predictions. Candidate partitions inspected only for support, so validation/test predictive metrics/dumps and affinity/binder metrics are inapplicable. The1702observer remains frozen; no biological transfer claim.
- **Verification / resources:** independent OOXML and graph reconstruction verifies38,268 quantities,12,756 sequence/parent cells,6,378 dispositions and13,770 candidate groups. Inventory17.05s/0.948GiB; verifier1.02s/0.474GiB. Initial verifier numeric-header text mismatch preserved; rendering-only correction passes without source/gate/split changes. Failed primary XML HTTP500 receipt retained with public-server cookie headers redacted from publication derivatives.
- **Decision:** no fit under this design; retain failed support screen. TMT/form aggregation differs from unmodified MS1 calibration, and exact specimen/acquisition links remain unresolved. Need an independent quantitative biological cohort for observer transfer; do not reopen inspected tests or relax this split. Full APC availability/concentration goal remains active and unachieved.


## 2026-10-24_1818_codex_do-source-qualification — Recover embedded HLA-DO spectrum intensities; quantitative semantics remain unqualified

- **Date/agent/reproduction:** October24,2026; Codex / GPT-6, Presto e8d5daf with task dirt. [Report, receipts, exact sources and independent checks](2026-10-24_1818_codex_do-source-qualification/README.md). Local CPU/one numerical thread, no Modal/GPU/environment changes.
- **Source/contract:** Nanaware2019 PMID30573663/PMC6398211/MSV000082570/PXD010301. Downstream MHC-II composition/measurement candidate; preserve human/mouse, clones, DO-loss, predicted cores, outcome-selected isotope standards and intracellular versus MHC-bound quantity distinctions. Existing Hitlist IEDB arm mismatch retained; no private training curator.
- **Initial error/correction:** field-name-only audit incorrectly inferred absence of numeric observations. Full-attribute review recovered `intensity=...` inside spectrum titles. Initial source/result/report/issue preserved; [Hitlist816](https://github.com/pirl-unc/hitlist/issues/816) corrected. Separate XML and string decoders verify4,751source results:4,256positive/495missing values,4,080distinct positive values,349results sharing a positive value;4,800identification alternatives retained. Duplicate name/CV titles are one observation, not two. Units/precursor integration/normalization/aggregation remain unresolved; no source-equivalent peptide quantities admitted.
- **Other evidence/access:**328-file repository inventory and10original hashes/14,866,858bytes verified. Original quantitative supplements not recovered: EuropePMCXML500, retired OAAPI404, documented cloud-prefix empty, supplement API200 error XML, supplemental-index200 browser challenge. Preserve failures without an exhaustive data-absence claim. Three large parental-lineMSF files remain uninspected. Vendor normalization documentation is not proof of this historical export's settings.
- **Training/evaluation:** none; no model/pretraining/optimizer/synthetic targets/loss/output mapping or predictive split/dumps. Regression/binder metrics inapplicable; no predictive winner or baseline promotion.
- **Resources/decision:** initial inventory2.66s/0.207GiB; independent title recovery0.49s/0.198GiB. Retain recovered observations for canonical source-semantic qualification; supplements/access and aggregation remain upstream gaps. Next independent2024lymph/DCsource now has original workbooks retrieved; eligibility/paired materials still pending. Full APC availability/concentration goal remains active and unachieved.


## 2026-10-24_1829_codex_lymph-source-qualification — Original lymph/DC quantities recovered; admission remains pending

- **Date / agent:** October24,2026, Codex / GPT-6. [Registered source audit and current status](2026-10-24_1829_codex_lymph-source-qualification/README.md); frozen plan, exact retrieval stages, original bytes and independent verification retained. Source qualification is closed by the independent1903canonical source replay; this is not a training family.
- **Source:** Nanaware2024 PMID38848214 / PMC11233987.319, seven original workbooks plus primary XML/license metadata, 80,601,182bytes in the processed stage. Native human lymph, mouse lymph proteomics and separate DC I-Ab immunopeptidomics are distinct populations. Exact specimen/precursor joins and confidence/normalization policies remain pending.
- **Verification:** Nine source hashes/MD5s, seven ZIP CRCs,72sheet identities and14,567original cells pass independent decoding;1.129s,0.157GiB,localCPU. Original S6(a) has1783positive area-bearing rows/1138literal forms. Two S4(e) algorithm blocks have1581/1020rows and2536/4632positive area cells; these are not independent replicates or admitted examples. Multipart-ETag/shared-formula/elapsed-time/time-of-day failures and serialization corrections are preserved without changing source quantities.
- **Training/evaluation:** None; no optimizer, pretraining, loss, synthetic labels, assay head, held-out metrics or per-example predictions applies. This does not raise biological predictive accuracy or establish APC concentrations. Complete inventories, sample semantics, independent quantity-count replay and canonical Hitlist admission before a fresh frozen biological comparison; old predictive tests remain closed.


## 2026-10-24_1903_codex_lymph-canonical-admission — Shared original lymph/DC quantities and deposited references verify

- **Date / agent:** October24,2026; Codex / GPT-6. [Report and frozen reproduction](2026-10-24_1903_codex_lymph-canonical-admission/README.md). Presto e8d5daf, isolated Hitlist base68dea0b with exact changed-source snapshots; imported2.0.27development while shared installed1.82.2remains unchanged.
- **Source/curation:** Nanaware2024 PMID38848214,19original public assets. Project only human S6(a) A:Q and independent mouse DC S4(e) database/library blocks.4384source observations,32995quantity cells including missing,3430deposited reference entries and23455identification alternatives. Three typed contexts/seven exported channel labels/fourteen assertions preserve species, compartment, DC expansion, MHC capture, instrument conflict, unknown donor/specimen/KO links and algorithm dependence. No row-wise cross-block joins, normalization, filtering, aggregation or synthetic biological labels.
- **Verification/resources:** Independent121346original-cell/32995quantity-cell replay and every reference/evidence alternative pass. Three PRIDE inventories4/13/50files and two MassIVE132/44files reconcile. Source build/API15.217s/0.461GiB; independent verification2.267s/0.557GiB,localCPUone thread. No Modal/GPU/RAW acquisition/environment change. Exact manifest6e82a782b0511ba56425855f0764c4c40127e715fdb10b10e6c12118ab539c4c, input paths/snapshots/receipts retained.
- **Training/metrics:** None; no pretraining, optimizer, assay loss/output, test split, prediction dump or binding cutoff applies. Eight source-focused tests and43source/catalog focused tests pass; full required local suite passes3466tests after updating seven catalog-version expectations. PR818published; CI/merge/release pending. This source result does not improve biological predictive accuracy.
- **Decision:** Source quantities are usable as evidence; training eligibility is not yet established. Native human lymph is extracellular and DC peptides are MHC-selected. Exclude mouse no-enzyme re-searches from native-processing supervision until preparation is resolved. [Hitlist817](https://github.com/pirl-unc/hitlist/issues/817) owns source/metadata requirements. Next register the observable-composition recipe and a prospective matched biological comparison; old predictive tests remain closed, full APC-concentration goal active.

1903structural follow-up: declared accession-token matching supports1612human/875DCdatabase/941DClibraryrows with186/196/191candidate parent sequences. Raw full-label control and per-observation alternatives remain preserved. All2601DCreportedlengths are one residue short; all2601full literal modified masses match source Mass within0.1Da, while none of the shortened hypotheses match. Two independent mass calculators agree within0.002Da. No sequence/source correction or training admission occurred; Hitlist817records the evidence. Full per-row structural/mass results and frozen commands retained.

## 2026-10-24_1938_codex_lymph-composition — Independent biological MS transfer and processing fail held-out gates

- **Date/agent/reproduction:** October24,2026; Codex / GPT-6. [Full report, checkpoints, trajectories, prediction dumps and exact reproduction](2026-10-24_1938_codex_lymph-composition/README.md). Presto992cc934with recorded task dirt and frozen new source. Isolated Hitlist2.0.28recipe PR819/c61ad506; source818merged9133736d. Exact runtime patch/dependencies/env in reproduce. Local float64CPU/oneTorch-BLASthread; no Modal/GPU/shared-environment changes.
- **Data/curation:** Nanaware2024 original human native extracellular IBD lymphS6(a), separate mouse expanded DC I-AbS4(e)database/library. Source manifest6e82a782… and registered `nanaware2024_observed_composition_v1`. Positive originalPEAKSarea; representative feature chosen by confidence then charge/ID before quantities; no charge sum, missing negatives, imputation, newFDRclaim or synthetic biological labels. Chemistry matches frozenobserver: canonical unmodified except explicitCAMonallC. One distinct original parent sequence from suppliedaccession and same-study deposited alternatives; retain all occurrences. All-source peptide/accession/parent graph includes excluded bridges and bothDCalgorithms before fixed60/20/20hashsplit. Group>=3peptides; targets centeredlog10area/area fractions withinparent/sample. Conditional feature signal, not molarity or complete freepeptidepool.
- **Support:**2,472admitted feature measurements from32,995sourcecells. Human34/14/12components,36/14/14parentgroups,459/135/74features train/validation/test. DCdatabase36/1/6 andDClibrary32/3/11components fail predeclared20/5/5support; noDCfit, testmetrics or dumps. Algorithms reuse acquisitions; human oneexportedchannel has unresolveddonor/pooling. No source/split/gate relaxation.
- **Training/output:** length rank2 and sequence rank56ridge, alpha.001/.01/.1/1selected byvalidation (both1), training-onlystandardization/centering/SVD; zero, frozen1702MSridgealpha.1and train-calibratedscalar controls. Four physicalMSEarms: uniform/adjacent-residue sequence cuts × no/fixedMS. Unitparentinflux/commonremoval1, all1..Llatentproducts, exactqueriedspan amounts/occurrencesum withoutO(L²)materialization.1or39parameters, allgradient-supported/changed.500fullbatchAdamstepslr.02,betas.9/.999,eps1e-8, .01mean(residue²),norhopenalty,nostepsselection/restarts. Assayarea→centeredlogscore/distribution; noIC50/bounds/binderthreshold/TCR/MHCaffinity/stability/species/KOauxiliarytraining.
- **Matched RL contract:** sequence+fixedMS exactreverseKL(q||observedfractions) vs freshREINFORCE64withleave-one-outrewardbaseline,seeds42/43/44,same500updates/model/initialization/penalty. Genuine discrete candidate-identity sampling conditionalonobservedset, not completeprocessingtrajectory simulation or priorconditional-agehybrid. Allcandidate scorescomputedbotharms; no clipping/std/reuse/referenceKL/pretrainedprocessing. Exactseedsintentionallyidentical; seedSDnotbiologicaluncertainty.

| Fixed model | Validation RMSE | Test RMSE | Test reverseKL | Test R² | Test Spearman |
|---|---:|---:|---:|---:|---:|
| Uniform observed composition |.56074|.48566|.54365|0|undefined|
| Selected length ridge |.57622|.47813|.52164|.03077|.36789|
| Selected sequence ridge |.63085|.46704|.51001|.07520|.28195|
| Frozen measuredMS |.59776|.48296|.53494|.01107|.25317|
| Train-calibratedMS (coefficient.23353) |.56271|.47913|.52722|.02670|.25317|
| Sequence processing only |.53267|.49952|.60151|−.05790|.23842|
| **Primary sequence processing+MS** |**.56201**|**.48463**|**.54260**|**.00424**|**.25793**|
| Matched exactreverseKL |.58684|.54705|.70626|−.26880|.27147|
| MatchedREINFORCE64 (3seedmean) |.58540|.54493|.71020|−.25906|.26918|

- **Decision:** primaryfails5%gatesagainstsequence/calibratedMS; fixedMStransferfailsagainstlengthonbothheldouts. Primarytestgainvssequence−3.77%, paired12componentbootstrap95%[−18.39%,+10.56%]; versuscalibratedMS−1.15%,[−16.37%,+12.21%]. OutputSD.27949vs target.48566isnotconstant. Exact/RLbothlose touniform; RLtestRMSESD.00481/reverseKLSD.01514. No modelpromoted; no test-selected replacement. ExistingHAP1sequence-responseevidence remainsstronger. Keep exact as practicalreference, investigate processing/observationtransfer ratherthan optimizerelaboration; preserve inspectedtests.
- **Verification:** independentoriginal-cell/origin/featureselection/NetworkXdependency/split/targetreplaypasses. Fiveexistingsteady-state tests; realfinitedifferenceerror2.90e-10, exactscore-gradientidentity1.59e-15.256×64samplingprobe relative meanerror.01342, absolute.00209,relativeRMS.27666passes. Allcheckpoints/testcovariatepredictionssealed beforetestlabelread. IndependentNumPyfeatures,ridgecoefficients,observer,scalarparentformulas,dumps and63split-modelmetricrecords reproduce14,028predictions max9.58e-13;coefficients2.22e-16. Adamtrajectoriesretained,notindependentlyreplayed. Initialverifierweight-variable shadowingfailure/code/logpreserved; verifier-onlyfix changesno model/metric/tolerance.
- **Resources:** admission7.02s/.813GiB; independentdata1.95s/.505GiB; qualification.95s/.302GiB; all21fits12.77s/.395GiB; eval.56s/.296GiB; independentpredictionreplay.77s/.116GiB. Exact1.47s/fitvsREINFORCE1.71s(~16%slower),334,668candidatescores each,1,152,000sampledraws perRLfit. Orderedlocalprocess,notreplicatedhardwarebenchmark. Allcaps pass. Hitlistformat/lint/3,477fulltests and19focusedtests pass; sourcePyPIrelease40126246020andrecipeCIremainseparatependingwork.
- **Transfer/metadata limits:** post-evaluationcovariateauditfinds1testpeptide previouslyinobservertrain and1inobservervalidation;5/74testfeatures exceed anactiveobservertrainingfeaturerange. No exclusions/refitfollow. Detectorresponse,sourcefeaturegrain,extracellularkinetics,selectionandunmeasuredoriginsremainconfounded; thesechecksdo notdiagnosethecause. [Hitlist817 follow-up](https://github.com/pirl-unc/hitlist/issues/817#issuecomment-6388815228) retainsquantityaggregation,originalFASTA andspecimen/precursorrequests. FullAPCconcentration/availabilitygoalactiveandunachieved.

## 2026-10-24_2018_codex_hap1-context-ablation — Parent context fails the incremental prediction gate

- **Date / agent / reproduction:** October 24, 2026; Codex / GPT-6, Presto d96f958 with recorded unrelated task dirt and frozen additions. [Report, predictions and reproduction](2026-10-24_2018_codex_hap1-context-ablation/README.md). Prospective nine-arm specification, exact launch/env/source hashes, full/fold checkpoints and independent closure retained. Local float64 CPU / one numerical thread, no Modal/GPU/environment changes.
- **Data / curation:** unchanged Shapiro 2025 PXD056426 HAP1 source 88cad10d… / recipe a1eec392…; only original complete-precursor outer-training rows: 87,260 observations / 6,317 peptides / 1,798 components / ten known KO conditions. Positive replicate log10 KO/WT MHC-elution responses; original WT support, B2M exclusion, precursor ambiguity and missingness policies preserved. No imputation, generated negatives, synthetic biological targets or censoring. Other assay families and nM/500 nM metrics inapplicable.
- **Training / output:** original calibrated precursor coefficient and condition offsets; separate per-condition ridge effects, fixed penalty 0.1 × sum(beta²), fitting-only weighted scaling/centering. Nine arms × full/five component folds = 54 fits. Length, P2/C, P1/P2/C, original 61 sequence features, and geometry/flank/both/shuffled-context additions. All 6,499 occurrences in 2,077 parents retained with equal-parent then equal-occurrence weights. No pretraining or optimizer; no other Presto heads train. One continuous log10 elution-response output.
- **Evaluation scope:** reused original five training folds; full-population hierarchical weights for pooled CV, within-fold fitting weights. Outer validation/test intentionally remain closed after prior 1503 evaluation, so no new test predictions/metrics. This is development evidence, not independent validation, unseen-KO transfer or concentration calibration. No new exact/RL comparison; the diagnostic addresses representation after both gradient approaches failed biologically.

| Fixed arm | Train RMSE | Inner-CV RMSE | CV R² |
|---|---:|---:|---:|
| calibrated_precursor | 0.637901 | 0.638745 | 0.0591 |
| length | 0.632084 | 0.633505 | 0.0745 |
| position_p2_c | 0.591769 | 0.604202 | 0.1582 |
| position_p1_p2_c | 0.582525 | 0.602395 | 0.1632 |
| sequence | 0.554784 | 0.574832 | 0.2380 |
| sequence_geometry | 0.546492 | 0.568473 | 0.2548 |
| sequence_flanks | 0.541846 | 0.577124 | 0.2319 |
| sequence_context | 0.537528 | 0.573501 | 0.2415 |
| sequence_shuffled_context | 0.547377 | 0.584759 | 0.2115 |

- **Decision:** primary context gain 0.232% versus sequence (required 2%), 1.925% versus shuffle (required 1%), three of five folds improve (required four): **FAIL**. Geometry-only descriptive gain 1.106%; flanks worsen CV despite better training error. Position-only controls underperform composition-containing sequence; this does not exclude MHC selection. Retain existing held-out sequence reference, stop this feature/penalty family, qualify explicit selection covariates and independent biology before a further mechanistic comparison.
- **Verification / resources:** reference replay ≤4.00e-14, independent context ≤1.78e-15 and all canonical occurrence sets agree. 54 fits 60.62 s, total 62.72 s, peak 2.627 GiB. Independent SVD reconstructs 540 condition systems / 1,570,680 predictions, max 1.57e-13, coefficients 2.34e-14; archived metrics ≤2.78e-15, 225 strata ≤1.23e-15. Verification 77.04 s / 2.136 GiB. Lint/format pass; figure visually reviewed. Single-process timings are not replicated hardware benchmarks.
- **Preserved verification failure:** raw cross-implementation Spearman differed 2.50e-8 / 1.25e-8 for geometry train/CV because identical inputs acquired 3.3–4.4e-16 numerical differences. Failed verifier/log retained; exact-input tie qualification gives identical rank metrics. Revised check separates unchanged-tolerance prediction/SVD agreement, metrics of archived bytes and continuous replay metrics; original raw-rank equality check explicitly replaced, no fitted result, biological gate or tolerance altered. Raw/tie-qualified scores and predictions retained.
- **Upstream / limits:** primary source names MixMHCpred 2.2; nine inspected text-export headers expose no binding-score field, not an exhaustive data-absence claim. [Hitlist #726 follow-up](https://github.com/pirl-unc/hitlist/issues/726#issuecomment-6389204773) owns request for original per-allele score/assignment provenance, kept distinct from measured restriction. Lymph source 2.0.27 PyPI bytes verified; recipe #819 merged c7abcced, tested release 40129209653 still running. Full APC availability goal active and unachieved.

## 2026-10-24_2041_codex_hap1-affinity-control — Frozen MHC scores add modest development signal

- **Date / agent / reproduction:** October 24, 2026; Codex / GPT-6, Presto e85deef with recorded task dirt and frozen additions. [Report, score/member outputs, checkpoints and predictions](2026-10-24_2041_codex_hap1-affinity-control/README.md). Prospective six-arm plan, exact qualification/fit/closure commands, source/model/data hashes and runtime environment retained. Local CPU / one Torch-BLAS thread, no Modal/GPU/installations.
- **Data / endpoint:** unchanged Shapiro2025 PXD056426 source88cad10d… / Hitlist recipea1eec392…; original complete-precursor outer-training87,260positive KO/WT response cells,6,317peptides,1,798components,ten known conditions. Original WT/B2M/precursor/origin policies; no eligibility change, imputation, generated HAP1 labels or inferred bounds. Assay→one continuous log10 elution-response output; other assay/binder metrics inapplicable. Original five component folds reused; old outer validation/test remain closed and no new test predictions/metrics are produced.
- **Frozen representation / overlap:** installed clean MHCflurry2.3.14 sourcecd141c8 / 2.3.0 affinity component, eleven selected networks, three supplied HAP1 alleles, all18,951predicted nM scores log10-transformed. No best-allele assignment, processing/presentation score, percentile or neural update. Training includes affinity, reassigned MS, synthetic negatives and teacher pretraining; scores are empirical covariates, not measured Kd/occupancy. Bundled811,134training rows (239,021affinity/572,113MS);4,925HAP1peptides occur in it. No searched HAP1 source marker found, but complete teacher/selection/study/specimen lineage is unavailable. No independent or pure-affinity claim.
- **Training / objective:**36closed-form fits, six arms × full/five folds, original fitting-only precursor gamma/condition offsets, weighted standardization/condition centering, condition-normalized MSE+0.1sum(beta²). Original61features, length+three scores, sequence+scores, triplet-shuffled scores and secondary four-geometry extension. Primary64columns/610supported nonzero condition-feature coefficients; other Presto heads untrained. No new stochastic optimizer/RL comparison; this establishes a stronger observation-related control before further matched physical training.

| Fixed arm | Training RMSE | Inner-CV RMSE | CV R² |
|---|---:|---:|---:|
| calibrated_precursor | 0.637901 | 0.638745 | 0.0591 |
| length_affinity | 0.598078 | 0.600516 | 0.1684 |
| sequence | 0.554784 | 0.574832 | 0.2380 |
| sequence_affinity | 0.542454 | 0.562644 | 0.2700 |
| sequence_affinity_geometry | 0.536652 | 0.558574 | 0.2805 |
| sequence_shuffled_affinity | 0.553179 | 0.575043 | 0.2374 |

- **Decision:** primary development gate **PASS**:2.120%CV improvement versus sequence (required2%),2.156%versus shuffle (required1%),all five folds improve (requiredfour). Nine of ten condition strata improve. Score-only does not replace composition features; secondary geometry adds0.723%descriptively. Use sequence+scores as a development control; preserve earlier held-out reference and do not promote an independent biological/availability claim.
- **Exposure strata:** listed peptides70,963cells:RMSE.565804→.554473 (2.003%gain);1,392peptides not found in bundled table16,297cells:.610611→.595117 (2.537%). The latter is not certified absent from teacher/selection training. Strata retain and renormalize original weights. No independent-study confidence interval or test reuse.
- **Verification / resources:** all member geometric means agree≤2.22e-15log10, reversed/rebatched probes≤1.28e-7 against1e-5gate, model/source hashes unchanged. Scores3.80s; qualification13.59s/1.182GiB. Fit pipeline49.42s/1.448GiB; reference replay≤4.00e-14. Independent training-exposure and SVD replay:360condition systems/1,047,120predictions max1.48e-13, coefficients1.75e-14, archived metrics4.00e-15 and174strata1.78e-15;39.29s/1.764GiB. Prespecified identical-input-tie audit resolves raw Spearman differences≤4.18e-8 without a failed gate or result change. Default isolated Ruff/100-column format and visual review pass; expanded E/F/I reports seven preserved style findings (`results/code-review.json`); local single-run timing, not hardware ranking.
- **Next / limits:** qualify stage-appropriate processing/transport and selection separately, retain this frozen development control, require a new biological population for generalization. Original study prediction provenance remains under Hitlist726. Intracellular APC concentrations, direct kinetics and independent observation calibration remain unvalidated; full goal active. Recipe2.0.28 release40129209653 succeeded; published wheel/sdist bytes match tested artifacts (tasks/modular_model/lymph-composition-publication.json).

## 2026-10-24_2106_codex_hap1-tap-route — Shared delivery-response model misses matched feature control

- **Date / agent / reproduction:** October24,2026; Codex/GPT-6, Presto59a8db6 with recorded unrelated task dirt and frozen experiment additions. [Full report, plan, checkpoints and predictions](2026-10-24_2106_codex_hap1-tap-route/README.md). Exact launch/closure commands, source/model/data hashes and environment retained. Local CPU, float64, one Torch/BLAS thread; no Modal/GPU/installations.
- **Data / endpoint:** unchanged Shapiro2025 PXD056426 source88cad10d… / canonical recipea1eec392…; original complete-precursor outer-training rows restricted prospectively to supplied TAP1/TAP2 KO.9,168positive response cells/3,110peptides/1,248components; TAP1 7,006cells/3,071peptides, TAP2 2,162/949. All support gates pass. Original source/background/WT/parent/precursor policies; no imputation, score filtering, new negative or synthetic biological label. Assay→log10 positive KO/WT elution response; IC50/500nM/other assay families inapplicable. Original five component folds reused with equal-condition/component/peptide/replicate weights. Outer validation/test remain closed; no new test predictions/metrics, because those sets were inspected in1503.
- **Representation / pretraining:** shared route receives105raw length/composition/P1/P2/P3/C/all-parent geometry inputs; MHC scores enter only a separate condition-specific observation term. Reuse2041 frozen MHCflurry scores for all3alleles, no new neural inference/training.2,474peptides occur in bundled predictor training table;636notfound does not imply teacher/sample independence. Affinity/MS/teacher mixed supervision remains explicit. No other Presto branches trained.
- **Model / loss:** rho=sigmoid(shared linear sequence/geometry), epsilon=sigmoid(per-KO eta), two path weights1-rho and rho*epsilon, relative deliveryR=sum. Outputgamma*precursor+profiledKOoffset+log10R+optional3score covariates. Gamma fitting-only calibrated precursor. Fixed weightedMSE +0.01(sum(theta²)+mean((eta-logit(.05))²))+0.1mean_KO(sum(beta²));250Adamsteps/lr.03/seed42, theta/beta0 andepsilon.05. Two direct regressions use original fixed0.1ridge; expanded ridge has identical105+3features. Five arms×full/five folds=30fits.
- **Sampling contract:** exact-forward conditional-score hybrid with64fresh posterior-route samples/row/update; analytic observation/penalty gradients.11.46million binomial count variates represent733.44million conditional Bernoulli draws without materializing each path. No clipping, group scaling, reuse or referenceKL. This is not simulator-onlyREINFORCE/GRPO or full cleavage trajectories; one seed is not an optimizer-variance study.

| Fixed arm | Train RMSE | Inner-CV RMSE | CV R² | CV Spearman | Full+5fold seconds |
|---|---:|---:|---:|---:|---:|
| Original sequence+scores |0.686770|0.720640|0.3030|0.5424|0.145|
| Expanded matched-feature ridge |0.650475|0.707492|0.3282|0.5637|0.166|
| Shared route alone, exact |0.709329|0.740153|0.2648|0.4893|3.527|
| Shared route+scores, exact |0.681972|0.711942|0.3198|0.5462|2.927|
| Shared route+scores, sampled |0.681925|0.712170|0.3193|0.5457|4.011|

- **Decision:** primary biological development gate **FAIL**:1.207%gain over original (required2%),-0.629%versus expanded (required1%),2/5folds improve (required4). Expanded ridge improves1.824%over original for this subgroup; retain as development comparator, no held-out promotion. Exact/sample RMSE differ0.032%; sampled1.37×runtime. Both exact/expanded beat original inTAP2, but expanded is better inTAP1. For636unlistedpeptides, all new arms worsen original; preserve this adverse stratum. Stop this bounded family instead of tuning.
- **Training / verification:** all109score-augmented trainable coordinates receive gradients; profiled Jacobian rank104/109, exact finalgradient5.71e-7 and last50stepobjective change1.86e-10relative. No global-optimum claim. Predictions vary (CVSD.464vs target.863). Qualification1.85s/.599GiB; fullfamily12.81s/.796GiB. Independent source/feature/SVD/analytic replay checks30fits/91,680final predictions (max6.23e-14),825,120history predictions via continuous metrics, latent outputs, offsets, all65strata and analytic gradients (8.05e-16);2.52s/.543GiB. Default isolated Ruff/100-column formatting and visual figure review pass. Single local timings, not hardware benchmarks.
- **Qualification coverage correction:** original epsilon=1/monotonicity checks evaluated only reference algebra, though finite-grid production checks passed before fits. Post-fit frozen amendment calls production boundary and sampled-forward functions; passes<=5.56e-17. Original qualifier/logs retained; not retrospectively labeled a complete prospective boundary gate. Model, outcomes and decisions unchanged.
- **Full-goal implication:** arbitrary peptide-specific baseline amount cancels from this ratio endpoint. Effective shared-route parameters neither identify baseline production nor establish concentrations; TAP also has loading-complex effects. Preserve same-study/positive-selection/precursor/provenance limits. Next work must constrain baseline quantities and selection/observation separately and qualify prospective biology. Full APC availability goal remains active and unachieved.


## 2026-10-24_2130_codex_mapp2025-source — Recover baseline MAPP quantities and expose source-selection/identity limits

- **Date / agent / reproduction:** October24,2026; Codex / GPT-6, registered at35a2b972 with recorded unrelated task dirt. [Full source audit and exact reproduction](2026-10-24_2130_codex_mapp2025-source/README.md). Original/final executed scripts, recovery notes, source hashes, read-only workbook/XML checks and local CPU resource metrics retained. No cloud/GPU/environment changes.
- **Source / curation:** Goldberg2025 DOI10.1038/s41586-025-08615-w/PXD052314. Official publisher48,110,489byte supplement with10members; original deposition table/community SDRF; PRIDE41file inventory; original356,132,312byte MAPP processed archive. No RAW spectra, no activity-ranked selection or pathogen-response model. Basal captured MAPP, whole-cell proteome, proteasome-associated proteins and extracellular secretome remain separate endpoints. Numeric zeros are not negatives; full reference/material precision is unresolved.
- **Quantity coverage:** Table5 has21,082unique MAPP peptides,12intensity columns;6control columns have79,025positive cells/17,282peptides with any signal/8,025positive in all6. Positive baseline lengths7–40;48,885MS/MS and30,140matching observations. Whole-cell5,387protein-group rows have29,037positive control LFQ cells; no exact aliquot pairing is asserted. Secretome963rows/16columns/2,087positive cells is extracellular. Table3's199,281rows contain3,095,695numbers embedded in Treatment strings (1,773,019positive),141labels/33ordered patterns;197,785rows repeat labels. Numerical existence does not establish run identity or quantity semantics. Preserve202date-styled gene cells without repair. Initial header-only commentary was corrected before admission.
- **Original-source reconciliation / adverse findings:** all21,082publisher IDs/sequences match deposited30,364rows. Ten exact-name columns/210,820cells, including all126,492baseline cells, agree exactly; two nonbaseline replicate suffixes disagree and remain unresolved. Publisher omits9,282rows, including6,817without reverse/contaminant flags. Source-header organism assertions identify6,633omitted unflagged human-only rows (4,862with baseline signal) and130published nonhuman-only rows (113with baseline signal); mixed/unresolved rows retained. Search parameters name a human-plus-bacterial FASTA absent from archive, conflicting with human-only project description. Two truncated taxonomy tokens remain unresolved. This is source-assertion auditing, not independently validated organism classification or a supervised eligibility policy.
- **Metadata:** original deposition table matches all36RAW files/three assay groups. Community SDRF calls all assays tryptic shotgun, gives biological-repeat1/4 and replaces actual5566with nonexistent5582. Preserve conflicting assertions and distinguish biological-repeat labels from exact specimen/aliquot pairing. Older MAPP/PBMC work remains owned by Hitlist820–823.
- **Training / evaluation:** no model, pretraining, optimizer, synthetic biological labels, loss or assay-output mapping. No validation/test split/prediction dumps because source qualification precedes modeling. Regression/binder metrics inapplicable; no old outcomes reopened or model promoted.
- **Verification / runtime:** independent OOXML agrees on3,822,731populated cells; split/regex decoding agrees on all embedded numeric tokens. Workbook24.52s/.165GiB, XML22.38s/.682GiB, archive reconciliation6.68s/.389GiB; metadata/supplement97.26s and processed archive539.59s retrieval. Original failures retained: sandbox DNS, datetime serialization, secondary metadata mismatch, two archive headers, CSVfield ceiling and truncated taxonomy assertions. Corrected parser bounds/serialization preserve source quantities. Current entrypoint lint/100-column format pass; final executed snapshots precede publication-only style changes. Reviewed log archives and exact artifact manifests retained.
- **Decision / next:** source quantity availability passes; a clean human-only supervised cohort is not yet admitted. [Hitlist825](https://github.com/pirl-unc/hitlist/issues/825) requests canonical original-source quantities/dispositions, search-reference/selection recovery, scoped material links and pooled-table identities. Use the reconciled supplement as a view, not an assumed complete curated cohort. Next admit the shared source and freeze grouped baseline-quantity exact/sampled comparisons. No predictive improvement; APC availability/concentration objective remains active and unachieved.


## 2026-10-24_2210_codex_mapp2025-canonical-admission — Original baseline MAPP and candidate decisions independently verify

- **Date/agent/reproducibility:** October24,2026; Codex/GPT-6. [Full report, source/recipe manifests and exact launch/replay](2026-10-24_2210_codex_mapp2025-canonical-admission/README.md). Presto a8d5ff0 with unrelated task dirt; isolated Hitlist9583aeb plus frozen source/candidate implementations, candidate2.0.30. All original assets reused without download. Local CPU/one numerical thread; no Modal/GPU/installations.
- **Source contract:** six pinned Goldberg2025/PXD052314 original assets. Source schema25/catalog31;16typed relations retain30,364peptides/32,636forms/294,756identification rows and1,113,756separate intensity cells. Original acquisition/arm/project contexts use the common material graph; no specimen/aliquot pair is invented. Original/community assertions, source taxonomy ambiguity, matching/confidence/flags, zero/missing values, all parents/contributors and distinct assay pools remain queryable.3,851,119literal publisher cells and3,095,695ordered pooled slots retain unresolved identity/units and202date styles. Publisher exact-run/aggregate reconciliation231,902cells;42,164cells have unresolved column names, not guessed aliases.
- **Candidate contract:** original control acquisition/captured-peptide scope; positive finite canonical sequence, explicitly blank flags, human-only source header annotations. No publisher-selection filter, numerical biological imputation, new FDR guarantee or missing-negative label.394,732original peptide-cell decisions:182,184baseline,94,034review candidates,88,150excluded baseline,212,548outside scope.19,058candidate cells are omitted from the publisher view. Confidence remains separate:50,921direct-MS/MS and20,919matching cells havepeptidePEP<.01;22,194candidates havePEP>=.01. Full-reference support/physical precursor pairing/confidence policy/processing eligibility remain explicit independent axes. No split or fitting.
- **Verification:** pinned-byte replay reproduces every typed relation; independent readers check23,721,081original TSV fields, allsearch quantities/publisher cells/pooled slots, material precision and all394,732candidate decisions. Source build+internal replay190.29s/1.80GiB; independent source41.91s/1.93GiB; candidates+source/recipe replay120.78s plus3.83sindependent audit/2.32GiB. Combined source/candidate/audit1,385,928,868logical bytes; all6GiB/20minute/10GiB gates pass. These are single-run observations, not hardware comparisons.
- **Software checks/failures:**9focused importer/recipe tests,26earlier source/context/query tests, format/lint and3,489non-integration tests pass. Initial undeclared catalogfield and empty-versus-absent XML fixture expectation fixed before compilation. Seven old globalcatalog30assertions updated to31after the first full-suite failure; no weaker gate. Sandbox memory-probe refusal preserved; host script retains the guard. Full-corpus integration/release remain pending. Executed source snapshots and reviewed log archives remain unchanged; canonical launcher publication-only style correction is disclosed.
- **Training/evaluation:** no model, pretraining, synthetic biological outcome, optimizer, loss or assay-head output. No validation/test split or predictions because this is source/candidate qualification. Existing biological held-outs remain closed; no baseline promotion or predictive improvement claimed. nM/binder metrics inapplicable.
- **Decision/next:** shared source/candidate interfaces now work locally. [Hitlist827](https://github.com/pirl-unc/hitlist/pull/827) is a published draft at176a62e; CI, merge and verified release remain pending. [Hitlist825](https://github.com/pirl-unc/hitlist/issues/825) remains open for source reference/selection/material/pooled-run recovery. [Next prospective comparison](agents/codex/plans/mapp2025-baseline-prediction.md) tests simple sequence versus length/uniform captured-composition prediction, then matched exact/REINFORCE with fixed features/objective. Full APC availability/concentration goal remains active and unachieved.


## 2026-10-24_2244_codex_mapp2025-baseline-prediction — Modest original MAPP signal; exact and RL beat uniform

- **Date/agent/reproduction:** October24,2026; Codex/GPT-6, Presto98d0ac8 with recorded task dirt and new frozen code. [Full report, exact launch, all predictions/checkpoints/metrics](2026-10-24_2244_codex_mapp2025-baseline-prediction/README.md). Isolated Hitlist2.0.31 development on176a62e; source827merged5a8b1aa. Local float64CPU/one numerical thread; no Modal/GPU/installations.
- **Data:** original Goldberg2025/PXD052314 six A549 baseline control acquisitions, canonical sourcecb1bbc22… and shared `goldberg2025_observed_baseline_composition_v1`. Positive finite canonical unflagged source-human-only, PEP[0,.01), direct-MS/MS, one supplied accession, >=3distinct observed peptides/group. All-source peptide/accession graph includes excluded bridges before fixed60/20/20hashsplit.40,701measurements; train/val/test274/74/110components,1330/372/649groups,19593/8649/12459cells. Original394,732decisions retained; no publisher filter, imputation, synthetic biological labels, missing negatives or censor/binder interpretation. Conditional observed captured signal, not complete free pool/molarity.
- **Training/output:** peptide61features→training-only weighted scaling/group centering/SVD→58supported coordinates→scalarlog10signal; group softmax gives fractions. Equal component/group/peptide errors. Length rank1andsequence ridgealpha.001/.01/.1/1selected onlybyvalidation (.001/1). Exact reverseKL(q||p)+.01mean(beta²),500full-batch Adamstepslr.02,betas.9/.999,eps1e-8,zeroinit/finalcheckpoint. Matched genuine discreteREINFORCE64fresh observed-identity draws/LOO rewards,seeds42/43/44,same objective/budget; no clip/std/reuse/referenceKL/pretraining. All58coordinates train/change; no other Presto assay/processing/MHC/TCR/species/KO/detector branches train.

| Fixed model | Validation RMSE | Test RMSE | Validation reverse KL | Test reverse KL | Test R² | Test Spearman |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 0.419737 | 0.439892 | 0.431589 | 0.477545 | 0.00000 | undefined |
| length_ridge_0.001 | 0.410127 | 0.434413 | 0.410675 | 0.464617 | 0.02476 | 0.12796 |
| sequence_ridge_1 | 0.388039 | 0.415585 | 0.378544 | 0.424381 | 0.10746 | 0.29551 |
| exact | 0.388470 | 0.415095 | 0.376448 | 0.422502 | 0.10956 | 0.29363 |
| reinforce_42 | 0.387946 | 0.415376 | 0.375926 | 0.422539 | 0.10836 | 0.28992 |
| reinforce_43 | 0.389502 | 0.416381 | 0.378255 | 0.425052 | 0.10404 | 0.28877 |
| reinforce_44 | 0.388821 | 0.414296 | 0.376133 | 0.421299 | 0.11299 | 0.29057 |

- **Decision:** primarysequence-versus-lengthFAIL:5.386%validation/4.334%testRMSE gain, required5%both; testpaired110componentbootstrap95%[2.288%,6.643%]. Both exact and allRLseeds passdistribution-versus-uniform onbothheldouts. ExacttestreverseKL.422502vsuniform.477545;RLmean.422963±.001912. ExacttestRMSEgainvsuniform5.637%,95%[3.220%,8.118%]. Modest nonconstant signal; no clearRLadvantage, noAPCconcentration/fullavailabilityclaim, no test-selected promotion or refit. Exact remains practical bounded distribution reference.
- **Verification:** allsourcebaseline decisions match prior independently verified generation; independentNetworkXgraph/targets/weights max3.55e-15. Three actual-cohort probes: analytic/autograd3.11e-15,finite differences2.08e-9;samplegradientmeanrelative.00326–.00370/max2.79SE. Allcheckpoints/covariatepredictionssealed beforeonce-onlytest. Independent58-coordinateSVD/ridge/forward replay529,113predictions(max1.18e-14),39overallmetricrecords(max1.22e-15),6,188strata/ranks andbootstrap(max2.16e-16). Optimizertrajectoriesretained,notindependentlyreplayed. AllpredictionspublishedlosslesslyasParquet;rawgzipCSV/localinputpathsandhashesretained.
- **Runtime/software:** admission146.22s/2.185GiB;dataaudit5.01s;qualification7.78s;allfits19.31s/.846GiB;eval19.08s;replay11.26s. Exact3.08s/fitvsRL4.30–4.34s(~1.41×);19,593,000candidate scores/fitincludinglogging,42,560,000draws/RLfit. No score-saving claim;single-runlocaltimings.13focusedtests/format/lintpass;firstfullsuite16disk-reservefailures/3477passesduringtemporarysourcebuild,unchangedguardedrerunpasses3493tests/392warnings/314.19s. RecipePR829published5d648602afterrebaseonmergedsource/PBMCmain;fourruntimechangedfilesbyte-identical,post-rebaselocal3481passes/15setupdisk-guarderrorsat~.84GiBfreevs1GiBreserve(309.60s),unresolved;hostedCI40140838953passes all unit/integration/lint/docs/coverage checks; recipe829merged58f67cd. Source2.0.30PyPI wheel/sdist byte-match successful clean-main40140479531; recipe2.0.31release40143097533pending. Canonicaladmit/launcherpublicationimportorderingonly;executedsnapshotsunchanged.
- **Limits/next:** samecohort/runs and unknownphysical/homologyindependence; totalsmayaggregateforms;capture/detection/processingconfounded. Postfitprovenanceauditfinds3/2/2testpeptidesinpriorMAPP/HAP1/lymphDC; no previousoutcomesorweightsused. Literalaccessionformatdiffers;directfieldextractionfinds134testaccessionsinHAP1referenceinventory,notprovenpriortrainingmembership. Testclosed. Qualify independent baseline biology and observation/processing constraints; Hitlist825remainsopen. Fullgoalactive/unachieved.


## 2026-10-24_2315_codex_pbmc-frozen-transfer — PBMC support fails before frozen-model scoring

- **Date / agent / reproduction:** October 24, 2026; Codex / GPT-6, Presto c7fa668 with recorded task dirt. [Report, original-source audit, all cohort decisions and exact launch](2026-10-24_2315_codex_pbmc-frozen-transfer/README.md). Prospective source/model/curation contract, frozen adapter and recipe runtime, local CPU / one numerical thread; no cloud/GPU/environment changes.
- **Source / endpoint:** original PXD009438 seven pinned assets and verified search archive, 985 peptide rows / 8,851 evidence rows / 14 acquisitions / 13,790 reported intensity cells. Source generation63752e02; shared `pxd009438_observed_baseline_composition_v1`, candidate Hitlist2.0.32 on5d648602. Positive finite canonical explicitly unflagged, peptide PEP[0,.01), per-run By MS/MS, positive direct evidence, complete linked intensities, one supplied parent; exclude every ambiguous duplicate and require three sequences per group. All-source sequence/parent graph includes excluded bridges. CTRL/LUPUS separately weighted equal component/group/peptide. Targets are centered log10 captured intensity and positive observed-set fractions; no synthetic labels, missing negatives, imputation, censoring or binding metrics.

| Source cohort | Components | Accession/run groups | Measurements | Unique peptides | Required support |
|---|---:|---:|---:|---:|---|
| CTRL primary | 6 | 27 | 234 | 82 | **Fail** |
| LUPUS secondary | 7 | 20 | 161 | 66 | **Fail** |

- **Decision:** both fail the prospectively fixed20component/20group/100measurement gate. No prediction or fitting follows; no weakened threshold, regrouping or test adaptation. This is a qualification failure, not measured model-transfer performance. The existing A549 gain remains within-study evidence. Components are dependence groups, not verified independent specimens.
- **Training / evaluation contract:** no new pretraining, updates, optimizer/loss execution or model selection. Seven predeclared A549-frozen uniform/length/sequence/exact/REINFORCE predictors and their hashes remain unchanged. External-test-only intended; no train/validation split and no prediction dump because neither cohort passes support. Other Presto branches untrained; predictive regression/ranking/binder metrics are inapplicable to this unscored family.
- **Verification / resources:** original CSV independently verifies692,452fields; NetworkX/source/evidence replay verifies all13,790decisions,1,756nodes and395targets/weights(max1.78e-15). Source rounding discrepancy6.50e-5retained; all14literal aliases and label projections independently agree. Source62.66s/.897GiB; admission17.16s/.720GiB; audit.83s/.258GiB; closure.35s/.079GiB. All resource caps pass; source uses verified CoW, no new multi-GB physical copy.28focusedtests/format/lint pass; fixture reciprocal-ID and two lint corrections precede admission. Full local suite deferred below2GiBfree-space gate; [Recipe PR830](https://github.com/pirl-unc/hitlist/pull/830) published1619c37 with identical frozen runtime; CI/merge/release remain unfinished. Source2.0.30 PyPI bytes match successful clean-main build40140479531.
- **Limits / next:** project human scope does not establish per-parent origin; original full FASTA absent. Publication PBMC versus repository serum conflict, mixed-cell context, specimen/aliquot/precursor uncertainty, molecular-form aggregation and capture/detection remain explicit. No clinical inference or APC concentration claim. [Shared material-context follow-up820](https://github.com/pirl-unc/hitlist/issues/820#issuecomment-6390908317) records gaps; source822 stays legitimately closed. Require a prospectively qualified independent cohort before scoring transfer. Full availability goal active/unachieved.


## 2026-10-24_2338_codex_intact-degradomics-source — Original kinetic quantities reconcile with source exceptions

- **Date / agent / reproduction:** October 24, 2026; Codex / GPT-6, Presto1de0a12 with preexisting task dirt. [Full report, frozen commands/source, cell/quantity audits and receipts](2026-10-24_2338_codex_intact-degradomics-source/README.md). Local CPU, one numerical thread; no cloud/GPU/installations.
- **Source / curation:** PXD058121 / DOI10.1073/pnas.2419607122: human alpha-synuclein degraded by mouse-organ-derived purified20S. All organs are from one animal; three reported replicates have unresolved biological independence. Complete105-file PRIDE inventory, three processed and five supplementary XLSX, SI PDF/XML/metadata. No RAW spectra, imputed biology, absent-negative labels, molar conversion or model-based assignment choice. Source qualification is not a supervised recipe.
- **Quantitative coverage:** Eight workbooks /133sheets /325,718 populated cells. KineticS4:1,289 positive rows/237sequences/21labels, lengths5–140; organS5:2,138/313/44, lengths4–140. Intact substrate included. All3,427rows resolve to original identities;3,426intensities match exactly. Search workbooks contain17,862input masses/8,138assignment rows before duplicate views. All nine commonF1/F2sheets agree and must not count twice.
- **Exceptions / context:** Extra unlabeled H field shifts F2,40minuteR1 assignments; F3brain-zeroR2input IDs are absent despite its header. Explicit layout interpretation restores source-span/value agreement without modifying originals. Spleen30minuteR3is absent from both processed and RAW inventory. One S4/deposit mass/error/intensity disagreement, one reused absent input ID and three assignment/input intensity mismatches remain. Numeric machinery densitometry/material links not found. Nominalzero already contains products; raw deconvoluted sums differ from TIC/normalized plots. All source confidence/ambiguity/recurrence and unknown MS-response scales remain separate.
- **Verification / runtime:** Independent OOXML/openpyxl replay verifies every populated cell and2,991stored formula caches; source hashes, CRCs and advertised sizes agree. Final positional discrepancies zero. Retrieval4,718,213bytes/44.07active seconds including failed bundle timeout; cell audit9.11s/84.94MiB, final quantity audit1.17s/64.88MiB, closure.013s/53.41MiB. All1GiBfree-disk/3GiBRSS/20minute/256MiBdownload gates pass. Original wrong-route empty response, formula-reader failure, abbreviated header and superseded header-driven findings retained; final scripts pass isolated lint/format. SI figures/captions visually inspected; no plot digitization or RAW reprocessing.
- **Training / evaluation:** none: no pretraining, optimizer, loss, synthetic biological targets, assay-head mapping, validation/test partition or prediction dumps. Source qualification precedes shared admission; regression/binder metrics inapplicable. Prior biological held-outs remain closed and no predictive baseline changes.
- **Decision / next:** original kinetic quantities are available with explicit exceptions. [Hitlist831](https://github.com/pirl-unc/hitlist/issues/831) owns homogeneous source ingestion, conflicts, missing acquisition/replicate/material and machinery values. Admit/replay through shared APIs before a separately registered matched exact/sampled generation-destruction comparison. One substrate/purified preparation does not establish APC concentrations or new-protein transfer; full goal remains active and unachieved.


## 2026-10-25_0021_codex_intact-canonical

**Date/agent/model:** October25,2026; Codex/GPT-6.
**Directory:** [canonical intact-degradomics admission](2026-10-25_0021_codex_intact-canonical/).
**Status:** source build, independent public-query replay and prospective support
census complete; no biological fit or prediction. Shared source[Hitlist832](https://github.com/pirl-unc/hitlist/pull/832)
is draft at7ece0885; full hosted gates/release pending.

**Data/curation:** original PXD058121 human alpha-synuclein/mouse-organ purified20S;
14pinned assets/eight workbooks. Source schema26/catalog32; standalone manifest
fd91bb6ff3178a2246b760f6b0f99bc3824d83815a65719c34587816f3459974.
All410,349XML cells/325,718populated values,2,991formulas,17,862input-mass rows,
8,138assignments and44,457quantities are retained. Independent original OOXML,
openpyxl and public joined-query checks pass. Publisher3,427links retain one
quantity conflict; nine repeated views remain non-independent. Unknown molarity,
initial product pool, physical specimen/replicate identity and mass-match
confidence are not imputed or promoted to negative/censor labels.

**Training/pretraining/synthetic/loss/output contract:** none; source qualification
only. Software fixtures are synthetic tests, not training labels. No affinity,
stability, MHC/TCR or other assay family trains. No validation/test split or
prediction dump exists because a separate kinetic model contract must precede
fitting. Binding/500nM metrics are inapplicable. Local CPU/one numerical thread;
no Modal/GPU/installations. Exact git state, runtime snapshots, invocation/env and
resource caps are in `reproduce/launch.json`, scripts and copied plan.

| Condition | Wall seconds | Peak polled RSS | Outcome |
|---|---:|---:|---|
| Build with deterministic producer replay |13.37|659.4MiB|Pass|
| Initial independent source replay |12.14|943.2MiB|Verifier omitted substrate-only control grammar; preserved failure|
| Explicit control-grammar amendment |12.25|946.7MiB|Pass; canonical bytes unchanged|
| Prospective kinetic support census |8.12|748.5MiB|1,278candidate cells/237sequences; intact parent in all21acquisitions|

**Decision:** source package can support a separately qualified kinetic recipe.
No winning predictive model. Unlike the MAPP/lymph composition targets, these
measurements include ordered substrate/product time courses and short products;
they remain purified-mixture signal, not APC concentrations.80sequences recur at
at least two early and one late time, enabling a prospective temporal forecast.
The next frozen plan compares direct controls with exact expected fragmentation
and conditional trajectory score gradients. Artifacts include every candidate
disposition, source/build/replay summaries, resource receipts and failed audit.
The full APC availability goal remains active and unachieved.

## 2026-10-25_0046_codex_intact-kinetic-forecast — Fragmentation loses to direct temporal regression

- **Date / agent / reproduction:** October 25, 2026; Codex / GPT-6, Presto `3ab625edce9ceb6aed9d94ee104306dcf84e248c` with recorded preexisting task dirt and new experiment code. [Complete report, frozen plans/commands/runtime, checkpoints and predictions](2026-10-25_0046_codex_intact-kinetic-forecast/README.md). Local macOS arm64 CPU, float64, one numerical thread; no Modal/GPU, pretraining or installations. Exact interpreter and `PYTHONPATH`/thread overrides are in `reproduce/launch.json`; every executed stage/import is frozen.
- **Dataset / curation:** PXD058121 F2 human alpha-synuclein/mouse-liver purified 20S kinetics, one animal and unresolved replicate independence. Canonical source manifest `fd91bb6f…`; shared Hitlist `pxd058121_product_substrate_forecast_v1`, recipe manifest `fb04bd48…`, development 2.0.34 on `97d3311a`. Original positive finite, explicitly unmodified, unambiguous assignment/span with <=10 ppm mass error, reconciled input intensity, no ambiguous duplicate mass, unique same-acquisition intact denominator. Product requires two fitting times; no late-presence selection. All 8,138 dispositions retained. Training 0–80 minutes: 626 observations/96 peptides; validation 100 minutes: 181/75; test 120 minutes: 186/74. Equal peptide/time/replicate weights. Same-peptide temporal forecast, not new-substrate or specimen transfer.
- **Assay / output:** measured `log10(product intensity / intact intensity)`. Missing values stay missing; no synthetic biological targets, imputation, generated negatives or inferred censoring. No affinity/stability/TCR/MHC/species/APC-perturbation head trains; IC50 and 500 nM metrics are inapplicable. Fixed peptide response is absorbed by 96 training-only nuisance offsets, so predictions are anchored to previously observed peptides rather than calibrated absolute abundance.
- **Models / training:** parent sequence, query span and time → independent-bond fragment expectation → product/parent log ratio + profiled peptide offset. Uniform rate+pre-exposure (2 coordinates) or residue-before-bond rates+pre-exposure (18); all receive gradients/change. All 9,870 latent spans represented mathematically; queried spans evaluated directly. Fourteen fits: exact for both models, plus conditional-score 16/64 draws × seeds42/43/44 for both. Fixed 800 Adam updates, lr0.03, betas(.9,.999), eps1e-8; weighted MSE plus .01 mean squared centered log rates for residue. Exact-forward conditional REINFORCE gradients use Binomial sufficient statistics, not simulator-only RL/GRPO; no clip/std/reuse/reference KL. Seven controls: peptide mean, last, shared linear slope, and slopes shrunk toward shared slope at .001/.01/.1/1. Validation-only selection; no test refit.

| Condition | Validation RMSE | Test RMSE | Test raw-ratio R² | Test raw-ratio Spearman |
|---|---:|---:|---:|---:|
| Last fitting-time mean | 0.229625 | 0.364187 | 0.503660 | 0.908959 |
| **Selected direct slopes, shrinkage .1** | **0.206085** | **0.228184** | **0.805149** | **0.909977** |
| Uniform fragmentation, exact | 0.304794 | 0.359391 | 0.516646 | 0.810736 |
| **Selected residue fragmentation, exact** | **0.304593** | **0.359038** | **0.517596** | **0.810438** |

- **Decision:** primary gate fails: physical validation/test RMSE is 47.80%/57.35% worse than selected direct regression, rather than the required >=5% improvement. Paired peptide bootstrap gain intervals [-62.29%,-34.70%] / [-86.69%,-34.34%] do not measure independent-specimen uncertainty. Positive raw-ratio correlations/variation pass but are insufficient: test change-from-last Spearman is .23071 for direct and -.13886 for physical; change R² is -.07353/-1.65781. Direct shrunk regression is the provisional comparator for this narrow contract, not an APC concentration predictor.
- **Sampled comparison / runtime:** uniform16/64 test RMSE .359376±.000074 / .359391±.000045; residue16/64 .358994±.000025 / .359020±.000027 (three training seeds, not biological uncertainty). Exact core fits2.49/2.50s; sampled4.52–6.36s,1.82–2.55× slower with no predictive advantage. All20pre-closure stages total137.53s; max polled RSS823,705,600B. Closure0.49s. No resource guard changed.
- **Verification / artifacts:** independent source admission verifies993targets(max8.88e-16), eligibility and weights. Production numerical gates pass. Independent NumPy/SciPy reconstruction verifies13,902physical predictions+sevencontrols(max1.38e-13), all126overall metrics(max1.34e-15), every optimizer recurrence, and all800exact gradients per exact fit. Stochastic draws are not independently replayed. Closure reconstructs selection/bootstrap. All801checkpoints per fit, RNG/optimizer states, per-example validation/test dumps, strata, complete-span dumps for exact fits, plots and artifact manifest remain.32focusedsharedrecipe tests pass; full local suite deferred below2GiBfree-space prerequisite. Source/recipe release and publication remain separate unfinished work.
- **Context / next:** sequence baselines on HAP1/MS and modest MAPP observed-set gains do not establish processing dynamics or unobserved peptide mass. This biological test finds no mechanistic advantage despite correct tested gradients. Independent-bond assumptions, initial mixture and MS response remain candidate limitations, not diagnosed causes. Preserve inspected100/120-minute outcomes; require a fresh prospective evaluation for new model selection. Full APC availability goal remains active and unachieved.

## 2026-10-25_0116_codex_intact-shape-diagnostic — Optimization and curve shape explain little of the training miss

- **Date / agent / reproduction:** October25,2026, Codex/GPT-6, Presto3ab625e with recorded task/report dirt and new experiment-local code. [Complete report, frozen specification/inputs/invocation, all training predictions and checks](2026-10-25_0116_codex_intact-shape-diagnostic/README.md). Local macOS arm64 float64CPU/one numerical thread, no Modal/GPU/installations.
- **Data / curation / output:** unchanged0046 canonical PXD058121 training626measurements/96peptides, times0–80min, humanalphaSyn/mouseliverpurified20S. Same positive unambiguous source/intact-normalizer policy and equal peptide/time/replicate weights. Only measured log10product/intact intensity ratios train. No new source curation, synthetic biology, imputation, negative/censor labels or molarity. Affinity/stability/TCR/MHC/species/APC/MS heads and500nM metrics inapplicable. No pretraining. Validation/test files are intentionally unread: this is diagnosis after0046outer outcomes were inspected, not a new predictive evaluation. Per-example training dumps replace held-out dumps for this explicitly limited scope.
- **Hypotheses / conditions:** production independent-bond log ratios must increase concavely; fit descriptive free/increasing/increasing-concave curves per peptide by exact small active-set weighted quadratic programming. Separately reoptimize unchanged uniform2-coordinate/residue18-coordinate objectives from three frozen starts each (archivedAdam,original,lowrate). L-BFGS-B maxiter1500/maxfun3000/maxls40,gtol1e-9/ftol1e-13,rawbounds[-30,30]; original positive rate/tau parameterization and96profiledtrainingoffsets preserved. WeightedMSE plus.01mean centeredlograte² forresidue, no uniform penalty. No new RL run or held-out scoring; all conditions retained.

| Descriptive / physical training condition | Weighted MSE | Interpretation |
|---|---:|---|
| Free timepoint means |0.009991|Within-time replicate dispersion, not pure noise|
| Increasing curves |0.012096|Per-peptide shape bound|
| Increasing concave curves |0.013600|Per-peptide physical-shape bound, greater flexibility than shared model|
| Original uniform/residue Adam |0.045230 /0.045186|Unchanged0046reference|
| Reoptimized uniform, all3starts |0.044194775|2.2893%training gain|
| Reoptimized residue, all3starts |0.044136468–0.044136469|2.3219%training gain|

- **Decisions:** both optimizer gains miss fixed5%materiality threshold; stop optimizer extensions on this cohort. Shape penalty is7.98–7.99%of originalphysicalMSE, below fixed10%threshold.47/208adjacent empirical changes and16/96slopes are negative, without causal interpretation. All optimized tau approach120min upper bound, not evidence for actual pre-exposure. Shared-rate coupling/features/regularization are the next representation hypotheses; separate flexible curves are not generalizing biological models. No baseline promotion or APC concentration success.
- **Verification / runtime:** actual production analytic/autograd shape derivatives and constant/increase/decrease/peak solverfixtures pass. Independent PAVA/SLSQP predictionagreement6.67e-16; convexKKTresidual9.72e-17. IndependentNumPy reconstructs3,756optimized predictions(max3.55e-15), all246functionobjectives(max1.71e-13), finalfinite-differencegradients(max6.22e-11), shapeMSE anddecisionthresholds. All allocated optimizercoordinates change withnonzerocumulativegradients; L-BFGSinternalupdatesnotindependentlyreplayed. Sixcorefits0.052–0.099s/25–48iterations; fullstage2.04s/344,981,504BpeakpolledRSS;replay0.87s/139,100,160B;closure0.049s. All resourcecaps pass. Every evaluation/acceptediterate/gradient, trainingprediction, summary, logs andhashmanifestretained.
- **Next / full scope:** quantify sequence representation and shared-rate coupling on training development before a fresh prospective biological evaluation. Do not relabel inspected100/120min asblind or invent MS/initial-pool calibration. Prior exact-versus-sampled biological failure stands; full precursor-relative APC availability/concentration goal remains active/unachieved.

## 2026-10-25_0143_codex_intact-temporal-representation — Sequence slopes improve development predictions; penalty removal does not

- **Date / agent / reproduction:** October 25, 2026; Codex / GPT-6. [Report, frozen plan, invocation, all predictions and independent replay](2026-10-25_0143_codex_intact-temporal-representation/README.md). Registered at `d67fb48488960882481627c6f3a8bdaeb6e58bd3`; actually executed at `3bb3d62e96bd2e989f33a3d4cfc382068101c1e6` with recorded task/cleanup dirt. Per-stage receipts record actual git/package/environment state; every executed source is frozen. Local macOS arm64 float64 CPU, one numerical thread; no Modal/GPU, environment changes or pretraining. OMP/OPENBLAS/MKL thread overrides are 1. The previous prepare preflight refusal remains; execution began only after verified cleanup restored the original reserve.
- **Dataset / curation / output:** only the 0046 original training cohort, 626 positive unambiguous product/intact measurements from 96 peptides at 0–80 minutes; PXD058121 F2, human alpha-synuclein with purified mouse-liver 20S, one animal and unknown replicate independence. Same canonical `pxd058121_product_substrate_forecast_v1` recipe, manifest `fb04bd482f5902d776bcd853d5280d86df7fd8b74424574c1bf793e3d2332f0c`; source/input hashes fixed. First-time replicate means become supplied baseline measurements, yielding 463 later log10-ratio changes. Baseline IDs/values/shared-noise dependencies are retained. Equal peptide/time/replicate weights, recomputed per split. No imputation, generated negatives, censoring or synthetic biological labels; no affinity/stability/TCR/MHC/APC/detector heads train. IC50/nM/binder metrics are inapplicable.
- **Evaluation:** five deterministic peptide folds and three inner folds meet fixed support thresholds. Original 100/120-minute outcomes are unread. This is nested development CV on previously inspected training, with overlapping peptides from one parent; no new blind test or independent-protein/specimen transfer. Test absence is deliberate. Preserve all 16,205 outer training/validation predictions, 14,816 inner predictions and all physical-start training predictions.
- **Models / losses:** seven arms below. Direct slopes use elapsed time/80 times a shared intercept plus training-only projected features: log length; or length plus 20 residue fractions and 20 indicators at each terminus. Constant/null directions are removed at the preregistered covariance cutoff. Balanced change MSE plus alpha times mean squared nonintercept coefficients; inner peptide CV selects .001/.01/.1/1. All 120 inner fits, ten selected fits and five shared slopes retained; every selected alpha is 1, with sequence ranks 42–46. No identity or fitted prior slope feature. Physical changes subtract baseline-time from later-time independent-bond product/parent predictions; uniform two parameters or residue 18, with centered log-rate penalty .01 or 0. Three data-free starts per fold/arm, 45 total; fixed L-BFGS-B settings/bounds and lowest-training-objective selection. No new sampled arm: this isolates representation after the matched 0046 exact/conditional-score comparison failed similarly.

| Condition | Pooled development RMSE | R² | Pearson | Spearman | Prediction SD |
|---|---:|---:|---:|---:|---:|
| Persistence / zero change | 0.470580 | -0.693432 | undefined | undefined | 0 |
| Shared elapsed-time slope | 0.379389 | -0.100703 | 0.014003 | -0.006312 | 0.127201 |
| Length-conditioned slope | 0.379042 | -0.098690 | 0.008941 | -0.004936 | 0.128035 |
| **Sequence-conditioned slope** | **0.345792** | **0.085610** | **0.599177** | **0.599769** | **0.304500** |
| Uniform physical model | 0.386543 | -0.142609 | -0.012099 | -0.030995 | 0.149976 |
| Residue physical, penalty .01 | 0.386013 | -0.139474 | -0.008486 | -0.030941 | 0.152184 |
| Residue physical, no penalty | 0.380755 | -0.108643 | 0.045984 | 0.007165 | 0.165161 |

- **Decision / uncertainty:** sequence beats length by 8.772%, passes the >=5%/positive-correlation/nonconstant screen and improves four of five folds. Target SD is .361618. The 1,000-resample paired peptide bootstrap gain interval is **[-4.259%,20.109%]**, so this is a promising development comparator, not an established transfer benefit. Removing the physical penalty improves only 1.362% (interval [-1.465%,4.198%]), failing its 5% gate. All 15 selected physical fits converge and approach the 120-minute effective-pre-exposure upper bound; this is not evidence of literal pre-exposure or a diagnosed biological mechanism. No grid extension, test reuse or model promotion follows these observations.
- **Verification / runtime:** prepare 1.89s, qualify 3.42s, fit 13.69s (peak polled RSS 475,496,448B), evaluate 2.56s, independent replay 19.49s, closure .23s. All resource limits pass unchanged: 3GiB RSS, 20min/stage, 32MiB artifacts, 1GiB free plus32MiB startup margin. Independent SVD/augmented least squares and NumPy physical formulas replay 135 direct records/45 physical starts, objectives, final gradients, selection, all 749 overall/stratified metric records, bootstrap and gates. Prediction errors <=3.02e-14; objective error <=4.66e-9; final gradient error <=3.08e-10. L-BFGS internal updates are not independently replayed. All 231 manifest entries (14,464,690 logical bytes) verify after closure.
- **Context / next:** there is sequence-associated temporal signal that the tested shared kinetics do not recover, but one-substrate CV does not identify the missing mechanism. Keep the parsimonious direct comparator and unchanged physical/exact/sampled references; qualify broader measured substrates and fresh prospective transfer before changing biological scope. Any future exact/discrete comparison must share representation, objective, conditioning and observation model. The full precursor-relative intracellular/APC availability goal remains active and unachieved.

## 2026-10-25_0344_codex_proteasome-quantity-source — Recover native intracellular machinery-perturbation quantities

- **Date / agent / reproduction:** October25,2026; Codex/GPT-6, registered at Presto `8454ef9c40913c89d56bca5cd02f923701af91f6` with task dirt. [Complete report, source plan, exact commands, asset hashes and independent replay](2026-10-25_0344_codex_proteasome-quantity-source/README.md). Local Python3.12.6/openpyxl3.1.5/numpy2.5.4; no installation changes, Modal or GPU. Source-only qualification; no pretraining, optimization, loss, synthetic biological labels, validation/test split, model prediction dumps or regression/binder metrics. Test absence is deliberate before canonical admission and predictive registration.
- **Primary data:** Sahu2021 DOI10.1038/s41467-021-26427-0, PMC8548400, native intracellular Hi20S panel in PXD018722. Dataset7 (`41467_2021_26427_MOESM10_ESM.xlsx`, SHA256`f29811e0…`) has10,647distinct peptide sequences,35columns,6,202literal parent-accession tokens and six intensity columns. Human HEK293T control versus inducible PSMD2/Rpn1 knockdown, with primary-study intracellular extraction; not MHC elution, a professional APC, or the separate proteasome-trapped panel. No canonical training rows admitted yet.

| Source acquisition label | Positive intensity cells | Source zeros |
|---|---:|---:|
| Cont1 |3,866|6,781|
| Cont2 |4,181|6,466|
| Cont3 |4,043|6,604|
| KD1 |7,418|3,229|
| KD2 |6,968|3,679|
| KD3 |7,560|3,087|

- **Quantity/curation contract:**34,036positive cells/29,846source zeros retained without imputation or negative/censor interpretation. No amount/effect summary or condition-selected eligibility. Source sequence-level `Intensity` is not certified LFQ, unmodified-form quantity, precursor-normalized availability or molarity. All alternatives, PEP, decoy/contaminant flags and per-acquisition identification types remain. Label numbers do not prove biological replicate independence or control/KD pairing. No matched precursor/recovery calibration admitted.
- **Metadata:**48unique PRIDE file records and empty nextpage; six source labels independently match deposited design and MaxQuant configuration to RAW identifiers. The156-byte design and22,385-byte config match official SHA1/size. Search lengths8–25 conflict with article-level minimum7; retain both and do not label unsearched shorter products negative. Matching-between-runs is enabled with its FDR option false; PEP, reported1%FDR and matching provenance remain distinct. Original33,003,906-byte reference is identified but not retrieved under this family's20MiBasset cap. No RAW/large search archives downloaded.
- **Verification and failures:**293,009populated cells in two recovered XLSX files independently reconcile, including13date-shaped gene annotations verified as original Excel serial/style/date representations without recovering gene names. Original failed inventory remains. Dataset6 adds34branched peptide-spectrum rows with identification/localization evidence, not an abundance endpoint. Publisher/client-challenge and Dryad403/401 responses are preserved. The20MiB-limited EuropePMC ZIP contains12complete members before a truncated Source Data workbook; only two complete XLSX were extracted. An independently rebuilt index plus standard ZIP reader verifies original descriptors/CRC/bytes and inner workbook CRCs. The incomplete outer directory/later members are explicitly unverified. Two metadata requests hit the unchanged shared-disk reserve; a fresh retry succeeds after recovery. Failed retrievals now return nonzero status; both executed helper versions remain frozen.
- **Complementary source:** official Dryad10.5061/dryad.nk453 metadata identifies a61,018-byte workbook for three fluorogenic substrate kinetics across mouse and human standard/immunoproteasomes. Numeric download is unresolved (public403/API401); no data are admitted or access controls bypassed. Fluorophore release is a separate kinetic constraint, not a natural fragment distribution.
- **Runtime/artifacts:** complete-cell inventory35.03s/315,588,608BpeakRSS; member recovery.047s; independent recovery.059s; support inventory2.03s/55,705,600B; closure.89s/69,926,912B. Single local observations, not benchmark comparisons. Every source has a pinned path/hash/receipt; raw/failed bytes and literal-cell dump exist once under shared `data/`. No source-copy proliferation into per-fit directories.
- **Decision / next:** proceed with [Hitlist837](https://github.com/pirl-unc/hitlist/issues/837) native-panel admission and exact reference/quantity-aggregation checks. [Next specification](agents/codex/plans/native-psmd2-canonical-and-contrast.md) tests same-parent control/KD relative-product contrasts, which can cancel shared precursor/run scale and static peptide response under explicit assumptions. Keep unknown extraction, clearance, missing products and baseline calibration separate. A qualified direct comparator precedes a matched exact/discrete generative comparison. This is new biological source evidence, not a new predictive result or completion of the APC availability goal.

## 2026-10-25_0424_codex_native-psmd2-source — Resolve original quantity grain and complete parent ambiguity

- **Date / agent / reproduction:** October 25, 2026; Codex/GPT-6; Presto `8454ef9c40913c89d56bca5cd02f923701af91f6` with documented source/report work. [Report, exact invocations, frozen code, original asset hashes and independent closure](2026-10-25_0424_codex_native-psmd2-source/README.md). Local Python 3.12.6, CPU only, no installs, Modal or GPU; `PYTHONDONTWRITEBYTECODE=1`. Source qualification has no pretraining, model optimization, synthetic labels, assay loss/head mapping, validation/test split, regression/binder metrics or model prediction dumps. Their absence is intentional: no biological fit or condition-effect calculation ran.
- **Exact data/curation:** Sahu2021/PXD018722 native Hi20S control/PSMD2-knockdown panel, reusing 0344's verified Dataset7/workbook, design/mqpar, inventory and article. The exact deposited FASTA is 33,003,906 bytes with SHA1 `ebfe21f8f804ceb7679d4c41c001fde25cea2a96`. Selected original processed tables contain 10,647 peptide rows/80 fields, 40,679 evidence rows and 10,935 modification-specific rows. Preserve source-zero versus missing, matching evidence, modification/form grain, raw versus LFQ, all candidate parents and unknown biological replicate/material identity. No MHC-bound, proteasome-trapped, fluorescence or other assay is substituted for native intracellular quantities.
- **Quantity evidence:** all 74,529 publisher intensity cells exactly match original text; all feature/form ID links and six run identities reconcile. Raw and LFQ differ in 30,170 cells. Some 5,765 positive sequence/acquisition cells contain positive modified-feature evidence. Feature/form sums match 60,520/63,243 of 63,882 cells exactly; maximum relative residuals are 8.05122e-5/7.54988e-5, retained rather than repaired. Small export discrepancies do not prove a rounding algorithm or molar calibration. Original text directly recovers 13 date-corrupted gene annotations; 19,635 mass/PEP/score representation differences remain explicit under the prespecified 2e-15 comparison.
- **Reference evidence:** 73,113 reference records, 72,855 distinct sequences, 24,735,594 residues. One engineered reporter has a human taxonomy tag. Across the complete reference, 38,641 peptide spans include 403 I/L-only alternatives; 6,809 peptides have multiple distinct parent sequences, 582 have candidate accessions absent from the declared list, 53 have reporter origins and 159 lack a reference match. Twenty-four declared absent accession tokens are contaminant-prefixed. Every source-declared and discovered alternative participates in dependency grouping before descriptive selection.

| Descriptive strict-origin support | Parents with at least two peptides | Peptides | Dependency components | Possible reference contrasts |
|---|---:|---:|---:|---:|
| At least one positive acquisition per condition | 138 | 603 | 131 | 465 |
| At least two positive acquisitions per condition | 113 | 462 | 109 | 349 |
| All three acquisitions per condition positive | 82 | 310 | 81 | 228 |

- **Interpretation / preferred next condition:** these are feasibility strata, not competing fitted conditions or a selected winner. The strict-origin definition also excludes source reverse/contaminant flags, PEP >0.01, multiple complete parents, reporter possibilities and I/L-only spans. It does not turn excluded rows into negatives. No reference peptide, split, numerical contrast or model was selected. Numbered acquisitions are not established biological replicates. The source supports a small prospective same-parent contrast comparison after canonical admission; full distribution, precursor-normalized concentrations and APC transfer remain unproven.
- **Resources / verification:** four bounded requests retrieve 40,969,817 bytes in 43.48 s, including only 7,965,911 bytes of the 167,389,796-byte processed ZIP. Five members pass original headers/CRCs and independent ZIP replay; full archive SHA1 is not verified. All new source files/receipts total 68,369,865 bytes. Cell/quantity reconciliation takes 2.36 s/306.4 MiB RSS; reference mapping plus independent all-query search 106.05 s/252.4 MiB; independent quantity/date/run replay 2.20 s/274.6 MiB; support 0.30 s/187.3 MiB; closure 0.77 s/241.3 MiB. Closure independently reparses every FASTA record, checks all spans, replays all 10,647 dependency assignments with a different graph algorithm and verifies all support strata. Resource guards remain unchanged. An initial verifier-only EmptyCell failure and its source/log are preserved; the narrow recovery passes. All eight current scripts pass isolated E9/F and syntax checks.
- **Artifacts / next:** original inputs remain once under `data/native_psmd2_sources/2026-10-25_0424_codex_native-psmd2-source/`; large local derived indices have explicit paths/hashes in `results/large-local-artifacts.json`, avoiding Git duplication. Published residuals, compressed origins, dependency assignments, summaries and reproduction files support replay. [Hitlist #837](https://github.com/pirl-unc/hitlist/issues/837) remains open. A durable isolated Hitlist branch based on current main has a precise source/query implementation specification; canonical adapter, tests, release and the subsequent biological fit remain unfinished. No new RL advantage or APC concentration success is claimed.
