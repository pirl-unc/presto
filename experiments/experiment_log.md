

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
