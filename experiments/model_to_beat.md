# Model To Beat

This document tracks the current stable baseline architecture(s) to beat across the main dataset contracts used in this repo.

It is not a chronological experiment log. Use:
- [experiment_log.md](./experiment_log.md) for the historical record
- per-experiment `README.md` files for experiment-specific details

Update this file only when:
- a new result changes the practical baseline for a contract
- a baseline's scope or caveat becomes clearer
- a previously separate contract needs to be split into distinct baseline tracks

Do not paste full experiment tables here. Link to the source experiment directory instead.

## How To Use This File

For each active contract:
- define the dataset and evaluation contract clearly
- name one current baseline to beat
- record the exact primary metric used for promotion
- link the supporting experiment directory
- note any caveats, positive controls, or unresolved questions

If two model families are effectively tied, say so explicitly instead of forcing a single winner.

## Promotion Rule

Promote a new baseline here only when the result is strong enough to matter operationally:
- better primary held-out metric on the same contract, or
- similar primary metric with meaningfully better robustness / simplicity / reproducibility

If the comparison is not apples-to-apples, do not promote it as the new baseline without saying so explicitly.

## Template

Copy this section for each stable contract.

```md
## <Contract Name>

- Status: `active` / `provisional` / `historical`
- Primary metric: `<metric>`
- Source experiment: [<dir>](./YYYY-MM-DD_HHMM_agent_slug/)

### Dataset Contract
- Source:
- Curation:
- Included assay families:
- Excluded assay families:
- Qualifier / censor policy:
- Split:

### Baseline To Beat
- Model:
- Training contract:
- Validation metric(s):
- Test metric(s):
- Why this is the baseline:

### Positive Control / Comparator
- Historical anchor:
- Closest competitor:

### Caveats
- Caveat 1
- Caveat 2

### Open Questions
- Question 1
- Question 2
```

## Important Caveat On All Pre-2026-08 Baselines

Every baseline recorded below was trained through the merged-TSV path, which has **no
flank columns**. `ProcessingRecord` was built without `flank_n`/`flank_c`, and
`PrestoCollator` only tokenizes flanks when at least one sample carries them, so the
`nflank` and `cflank` segments reached the model as a single `<MISSING>` token. The
"Inputs" lines below list `nflank` and `cflank` because the segments exist, not because
they carried sequence.

Baselines are therefore not comparable to anything trained through
`--data-source hitlist`, which supplies real flanks at ~98% coverage. Re-baseline before
promoting any post-migration result. See `tasks/todo.md` Stage 0.

## Active Baselines

### Frozen human-source/HLA IC50 cohort — provisional

Train-only alpha=1 ridge (length, composition, MHC class) is the baseline to beat
for the exact 9,053-row frozen cohort and declared effective target space.
Validation/test exact IC50 RMSE: **1.037626 / 1.138279 log10(nM)**; raw-space
RMSE: 1.047534 / 1.151585. The 200-step canonical Presto fit fails its prospective
quality screen and has near-constant predictions. This is exploratory
peptide-disjoint evidence with shared studies, unknown biological identity and
explicit species-context limitations; it is not an external validation result.
[Full contract and comparison](2026-10-16_1030_codex_effective-target-fit/README.md).


## 7-Allele Numeric Class I Binding, Strict Seq-Only

- Status: `active`
- Primary metric: `held-out test Spearman`
- Source experiment: [2026-03-17_1042_claude_7allele-bakeoff](./2026-03-17_1042_claude_7allele-bakeoff/)

### Dataset Contract
- Source: `data/merged_deduped.tsv`
- Curation: peptide-group `80/10/10` split with split_seed `42`
- Alleles: HLA-A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02
- Included assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- Excluded assay families: qualitative-only measurements
- Qualifier / censor policy: `all`
- Inputs: `nflank`, `peptide`, `cflank`, `mhc_a`, `mhc_b`
- Assay-selector inputs: forbidden

### Baseline To Beat
- Model: `L2` — d128, `dag_prep_readout_leaf`, `assay_heads_only`, pretrained, `lr=3e-4`, `warmup_cosine`
- Training contract: `50` epochs, batch size `256`, `AdamW`, `lr=3e-4`, `warmup_cosine`, weight decay `0.01`, `mhcflurry`, `split_kd_proxy`, binding_core_lengths=8,9,10,11
- Test metric(s) (mean±std, 3 seeds): Spearman `0.8062±0.0060`, AUROC `0.9336±0.0039`, AUPRC `0.8725±0.0106`, RMSE log10 `0.8502±0.0195`, F1 `0.8202±0.0057`, balanced_accuracy `0.8612±0.0033`
- Probe discrimination: SLLQHLIGL A02/A24 ratio 2502x, A02 Kd 3.8±2.2 nM, A02 bind_prob 0.971
- Stability: 3/3 seeds converged (no NaN)
- Why this is the baseline: best combined regression + probe discrimination + stability across 13 conditions x 3 seeds. Top F1 and balanced accuracy. Probe discrimination is biological — the model correctly identifies SLLQHLIGL as a strong A02:01 binder while rejecting it for all other alleles.

### Positive Control / Comparator
- Regression-only champion: `D3` (d32 DAG full no-pretrain) — test Spearman `0.8101±0.0032` but collapses all non-binder alleles to identical predictions
- Prior 7-allele baselines: [2026-03-16_1454_claude_factorized-ablation-7allele](./2026-03-16_1454_claude_factorized-ablation-7allele/), [2026-03-16_1813_claude_lr-stability-7allele](./2026-03-16_1813_claude_lr-stability-7allele/)

### Caveats
- L2's A24:02 binding probability (0.096) is slightly elevated — seed 42 predicts A24 Kd=664 nM, which is borderline. This is a known trade-off of the DAG architecture.
- d32 D3 beats L2 on Spearman by 0.004 but lacks allele-level resolution. At full class I scale (100+ alleles), d128 likely matters more.
- 7-allele contract only. Results may differ at full class I scale.

### Open Questions
- Whether L2's advantage holds at full class I scale (105+ HLA alleles, ~250K+ rows)
- Whether pretraining becomes more important at full scale
- Whether d128 advantage over d32 grows with more alleles (as expected)
- Whether `dag_prep_readout_leaf` vs `dag_method_leaf` gap persists at 7-allele with seeds

### Recommended Recipe for Full Class I Training
```
--d-model 128 --n-layers 2 --n-heads 4
--affinity-assay-residual-mode dag_prep_readout_leaf
--affinity-loss-mode assay_heads_only
--lr 3e-4 --lr-schedule warmup_cosine
--affinity-target-encoding mhcflurry
--init-checkpoint /checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt
--kd-grouping-mode split_kd_proxy --max-affinity-nm 100000
--binding-core-lengths 8,9,10,11 --binding-core-refinement shared
--peptide-pos-mode concat_start_end_frac --groove-pos-mode concat_start_end_frac
--no-synthetic-negatives --binding-contrastive-weight 0 --binding-peptide-contrastive-weight 0
--weight-decay 0.01
--train-all-alleles --train-mhc-class-filter I
```

---

## 2-Allele Broad Numeric Class I Binding, Strict Seq-Only

- Status: `active`
- Primary metric: `held-out test Spearman`
- Source experiment: [2026-03-16_1621_codex_pf07-output-tying-weight-sweep](./2026-03-16_1621_codex_pf07-output-tying-weight-sweep/)

### Dataset Contract
- Source: `data/merged_deduped.tsv`
- Curation: peptide-group `80/10/10` split with seed `42`
- Included assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- Excluded assay families: qualitative-only measurements
- Qualifier / censor policy: `all`
- Split: `HLA-A*02:01`, `HLA-A*24:02`, `numeric_no_qualitative`
- Inputs: `nflank`, `peptide`, `cflank`, `mhc_a`, `mhc_b`
- Assay-selector inputs: forbidden

### Baseline To Beat
- Model: `PF07` assay-structured DAG, main `Presto` full-output path
- Training contract: `50` epochs, batch size `256`, `AdamW`, `lr=1e-3`, weight decay `0.01`, `mhcflurry`, `split_kd_proxy`, `dag_prep_readout_leaf`
- Validation metric(s): val Spearman `0.8230450`
- Test metric(s): test Spearman `0.8381589`, AUROC `0.9358775`, AUPRC `0.8780973`, RMSE log10 `0.8703900`
- Why this is the baseline: it is the strongest verified no-assay-input model currently in-repo, and it improves meaningfully on the previous honest PF07 flat-head baseline by adding output-side assay structure without reintroducing forbidden assay-selector inputs

### Positive Control / Comparator
- Closest honest legacy comparator: [2026-03-16_2142_codex_exp21-honest-no-assay-repeat](./2026-03-16_2142_codex_exp21-honest-no-assay-repeat/)
- Closest honest same-family comparator: [2026-03-16_2355_codex_pf07-assay-structured-dag-sweep](./2026-03-16_2355_codex_pf07-assay-structured-dag-sweep/)

### Caveats
- This baseline comes from a single-seed architecture sweep, so it is operationally current rather than fully seed-confirmed
- The honest no-assay repeat of the old legacy benchmark used a different output/head contract, so the comparison is clean on the input side but not a perfect architecture-isolation study
- `dag_method_leaf` is very close and has better AUPRC, so the exact winner between the two leaf variants should still be treated as somewhat seed-sensitive until confirmed

### Open Questions
- Whether `dag_prep_readout_leaf` stays ahead of `dag_method_leaf` across seeds
- Whether a groove-style honest model can beat this structured-output PF07 family once evaluated under the same no-assay-input output contract
- Whether the same output-side DAG idea extends cleanly to T-cell and presentation / mass-spec assay families

## Historical / Deprecated Baselines

## 2-Allele Broad Numeric Class I Binding, Legacy Assay-Conditioned Benchmark

- Status: `historical`
- Primary metric: `held-out test Spearman`
- Source experiment: [2026-03-15_1226_codex_exp21-seed-epoch-confirmation](./2026-03-15_1226_codex_exp21-seed-epoch-confirmation/)

### Dataset Contract
- Source: `data/merged_deduped.tsv`
- Curation: peptide-group split varied with run seed in this legacy benchmark path
- Included assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- Excluded assay families: qualitative-only measurements
- Qualifier / censor policy: `all`
- Split: `HLA-A*02:01`, `HLA-A*24:02`, `numeric_no_qualitative`

### Baseline To Beat
- Model: `groove`, `cond_id=2`, no content conditioning
- Training contract: `50` epochs, batch size `256`, `AdamW`, `lr=1e-3`, weight decay `0.01`
- Validation metric(s): best single rerun reached val Spearman `0.845193`
- Test metric(s): best single run test Spearman `0.854139`
- Why this is historical only: the legacy distributional BA path uses assay embeddings from `binding_context`, so it is not a valid no-assay-input baseline

### Positive Control / Comparator
- Honest repeat: [2026-03-16_2142_codex_exp21-honest-no-assay-repeat](./2026-03-16_2142_codex_exp21-honest-no-assay-repeat/)
- Honest canonical baseline: [2026-03-16_1621_codex_pf07-output-tying-weight-sweep](./2026-03-16_1621_codex_pf07-output-tying-weight-sweep/)

### Caveats
- The honest repeat of the same single-seed groove c02 point fell to test Spearman `0.7995139`
- The drop from the old best-seed `0.8541390` result is `-0.0546252` Spearman

### Open Questions
- Whether the legacy groove encoder still has an advantage once paired with a canonical no-assay-input output contract


## PXD004886 raw-S4 conditional-response diagnostic reference

- Status: `diagnostic_only`; [full contract and report](2026-10-23_0359_codex_raw-s4-response/README.md). This is separate from MHC/availability baselines.
- Canonical shared recipe96b9232:22 selected peptides,224 peptide×site summaries,nominal load/site inputs; median positive log10 raw peak area, no imputation or certified molar interpretation.
- Reference: load/site ridge, nested fivefold peptide-grouped CV; inner alpha grid1e-4..100,training-only scaling,equal peptide weights. Outer pooled RMSE0.8084456,MAE0.5149689,Pearson0.8805007,Spearman0.9013222. Exact per-fold inner-validation/train/test metrics and predictions in the linked report.
- Controls: weighted mean RMSE1.1796061; same model plus length/AA fractions with shared regularization RMSE1.2761615. Sequence arm shrinks useful context coefficients too; a separate-penalty ablation remains untested.
- No untouched external test,homology/specimen independence,detector transfer,processing or concentration claim. These outer folds are now development data for any follow-up changes. No full-data production model fitted.

### Cellular MAPP captured composition — provisional development control

- Source: unchanged Hitlist MAPPmin2 manifest `df78d02f8200809c659e43adfae29866c48b712122b740379106994622f4ac80`, PXD009092 human HEK293 proteasome-associated fragments. Training 72 parents / 144 examples / 918 measurements; historically inspected validation 15 parents / 30 examples / 112 measurements. The 14 historical test parents are excluded from this diagnostic.
- Control: uniform independent cleavage with the earlier training-fitted common removal parameter, followed by one nonnegative training-only amplitude coefficient, beta = 0.3021227141. Uses span geometry and occurrence multiplicity, with no learned amino-acid preference.
- Primary validation metric: hierarchical-weighted centered-log10 RMSE **0.267107**, versus **0.286029** for uniform observed composition. Pooled Pearson/Spearman **0.441603 / 0.372251**; weighted R2 **0.127933**. Helps 10/15 validation parents, worsens 4, ties 1.
- [Registered comparison and independent verification](2026-10-24_0106_codex_cellular-amplitude/README.md). Calibrated adjacent/wider sequence priors score 0.275897 / 0.286965 and fail the predeclared comparison against both controls. Stop further tuning of this bridge on the same MAPP development set.
- This is a provisional control for this development contract, not a new blind test result or a replacement for historical test-set baselines. Calibration was motivated by prior development failures. Captured MS composition is not free-peptide abundance, measured clearance, or APC concentration; treatment response remains unmodeled. Independent cellular/source validation is still required.

### Native-cell chemical peptide response — provisional validation control

- Canonical Hitlist native response recipe manifest `c72980c070cb1296609bbe7d296e27b6a2a22d3ff354ef846c1c99ad7959ccfc`; Dasgupta2014/Gelman2013 HEK293T/SH-SY5Y native extract treated/control isotope ratios. Train1,861groups/6,735observations, validation245groups/932observations, all exact and source-declared bounds. Components retain all candidate parent/reuse/source-assertion links; weights average condition/component/form/observation.
- **Condition-only** linear model,28supported parameters, training-only scaling/SVD, penalty.01, fixed L-BFGS-B. Validation exact/censored weighted loss **0.073495** versus nochange **0.186297**; exact-value RMSE **0.247726 log10ratio**, pooled Pearson/Spearman **0.4518/0.4279**. Explicit bound satisfaction remains zero.
- [Registered comparison, source-linked predictions and independent closure](2026-10-24_0314_codex_native-response-baselines/README.md). Shared/compound-dependent sequence loss.129189/.153178; neither passes biological comparison. Training improvements do not generalize. Canonical outputs are `results/qualified-v2`, with exact input ties retained.
- This0314family did not evaluate the reserved631testgroups. The matched full-parent subset was subsequently evaluated once in0410 below. Same studies and treatment categories across partitions, unresolved specimen/shared-denominator/homology independence. Native ratios are not APC concentrations or genetic KO effects. These0314metrics apply to the original admitted set.


### Native-cell response, full-parent-supported subset — provisional processing reference

- [Fixed ten-model comparison and independent closure](2026-10-24_0410_codex_native-processing/README.md), same canonical recipec72980c0… but omit six unsupported training groups and renormalize original split weights once. Train1,855groups/6,707observations; val245/932; test631/2,447 across19/16held-out components. All six controls refitted; test opened only after all fixed fits completed and checkpoints were sealed.
- **Exact sequence-dependent independent cleavage/input/removal**,95supported parameters,200zero-init full-batch Adamsteps,lr.02, penalty.01 except sharedoffsetintercept. Predict native treated/control log10ratio; no per-peptide offsets. Validation primaryloss **0.070384**, test **0.129789** versus condition-only **0.073514 / 0.130557**. Validation/test exactRMSE **0.245952 / 0.243207** versus **0.247717 / 0.250953**. TestPearson/Spearman **0.5578 / 0.5373**; weighted exactR2 **0.0929**. Testloss improvement only0.59%, exactRMSE3.09%; small, single-seed provisional gains.
- Sampled conditional-score hybrid is close (testloss.129336/RMSE.242887) but about25times slower (138.74s versus5.55s); validation slightly favors exact. Keep exact as the practical reference, not a statistically established winner. Direct sequence regressions generalize worse. Testbound satisfaction remains14.92% for exactprocessing, validation1.30%.
- Parent-component holdout under seen treatments, same studies and unresolved specimen/shared-control/homology dependence. Test is now inspected; no further tuning against it. No causal machinery/APC concentration or independently calibrated detector claim. Preserve the simpler condition-only comparator because most predictive signal is treatment-level.
