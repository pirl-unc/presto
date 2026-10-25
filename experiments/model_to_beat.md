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

### PXD058121 same-peptide temporal product/substrate ratios — provisional

Direct peptide-specific linear slopes shrunk toward a jointly fitted shared slope
(penalty 0.1, selected on validation) are the reference for training at 0–80 minutes,
validation at 100 and test at 120. Validation/test RMSE **0.206085 / 0.228184** in
log10 product/intact-substrate intensity ratio, versus persistence **0.229625 /
0.364187**. Minimal independent-bond fragmentation fails at **0.304593 / 0.359038**;
conditional-score training gives no advantage over exact gradients.

Training-only peptide anchors are essential: test change-from-last Spearman is
only **0.23071** despite raw-ratio Spearman **0.90998**. One substrate/purified
preparation, unresolved replicate independence and unknown MS response prevent
new-peptide, independent-specimen or APC-concentration claims. These inspected
held-outs must not be reused as blind tests for revisions.
[Frozen contract and all outcomes](2026-10-25_0046_codex_intact-kinetic-forecast/README.md).

### Intact temporal changes — development comparator only

[0143 nested development comparison](2026-10-25_0143_codex_intact-temporal-representation/README.md)
uses only the original training cohort: 463 changes for 96 same-parent peptides,
conditioned on each peptide's measured first-time baseline. Sequence-conditioned
slopes with nested ridge selection score RMSE **0.345792**, versus length
**0.379042** (8.772% gain), R² **0.085610**, Pearson/Spearman **0.599177/0.599769**.
Four of five peptide folds improve; the paired peptide bootstrap gain interval
**[-4.259%,20.109%]** includes zero. All selected alpha values are 1. Preserve the
frozen folds, feature transforms and selection procedure for this narrow comparator.
This does not replace 0046's future-time benchmark: units are change from the
supplied first-time baseline, the split differs and the development data were
previously inspected. No fresh biological test or APC concentration qualification.
Unpenalized physical RMSE **0.380755** improves only 1.362% over penalized
**0.386013**; retain the existing physical references without claiming promotion.

### PXD058121 frozen organ transfer — no learned comparator promoted

[1747 fixed-model transfer](2026-10-25_1747_codex_intact-method-transfer/README.md)
admits 236 primary cells across Brain/Liver/Spleen using a separately qualified
study-method search policy. The frozen0143sequence ensemble has RMSE **0.288802**,
versus **0.216691 for no change** and **0.348240 for length**. Prediction SD remains
0.208033 versus target SD0.208481; poor transfer is not constant-output collapse.
The sequence model fails the fixed no-change gate, with a peptide-bootstrap95%
gain interval **[-70.50%,-5.62%]**. All physical/exact/conditional-score arms also
lose to no change; no RL advantage. Do not refit or tune on this inspected F3
cohort. These relative changes from supplied baselines do not calibrate APC
concentrations; the0143model remains a development-only comparator.

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

### PXD057588 macrophage MHC-I response — context reference after negative processing result

- [Fixed nine-model comparison and independent closure](2026-10-24_0449_codex_apc-precursor-response/README.md). Table9 MHC ligand log10 fold changes, Table11 condition-level digest-derived parent input. Canonical recipe93ef0b23…; all candidate parents require ≥2distinct digest sequences. Train492peptides/2,952observations; validation96/576; test129/774. Fixed all-source dependency-component split, condition/component/peptide/channel weights.
- **Condition-only**,2supported coordinates, fixed L-BFGS-B/.01penalty excluding globaloffset. Validation/test MSE **.017034817/.014402625**, RMSE **.130517497/.120010938**; testPearson/Spearman **.18099/.25118**, weightedR2 **.001305**. Only a weak reference: nochange testRMSE.120611.
- Sequence processing plus precursor testRMSE.139278; conditional sampled.139298 at35×cost. Removing precursor gives.120008, a negligible.0052%MSE gain and nearly constant within-condition predictions. Frozenhuman-native prior does not rescue precursor transfer. Keep simpler context reference rather than promoting numerical differences.
- Same study/conditions/shared controls; no exact aliquot matching or independent concentration/machinery validation. MHC-bound response combines processing, selection and observation. Test is now inspected; do not tune against it. Precursor calibration and an independent prospective biological population remain required.

### HAP1 machinery elution response — frozen held-out sequence reference

- [Training development](2026-10-24_0730_codex_hap1-response-baselines/README.md) and [once-only frozen outer evaluation](2026-10-24_1503_codex_hap1-frozen-outer/README.md). Canonical recipe a1eec392…; training 87,260 positive response cells / 6,317 peptides / 1,798 molecular components across ten KO conditions. Validation 19,744 and test 22,724 complete-precursor response cells; same-study molecular-component splits, now inspected and closed to further tuning.
- **Preselected condition-specific sequence ridge, lambda 0.1:** length, composition and terminal features plus calibrated protein response and condition offsets. Frozen validation/test RMSE **0.577617 / 0.573380 log10 elution ratio** versus calibrated precursor **0.648257 / 0.631249**; gains **10.897% / 9.167%**. Test weighted R² **0.24986**, unweighted Spearman **0.46623**. All ten KOs improve test RMSE (4.25–16.61%); within-condition prediction SD 0.126–0.474. Both prespecified direct-model gates pass, without refitting or outer model selection.
- **ERAP1 processing fails its gate:** preselected sequence baseline test RMSE **0.585327**; residue/length physical augmentation exact **0.585337**, sampled-trained **0.585335**. The physical-only residue/length model scores **0.698824**. Keep the simpler sequence reference; numerical correctness and nonzero correction variance do not establish additional biological signal.
- Source-positive, complete-case MHC elution responses; no imputed negatives. Candidate origins retained and ambiguous protein groups deferred. Shared controls/study and unresolved specimen/clone/homology independence remain. HAP1 is not a professional APC. This does not validate free-peptide abundance, causal processing, independent MS detectability, unseen KOs, external cohorts or APC concentrations. Use a new prospective biological cohort for subsequent model qualification.


- **New development-only control, not a held-out replacement:** [2041 frozen MHC-score comparison](2026-10-24_2041_codex_hap1-affinity-control/README.md) adds three MHCflurry log10-score features to the original sequence ridge. Reused inner-CV RMSE **0.562644** versus **0.574832**, a **2.120%** gain across all five folds; fixed shuffled-score control **0.575043**. Use its frozen full/fold checkpoints for future development comparisons under this exact contract. No new outer scoring; 4,925/6,317 peptides occur in the external predictor training table and full teacher/sample lineage is unresolved. This is neither independent binding evidence nor an APC-concentration result.


### PXD021013 conditional non-MHC MS1 response — pool-held-out sequence reference

- [Registered comparison and independent reconstruction](2026-10-24_1702_codex_ms1-grouped-observer/README.md). Canonical source e373789b…/recipe0183d91b…, positive intended variable-Unmodified forms under fixedCAM. Metadata-selected, intended-sequence-disjoint 16/4/4 pools; 8,712/2,104/2,061 examples. Test now inspected and closed.
- **Preselected sequence ridge alpha0.1**, 42 within-pool-supported standardized length/composition/N-terminal features, no intercept. Equal-pool MSE + alpha*mean(coefficient²). Validation/test centered-log10-signal RMSE **0.470624 / 0.449828** against length **0.534663 / 0.520456**; gains **11.98% / 13.57%**. Test weightedR² **0.253294**, unweightedSpearman **0.525644**, positive Pearson/Spearman in all4test pools; all prespecified gates pass.
- Prespecified neural ensemble is a secondary result: testRMSE0.443033,Spearman0.547499. Keep the primary sequence reference for future fixed comparisons; no post-test selection or refitting.
- This predicts measured conditional signal, not pure detectability or molarity. Crude synthesis, purity/load/handling confounded. All C-terminal main effects cancel under pool centering; missing category support is explicit. No biological elution transfer, unconditional detection, specimen independence or intracellular availability claim. Any biological use requires prospective transfer evaluation with a shared frozen observer for exact and sampled arms.


## Goldberg2025 conditional baseline MAPP distribution

- Status: provisional measured-signal reference, not an APC availability model.
- Source: [2244 original baseline comparison](2026-10-24_2244_codex_mapp2025-baseline-prediction/README.md).
- Contract: PXD052314 six A549 controls; source-human-only, unflagged positive canonical peptide PEP<.01/directMSMS; one supplied parent accession and>=3observed sequences/group. All-source dependency split;40,701cells/458admitted components. Positive observed-set conditioning, unknown physical/homology independence.
- Distribution reference:61raw sequence features→training-scaled/centered58supported coordinates→linearlog10score→groupsoftmax. Exact reverseKL+.01mean(beta²),500Adamstepslr.02. Validation/testreverseKL.376448/.422502 versusuniform.431589/.477545. AllRLseedsalso beatuniform but addno clear gain and cost~1.41×.
- Separate regression control: validation-selectedsequence ridgealpha1;testRMSE.415585 versuslength.434413. Primary>=5%bothheldoutgatefails(testgain4.334%);do not relabelthis as a fully qualified sequence predictor.
- Held-outs closed; a future model must use prospective development/evaluation rather than tuning these outcomes. No absolute/missing-mass calibration, measured precursor pairing or separated processing/capture/MS model is established.
