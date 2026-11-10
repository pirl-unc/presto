# Conversation observations and future-work map

Publication handoff, 2026-11-10. These are standing scientific requirements,
not claims that the present checkpoint is qualified. The shared architecture
and one observation-manifest training interface remain the default direction.
Preserve old checkpoints and frozen experiment snapshots for reproduction.

| Observation / question | Next measurable action | Owner |
|---|---|---|
| Shared register evaluation and MHC-I↔II transfer | Match data exposure, seeds and optimization; compare I-only, II-only and joint fits with both classes evaluated. Audit register weights, gradient/update coverage and full-flank versus restricted context. Keep one encoder/trainer. | [Presto130](https://github.com/pirl-unc/presto/issues/130) |
| Affinity depends on the actual assay, beyond an IC50/KD label | Keep categorical assay embeddings and a common latent decoder; qualify radioactivity, fluorescence, purified/cellular and source-verified protocol effects separately. Distinct physical quantities retain their units/targets. | [Presto140](https://github.com/pirl-unc/presto/issues/140) |
| Pooled and assay-stratified Spearman can move differently | Report exact n, protocol/class/study strata and within-assay calibration. Pooled rank correlation can benefit from between-assay offsets without improving within-assay ordering. Do not compare differently composed holdouts as if matched. | [Presto140](https://github.com/pirl-unc/presto/issues/140) |
| Bounds are poorly satisfied; repeated exact limits can hide censoring | Preserve explicit and inferred bounds separately, initially source/protocol-specific. Audit both upper and lower detection limits; require stronger independent evidence before sharing thresholds. Count missing qualifiers separately from explicit equals. | [Presto140](https://github.com/pirl-unc/presto/issues/140) |
| Multi-output training may dilute affinity exposure or cause interference | Separate sampling exposure from gradient conflict; run matched learning-rate schedule and within-MHC/assay ranking ablations. Protect ligand performance and report actual labels/updates per task. | [Presto141](https://github.com/pirl-unc/presto/issues/141) |
| Generated ligand backgrounds are not measured weak affinities | Keep sampled backgrounds distinct. Any randomized >10k/50k/100k/200k nM targets are an explicitly synthetic prior experiment with an unchanged real-measurement comparator and no synthetic evaluation targets. | [Presto142](https://github.com/pirl-unc/presto/issues/142) |
| MHC-II training coverage is smaller than published NetMHCIIpan totals | Reconcile measured ligands/BA versus generated negatives, full development corpus versus TRAIN, species, protocols, exact chain pairs, modifications and source overlap. Report net independent additions. | [Presto139](https://github.com/pirl-unc/presto/issues/139) |
| Ligand AUROC depends on negative construction and study split | Score NetMHCpan/MHCflurry/NetMHCIIpan on identical eligible examples; separate MS detection from affinity and processing. Include same-protein backgrounds, AUPRC, prevalence and allele/study support. Paper headline values are not matched baselines. | [Presto140](https://github.com/pirl-unc/presto/issues/140) |
| Stability is useful but weakly supported in class II | Extend the shared measurement interface to kinetic half-life, koff, thermal td50 and Tm with their own transforms and conditions. Audit real support and independent class-II validation before claiming successful prediction. | [Presto138](https://github.com/pirl-unc/presto/issues/138), [Hitlist875](https://github.com/pirl-unc/hitlist/issues/875) |
| Processing and DM/DO editing need qualified condition contrasts | Finish source and sequence overlap qualification, preserve cell/intervention evidence and unknown states; compare condition-aware versus masked common-model fits. Do not infer cleavage or abundance targets from MS detection. | [Presto155](https://github.com/pirl-unc/presto/issues/155), [Presto132](https://github.com/pirl-unc/presto/issues/132) |
| Auxiliary molecular identity pretraining could aid transfer | Test auxiliary-only tasks versus soft conditioning with explicit manual overrides; avoid inferring CD4/CD8 phenotype from TCR sequences as ground truth. | [Presto133–137](https://github.com/pirl-unc/presto/issues/133) (individual issues 134, 135, 136, 137) |

Current source findings are preserved in the canonical experiment registry:
21,547 DP/DM source candidates, 8,338 embargoed against presentation/Atlas
protected identities, and 13,209 clear only those checks. All remain ineligible
for TRAIN until the remaining identity reconciliation and a prospective
comparison contract are complete. The companion Hitlist integration is a draft
until software gates pass; [Hitlist897](https://github.com/pirl-unc/hitlist/issues/897)
tracks reproduced current-dependency incompatibilities.

Do not close scientific qualification issues 130/132 or merge draft PR131 based
on software tests, data availability, or the historical development metrics.
This publication step launches no new training or GPU run.
