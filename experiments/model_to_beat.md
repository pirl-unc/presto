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

## Active Baselines

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
