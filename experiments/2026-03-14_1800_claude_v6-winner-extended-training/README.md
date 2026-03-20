# EXP-18: Extended Training on v6 Winner

## Corrected Contract

Earlier summaries elsewhere described this follow-up as a 7-allele exact-IC50 warm-start experiment. The raw run summaries in `results/runs/` show the actual executed contract matched the broad 2-allele EXP-16 winner family:

- `measurement_profile=numeric_no_qualitative`
- alleles: `HLA-A*02:01`, `HLA-A*24:02`
- assay families: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- qualifier filter: `all`
- split sizes: train `15530`, val `1919`, test `1915`
- no warm start
- AdamW `lr=1e-3`, `weight_decay=0.01`, `batch_size=256`

## Hypothesis
The v6 factorial winner (cond=2: d=32, mhcflurry, max_nM=100k, no content-conditioning) showed the smallest overfitting gap (0.005 Spearman) of all configurations at 50 epochs. This suggests d=32 may still be improving. Extending to 100 and 200 epochs with multiple seeds tests whether we can push past the 0.84 Spearman ceiling.

A d=64 + content-conditioning comparator is included as the next-best family from EXP-16.

## Agent/Model
- Agent: Claude Code (claude-opus-4-6)
- Date: 2026-03-14

## Dataset & Curation Contract
- Source: `data/merged_deduped.tsv`
- 2 class-I alleles: HLA-A*02:01, HLA-A*24:02
- Assay families: IC50, direct KD, KD (~IC50), KD (~EC50), EC50
- Qualifiers: all
- Split: deterministic peptide-group train/val/test

## Training Contract
- Backbone: shared-path historical AblationEncoder
- Warm start: none
- Optimizer: AdamW, lr=1e-3, weight_decay=0.01
- Batch size: 256
- GPU: H100!

## Conditions (6 runs)

| Run | cond_id | cc | epochs | seed | Description |
|-----|---------|-----|--------|------|-------------|
| r01 | 2 | no | 100 | 42 | Winner, 100ep, seed 42 |
| r02 | 2 | no | 100 | 43 | Winner, 100ep, seed 43 |
| r03 | 2 | no | 100 | 44 | Winner, 100ep, seed 44 |
| r04 | 2 | no | 200 | 42 | Winner, 200ep, seed 42 |
| r05 | 2 | no | 200 | 43 | Winner, 200ep, seed 43 |
| r06 | 6 | yes | 100 | 42 | d=64 mhcflurry 100k cc1 comparator |

cond_id 2 = embed_dim=32, head=mhcflurry, max_nM=100k
cond_id 6 = embed_dim=64, head=mhcflurry, max_nM=100k

## Key Questions
1. Does d=32 mhcflurry continue improving past 50 epochs?
2. Is there seed variance at 100 epochs (3 seeds)?
3. Does 200ep help or does it overfit?
4. Does d=64+cc catch up with more training?

## Results

| run_id | description | test_spearman | test_auroc | test_auprc | test_f1 | test_rmse_log10 |
|--------|-------------|---------------|------------|------------|---------|-----------------|
| `v6ext-r01-c02-cc0-100ep-s42` | d32, no-cc, 100ep, seed42 | **0.8369** | 0.9371 | 0.8959 | 0.8347 | **0.8565** |
| `v6ext-r02-c02-cc0-100ep-s43` | d32, no-cc, 100ep, seed43 | 0.8320 | 0.9325 | 0.8934 | 0.8214 | 0.8806 |
| `v6ext-r03-c02-cc0-100ep-s44` | d32, no-cc, 100ep, seed44 | 0.8348 | 0.9360 | **0.9070** | 0.8140 | 0.8815 |
| `v6ext-r04-c02-cc0-200ep-s42` | d32, no-cc, 200ep, seed42 | 0.8320 | 0.9369 | 0.8941 | 0.8314 | 0.8830 |
| `v6ext-r05-c02-cc0-200ep-s43` | d32, no-cc, 200ep, seed43 | 0.8256 | 0.9297 | 0.8882 | 0.8261 | 0.8991 |
| `v6ext-r06-c06-cc1-100ep-s42` | d64, cc1, 100ep, seed42 | 0.8363 | 0.9353 | 0.8937 | 0.8235 | 0.8763 |

## Aggregate Readout

- `d32`, no content-conditioning, `100` epochs:
  - mean test Spearman `0.8346`
  - std `0.0025`
- `d32`, no content-conditioning, `200` epochs:
  - mean test Spearman `0.8288`
  - std `0.0045`
- `d64`, content-conditioned, `100` epochs:
  - test Spearman `0.8363`

## Takeaways

1. Extending the v6 winner to `100` or `200` epochs did not beat the original EXP-16 winner (`d32`, `mhcflurry`, `max_nM=100k`, no content-conditioning, `50` epochs, test Spearman `0.8435`).
2. `100` epochs is better than `200` epochs for the `d32` no-content-conditioned winner family; the longer run appears to overfit rather than improve.
3. Seed variance at `100` epochs is small, so the lack of improvement is probably real, not just unlucky noise.

## Artifacts

- Raw run artifacts were pulled locally under `results/runs/`
- Summary tables:
  - `results/condition_summary.csv`
  - `results/epoch_summary.csv`
  - `results/final_probe_predictions.csv`
- Plots:
  - `results/test_spearman_ranking.png`
  - `results/test_metric_grid.png`
  - `results/training_curves.png`
  - `results/final_probe_heatmap.png`
