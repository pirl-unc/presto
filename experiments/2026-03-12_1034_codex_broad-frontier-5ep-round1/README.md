# Broad Frontier 5-Epoch Round 1

- **Date**: 2026-03-12
- **Agent / model**: Codex / GPT-5
- **Purpose**: run the currently best broad-contract 3-epoch conditions head-to-head for 5 epochs on the same broad IEDB binding contract, so the frontier is comparable beyond short-run snapshots.

## Dataset and Curation Contract

- **Source**: `data/merged_deduped.tsv`
- **Alleles**:
  - `HLA-A*02:01`
  - `HLA-A*24:02`
  - `HLA-A*03:01`
  - `HLA-A*11:01`
  - `HLA-A*01:01`
  - `HLA-B*07:02`
  - `HLA-B*44:02`
- **Assay families included**:
  - `IC50`
  - direct `KD`
  - `KD (~IC50)`
  - `KD (~EC50)`
  - `EC50`
- **Assay families excluded**:
  - qualitative binding
  - ligand presentation / MS
  - T cell assays
  - other non-binding outputs
- **Qualifier policy**: `qualifier_filter=all`
  - exact values plus inequalities, handled with censor-aware loss
- **Train / val rows**: `32,855 / 8,194`
- **Synthetic data**: none
- **Ranking / contrastive**: none

## Training Contract

- **Epochs**: `5`
- **Warm start**:
  - canonical Presto conditions: `mhc-pretrain-20260308b`
  - groove controls: no MHC warm-start checkpoint
- **Runtime defaults**:
  - `allow_tf32=true`
  - `matmul_precision=high`
  - `num_workers=0`
  - `pin_memory=true`
- **Loss space**:
  - numeric assay supervision only
  - censor-aware loss for inequalities

## Conditions Tested

- **Directness carry-forwards**:
  - `DP00`, `DP01`, `DP05`, `DG1`
- **Positional carry-forwards**:
  - `PP00..PP07`, `PG00..PG07`
- **Assay-head carry-forwards**:
  - `A00..A07`

## Outcome Summary

- **Total conditions**: 28
- **Completed**: 26
- **Failed**: 2
- **OOM**: 2
- **Failures**:
  - `DP00`
  - `DP01`
  - both failed with CUDA OOM on the 5-epoch broad-contract batch size

## Top Conditions

| design | family | condition | best val | SLL A02 / A24 | SLL ratio | FLR A02 / A24 | FLR ratio | NFL A02 / A24 | NFL ratio | all 3 probes correct | probe score |
|---|---|---|---:|---|---:|---|---:|---|---:|---|---:|
| DG1 | groove | Directness G1 groove transformer | 0.8000 | 31.0 / 31225.5 | 1007.8 | 9.5 / 1372.3 | 143.9 | 2931.9 / 7.0 | 418.4 | yes | 7.783 |
| PG07 | groove | Positional G07 triple_baseline | 0.8000 | 31.0 / 31225.5 | 1007.8 | 9.5 / 1372.3 | 143.9 | 2931.9 / 7.0 | 418.4 | yes | 7.783 |
| PG06 | groove | Positional G06 mlp(concat(start,end,frac)) | 0.7831 | 30.7 / 33670.0 | 1096.9 | 9.1 / 921.6 | 101.3 | 6458.0 / 75.7 | 85.3 | yes | 6.977 |
| A03 | presto | Assay A03 shared_base_segment_residual split_kd_proxy | 0.8177 | 50.8 / 24795.1 | 488.3 | 31.4 / 424.2 | 13.5 | 7026.1 / 49.7 | 141.4 | yes | 5.970 |
| A07 | presto | Assay A07 factorized_context_plus_segment split_kd_proxy | 0.7865 | 36.3 / 2926.7 | 80.6 | 105.4 / 1177.1 | 11.2 | 5180.5 / 69.0 | 75.1 | yes | 4.830 |
| PP00 | presto | Positional P00 start_only | 1.3795 | 37.9 / 35471.1 | 937.0 | 36.4 / 13790.7 | 379.4 | 20351.0 / 695.8 | 29.2 | yes | 7.017 |
| PP03 | presto | Positional P03 concat(start,end) | 1.3866 | 138.9 / 16914.6 | 121.8 | 32.6 / 941.7 | 28.9 | 6392.0 / 299.2 | 21.4 | yes | 4.876 |
| PP07 | presto | Positional P07 triple_baseline | 1.7386 | 52.1 / 22169.6 | 425.9 | 62.9 / 1608.4 | 25.6 | 4747.0 / 207.9 | 22.8 | yes | 5.396 |

## Full Table

See:
- [options_vs_perf.md](options_vs_perf.md)
- [options_vs_perf.json](options_vs_perf.json)
- [analysis/parsed_status.csv](analysis/parsed_status.csv)
- [analysis/parsed_status.json](analysis/parsed_status.json)

## Runtime and Artifact Notes

- Raw harvested app logs are in:
  - [analysis/raw_logs/](analysis/raw_logs/)
- Plot/data artifacts:
  - [analysis/frontier_completed.csv](analysis/frontier_completed.csv)
  - [analysis/val_vs_probe_score.png](analysis/val_vs_probe_score.png)
  - [analysis/top12_probe_score.png](analysis/top12_probe_score.png)

## Takeaway

- The strongest **canonical Presto** condition in this 5-epoch broad frontier is still assay-head `A03` on probe score, with `A07` best on validation loss among the assay-head family.
- The strongest **overall** frontier conditions after 5 epochs are still the groove-transformer controls, especially `DG1` / `PG07` / `PG06`.
- The best canonical Presto positional carry-forward (`PP03` / `PP07` family) remains materially behind the groove controls on this broad numeric contract.
- Broad-contract `legacy` directness variants `DP00` and `DP01` are not even runnable at this batch size because they OOM.
- This points to two immediate next moves:
  1. carry `A03`, `A07`, `PP03`, `PP07` forward as canonical Presto candidates,
  2. compare them directly to `DG1`/`PG06` while sweeping target encoding and assay-family factorization.
