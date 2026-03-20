# Assay Head / KD Grouping Round 1 Results

## Contract

- 7 class-I alleles: `HLA-A*02:01,HLA-A*24:02,HLA-A*03:01,HLA-A*11:01,HLA-A*01:01,HLA-B*07:02,HLA-B*44:02`
- Broad numeric binding contract: `IC50`, direct `KD`, `KD (~IC50)`, `KD (~EC50)`, `EC50`
- Qualifiers: `all` with censor-aware loss
- Train/val rows: `32,855 / 8,194`
- Warm start: `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- Base architecture: canonical Presto using P04 positional base
- No synthetics, no ranking, 3 epochs, batch size 140

## Conditions

| design | residual mode | KD grouping | best epoch | best val | setup s | epoch s | fwd s | bwd s | data wait s | SLL A02/A24 | FLR A02/A24 | NFL A02/A24 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| A00 | `pooled_single_output` | `merged_kd` | 3 | 2.9206 | 49.7 | 84.9 | 38.5 | 11.7 | 4.3 | 1703.0 / 1703.0 | 1703.2 / 1703.2 | 1703.3 / 1703.3 |
| A01 | `pooled_single_output` | `split_kd_proxy` | 3 | 1.0437 | 53.3 | 86.3 | 39.1 | 12.4 | 4.4 | 48.4 / 1171.5 | 21.0 / 87.2 | 899.4 / 468.4 |
| A02 | `shared_base_segment_residual` | `merged_kd` | 3 | 1.4785 | 47.1 | 85.6 | 38.9 | 12.5 | 4.0 | 190.6 / 5983.7 | 87.5 / 2000.2 | 1382.4 / 902.1 |
| A03 | `shared_base_segment_residual` | `split_kd_proxy` | 3 | 0.8803 | 69.5 | 123.8 | 61.5 | 24.3 | 7.2 | 54.5 / 14799.9 | 46.5 / 1139.2 | 1117.0 / 268.7 |
| A04 | `shared_base_factorized_context_residual` | `merged_kd` | 3 | 2.2072 | 49.5 | 84.4 | 38.0 | 12.2 | 3.9 | 1036.1 / 1036.1 | 1036.1 / 1036.1 | 1036.1 / 1036.1 |
| A05 | `shared_base_factorized_context_residual` | `split_kd_proxy` | 3 | 1.1051 | 48.4 | 87.4 | 39.9 | 13.1 | 4.0 | 119.5 / 127.7 | 184.1 / 238.8 | 165.9 / 176.1 |
| A06 | `shared_base_factorized_context_plus_segment_residual` | `merged_kd` | 2 | 1.6669 | 47.7 | 84.9 | 38.3 | 12.3 | 4.0 | 880.8 / 2484.2 | 381.2 / 1506.5 | 573.4 / 654.6 |
| A07 | `shared_base_factorized_context_plus_segment_residual` | `split_kd_proxy` | 3 | 0.7885 | 50.4 | 87.2 | 39.5 | 13.2 | 4.0 | 58.8 / 7921.0 | 56.8 / 1099.1 | 4028.6 / 121.6 |

## Takeaways

- Best validation loss: `A07` with `shared_base_factorized_context_plus_segment_residual` + `split_kd_proxy` (`0.7885`).
- Strongest `SLLQHLIGL` separation: `A03` (`54.5` / `14799.9` nM; `271.7x`).
- Strongest `NFLIKFLLI` separation: `A07` (`4028.6` / `121.6` nM; `33.1x`).
- `pooled_single_output` appears too collapsed on the broad contract; the assay-specific outputs lose useful assay-family discrimination.
- Factorized assay-context residuals are competitive on loss, but the full factorized+segment residual should be judged against probe behavior, not loss alone.
- `split_kd_proxy` is now testable apples-to-apples against merged KD on the same broad contract.