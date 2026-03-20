# Hardware Generalization Bakeoff

- Agent: `codex`
- Source script: `scripts/benchmark_hardware_generalization.py`
- Created: `2026-03-12T19:02:45.466269`

## Dataset Contract

```json
{
  "alleles": [
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-A*03:01",
    "HLA-A*11:01",
    "HLA-A*01:01",
    "HLA-B*07:02",
    "HLA-B*44:02"
  ],
  "assays": [
    "IC50",
    "KD",
    "KD(~IC50)",
    "KD(~EC50)",
    "EC50"
  ],
  "measurement_profile": "numeric_no_qualitative",
  "qualifier_filter": "all"
}
```

## Training

```json
{
  "batch_size": 128,
  "epochs": 20,
  "gpu_matrix": [
    "A100",
    "H100!",
    "H200"
  ],
  "ranking": false,
  "synthetics": false,
  "warm_start": "/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt"
}
```

## Tested Conditions

```json
[
  {
    "design_id": "A03_log10_100k_warmup_cosine_lr1e-4",
    "gpu": "A100",
    "lr": "1e-4",
    "schedule": "warmup_cosine"
  },
  {
    "design_id": "A03_log10_100k_warmup_cosine_lr1e-4",
    "gpu": "H100!",
    "lr": "1e-4",
    "schedule": "warmup_cosine"
  },
  {
    "design_id": "A03_log10_100k_warmup_cosine_lr1e-4",
    "gpu": "H200",
    "lr": "1e-4",
    "schedule": "warmup_cosine"
  },
  {
    "design_id": "A07_mhcflurry_100k_warmup_cosine_lr2p8e-4",
    "gpu": "A100",
    "lr": "2.8e-4",
    "schedule": "warmup_cosine"
  },
  {
    "design_id": "A07_mhcflurry_100k_warmup_cosine_lr2p8e-4",
    "gpu": "H100!",
    "lr": "2.8e-4",
    "schedule": "warmup_cosine"
  },
  {
    "design_id": "A07_mhcflurry_100k_warmup_cosine_lr2p8e-4",
    "gpu": "H200",
    "lr": "2.8e-4",
    "schedule": "warmup_cosine"
  }
]
```

## Notes

- Reproducibility bundle: [`reproduce/`](./reproduce/)

## Assay Label -> Output Mapping

- `IC50` -> `assays.IC50_nM`
- direct `KD` -> direct KD output
- `KD (~IC50)` -> proxy KD(~IC50) output
- `KD (~EC50)` -> proxy KD(~EC50) output
- `EC50` -> `assays.EC50_nM`
- `A03` target space:
  - `log10`, max `100,000 nM`
- `A07` target space:
  - `mhcflurry`, max `100,000 nM`

## Results

| design | gpu | best epoch | best val | term val | setup s | epoch s | gpu util % | peak alloc GiB | SLL A02 / A24 | SLL ratio | FLR A02 / A24 | FLR ratio | NFL A02 / A24 | NFL ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | --- | ---: |
| `A03` | `A100` | 16 | 0.7831 | 0.8021 | 53.2 | 105.0 | 51.1 | 16.40 | 52.0 / 64333.5 | 1238.1 | 28.6 / 300.1 | 10.5 | 20370.3 / 10.0 | 2036.9 |
| `A03` | `H100!` | 18 | 0.8091 | 0.8260 | 35.1 | 58.9 | 58.6 | 16.44 | 51.3 / 39359.1 | 766.7 | 43.5 / 252.9 | 5.8 | 18953.1 / 13.2 | 1438.8 |
| `A03` | `H200` | 10 | 0.8138 | 0.8366 | 77.2 | 63.8 | 48.2 | 16.44 | 48.4 / 40528.7 | 837.7 | 58.9 / 135.3 | 2.3 | 12354.5 / 9.9 | 1246.3 |
| `A07` | `A100` | 10 | 0.0266 | 0.0271 | 52.3 | 110.9 | 49.0 | 16.40 | 116.1 / 29635.2 | 255.3 | 72.0 / 1172.8 | 16.3 | 3039.8 / 17.4 | 174.2 |
| `A07` | `H100!` | 13 | 0.0262 | 0.0273 | 41.6 | 73.0 | 48.2 | 16.44 | 69.2 / 21981.9 | 317.8 | 46.8 / 676.6 | 14.5 | 2424.4 / 106.5 | 22.8 |
| `A07` | `H200` | 15 | 0.0253 | 0.0260 | 43.2 | 71.5 | 44.1 | 16.44 | 151.5 / 31054.7 | 205.0 | 84.7 / 2668.6 | 31.5 | 3509.6 / 18.2 | 193.0 |

## Interpretation

### What generalized from the earlier hardware bakeoff

- `H100!` is still the best default hardware for runtime.
  - `A03`: `58.9s/epoch` on `H100!` vs `105.0s` on `A100`
  - `A07`: `73.0s/epoch` on `H100!` vs `110.9s` on `A100`
- `H200` is viable, but not a clear runtime win over `H100!`.

### What did **not** generalize cleanly

- At fixed batch size `128`, the earlier claim that `H100!` optimizes better than `A100` is not universally true.
- For `A03`, `A100` produced the strongest final broad probe behavior despite being much slower:
  - `SLLQHLIGL`: `52.0 / 64333.5`
  - `NFLIKFLLI`: `20370.3 / 10.0`
- For `A07`, `H200` achieved the best validation loss, while `H100!` remained the speed leader and `A100` stayed competitive on some probe ratios.

### Practical takeaway

- Use `H100!` as the default Modal GPU because it is still the best speed / reliability choice.
- Do **not** assume that `H100!` always gives the best optimization trajectory once the model also fits comfortably on `A100`.
- Hardware remains part of the experiment contract:
  - runtime comparisons should prefer `H100!`
  - accuracy claims across GPU families need to be checked, not assumed

## Relative To Earlier Hardware Bakeoff

Compared with [2026-03-12_1209_codex_modal-hardware-bakeoff](../2026-03-12_1209_codex_modal-hardware-bakeoff/README.md):

- Earlier bakeoff used batch size `140` and showed:
  - `DP00` / `DP01` OOM on `A100`
  - `H100!` clearly better than `A100` on both speed and the observed `A03` optimization trajectory
- This follow-up reduced batch size to `128` so all three GPUs fit comfortably.
- Result:
  - the speed conclusion survives
  - the optimization conclusion only partially survives

## Artifacts

- [options_vs_perf.md](./options_vs_perf.md)
- [options_vs_perf.json](./options_vs_perf.json)
- [analysis/parsed_metrics.csv](./analysis/parsed_metrics.csv)
- [analysis/parsed_metrics.json](./analysis/parsed_metrics.json)
- [analysis/per_epoch_metrics.csv](./analysis/per_epoch_metrics.csv)
- [analysis/per_epoch_metrics.json](./analysis/per_epoch_metrics.json)
- [analysis/summary.json](./analysis/summary.json)
- [analysis/a03_val_loss_by_epoch.png](./analysis/a03_val_loss_by_epoch.png)
- [analysis/a03_probe_ratios_by_epoch.png](./analysis/a03_probe_ratios_by_epoch.png)
- [analysis/a07_val_loss_by_epoch.png](./analysis/a07_val_loss_by_epoch.png)
- [analysis/a07_probe_ratios_by_epoch.png](./analysis/a07_probe_ratios_by_epoch.png)
- [analysis/speed_vs_best_val.png](./analysis/speed_vs_best_val.png)
- [analysis/final_probe_ratios_by_condition.png](./analysis/final_probe_ratios_by_condition.png)
