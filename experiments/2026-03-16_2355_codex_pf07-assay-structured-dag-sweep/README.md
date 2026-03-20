# PF07 Assay-Structured DAG Sweep

- Agent: `codex`
- Source script: `experiments/2026-03-16_2355_codex_pf07-assay-structured-dag-sweep/code/launch.py`
- Created: `2026-03-16T23:27:01.154527`

## Dataset Contract

```json
{
  "alleles": [
    "HLA-A*02:01",
    "HLA-A*24:02"
  ],
  "assay_families_supervised": [
    "IC50",
    "KD",
    "KD(~IC50)",
    "KD(~EC50)",
    "EC50"
  ],
  "assay_selector_inputs_forbidden": true,
  "input_fields": [
    "nflank",
    "peptide",
    "cflank",
    "mhc_a",
    "mhc_b"
  ],
  "measurement_profile": "numeric_no_qualitative",
  "probe_peptides": [
    "SLLQHLIGL",
    "FLRYLLFGI",
    "NFLIKFLLI",
    "IMLEGETKL"
  ],
  "qualifier_filter": "all",
  "source": "data/merged_deduped.tsv",
  "split_policy": "peptide_group_80_10_10",
  "split_seed": 42,
  "train_seed": 43
}
```

## Training

```json
{
  "affinity_loss_mode": "full",
  "affinity_target_encoding": "mhcflurry",
  "batch_size": 256,
  "binding_core_lengths": [
    8,
    9,
    10,
    11
  ],
  "binding_core_refinement": "shared",
  "binding_direct_segment_mode": "off",
  "binding_kinetic_input_mode": "affinity_vec",
  "d_model": 32,
  "epochs": 50,
  "groove_pos_mode": "concat_start_end_frac",
  "kd_grouping_mode": "split_kd_proxy",
  "max_affinity_nM": 100000,
  "n_heads": 4,
  "n_layers": 2,
  "optimizer": "AdamW",
  "peptide_pos_mode": "concat_start_end_frac",
  "probe_artifact_schema": [
    "KD_nM",
    "IC50_nM",
    "EC50_nM",
    "KD_proxy_ic50_nM",
    "KD_proxy_ec50_nM",
    "binding_affinity_probe_kd"
  ],
  "ranking_losses": false,
  "requested_gpu": "H100!",
  "synthetic_negatives": false,
  "warm_start": "",
  "weight_decay": 0.01
}
```

## Tested Conditions

```json
[
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_control",
    "description": "Honest PF07 control with flat assay residual heads",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_family",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_family",
    "description": "Output-side family-anchor DAG: KD -> {IC50 family, EC50 family} -> leaf heads",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_method_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_method_leaf",
    "description": "Family-anchor DAG with output-side assay-method leaves for IC50/EC50",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  },
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_prep_readout_leaf",
    "description": "Family-anchor DAG with factorized output-side prep/readout leaves for IC50/EC50",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000
  }
]
```

## Results

- Aggregated summaries:
  - [`results/condition_summary.csv`](./results/condition_summary.csv)
  - [`results/summary_bundle.json`](./results/summary_bundle.json)
- Raw per-run artifacts:
  - [`results/runs/`](./results/runs/)
- Requested GPU: `H100!`
- Observed peak reserved GPU memory:
  - control / `dag_family`: about `23.52 GiB`
  - `dag_method_leaf` / `dag_prep_readout_leaf`: about `23.52 GiB`
- Largest parameter counts:
  - control: `411,934`
  - `dag_family`: `415,078`
  - `dag_method_leaf`: `440,758`
  - `dag_prep_readout_leaf`: `438,356`

| Condition | Residual mode | Val Spearman | Test Spearman | Test AUROC | Test AUPRC | Test RMSE log10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pf07_control` | `shared_base_factorized_context_plus_segment_residual` | `0.7920` | `0.8131` | `0.9243` | `0.8782` | `0.9377` |
| `pf07_dag_family` | `dag_family` | `0.8072` | `0.8208` | `0.9263` | `0.8739` | `0.9088` |
| `pf07_dag_method_leaf` | `dag_method_leaf` | `0.8247` | `0.8360` | `0.9357` | `0.8853` | `0.8723` |
| `pf07_dag_prep_readout_leaf` | `dag_prep_readout_leaf` | `0.8230` | `0.8382` | `0.9359` | `0.8781` | `0.8704` |

## Winner

- Best primary metric: `pf07_dag_prep_readout_leaf`
- Held-out test metrics:
  - Spearman `0.8381589`
  - Pearson `0.8329636`
  - AUROC `0.9358775`
  - AUPRC `0.8780973`
  - RMSE log10 `0.8703900`
- Best within-sweep improvements versus the flat PF07 control:
  - `+0.02506` test Spearman
  - `+0.01162` AUROC
  - `-0.06728` RMSE log10
  - AUPRC was effectively flat (`-0.00010`)

## Comparison

- `dag_family` helps, but only modestly:
  - `+0.00769` Spearman vs control
  - `-0.02885` RMSE log10
- `dag_method_leaf` is strong and has the best AUPRC:
  - test Spearman `0.8360`
  - test AUPRC `0.8853`
  - best final validation loss of the four (`0.031908`)
- `dag_prep_readout_leaf` wins on the primary ranking/regression metrics:
  - slightly higher test Spearman than `dag_method_leaf` (`+0.00216`)
  - slightly lower RMSE log10 (`-0.00186`)
- Relative to the previous honest baseline from the untied PF07 sweep in
  [`2026-03-16_1621_codex_pf07-output-tying-weight-sweep`](../2026-03-16_1621_codex_pf07-output-tying-weight-sweep/):
  - `+0.01852` test Spearman
  - `+0.00704` AUROC
  - `-0.05046` RMSE log10
  - AUPRC is lower by `0.00706`

## Interpretation

- Output-side assay structure helps once it is expressed as output routing instead of forbidden assay-selector inputs.
- The coarse `dag_family` anchor is not enough on its own; the gain comes from leaf structure.
- Both leaf variants outperform the flat honest PF07 control, which suggests the seq-only model was bottlenecked by too-coarse output sharing rather than by lack of assay metadata as an input.
- `dag_method_leaf` generalized best on validation and AUPRC, but `dag_prep_readout_leaf` is the better overall test-time baseline on Spearman and RMSE.
- The old assay-conditioned results are still stronger in absolute AUPRC, but this is now the strongest verified no-assay-input model in the repo.

## Takeaway

- Promote `dag_prep_readout_leaf` as the current honest baseline to beat for the 2-allele broad-numeric affinity contract.
- Keep `dag_method_leaf` as the closest structural comparator.
- The next reasonable follow-up is not more flat PF07 tuning; it is either:
  - seed confirmation on `dag_prep_readout_leaf`, or
  - extending the same output-side DAG idea to richer assay families beyond affinity.

## Reproducibility

- Reproducibility bundle: [`reproduce/`](./reproduce/)
- Launcher: [`code/launch.py`](./code/launch.py)
