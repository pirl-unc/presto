# PF07 MHC Pretrain Impact Sweep

- Agent: `codex`
- Source script: `experiments/2026-03-17_1212_codex_pf07-mhc-pretrain-impact-sweep/code/launch.py`
- Created: `2026-03-17T12:15:53.376902`

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
  "downstream": {
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
    "requested_gpu": "H100!",
    "synthetic_negatives": false,
    "weight_decay": 0.01
  },
  "pretraining": {
    "batch_size": 192,
    "checkpoint_name": "mhc_pretrain.pt",
    "d_model": 32,
    "mode": "mhc_pretrain",
    "n_heads": 4,
    "n_layers": 2,
    "pretrain_epochs": [
      1,
      2
    ],
    "seed": 42,
    "targets": [
      "chain_type",
      "species",
      "class"
    ]
  }
}
```

## Tested Conditions

```json
[
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_control_constant",
    "description": "Flat honest PF07 control with constant LR",
    "init_checkpoint": "",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000,
    "pretrain_epochs": 0,
    "pretrain_key": "pretrain_0ep"
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_control_constant",
    "description": "Flat honest PF07 control with constant LR",
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000,
    "pretrain_epochs": 1,
    "pretrain_key": "pretrain_1ep"
  },
  {
    "affinity_assay_residual_mode": "shared_base_factorized_context_plus_segment_residual",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_control_constant",
    "description": "Flat honest PF07 control with constant LR",
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e02/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000,
    "pretrain_epochs": 2,
    "pretrain_key": "pretrain_2ep"
  },
  {
    "affinity_assay_residual_mode": "dag_method_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_method_leaf_constant",
    "description": "Method-leaf DAG with constant LR",
    "init_checkpoint": "",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000,
    "pretrain_epochs": 0,
    "pretrain_key": "pretrain_0ep"
  },
  {
    "affinity_assay_residual_mode": "dag_method_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_method_leaf_constant",
    "description": "Method-leaf DAG with constant LR",
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000,
    "pretrain_epochs": 1,
    "pretrain_key": "pretrain_1ep"
  },
  {
    "affinity_assay_residual_mode": "dag_method_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_method_leaf_constant",
    "description": "Method-leaf DAG with constant LR",
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e02/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000,
    "pretrain_epochs": 2,
    "pretrain_key": "pretrain_2ep"
  },
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_prep_readout_leaf_constant",
    "description": "Prep/readout-leaf DAG with constant LR",
    "init_checkpoint": "",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000,
    "pretrain_epochs": 0,
    "pretrain_key": "pretrain_0ep"
  },
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_prep_readout_leaf_constant",
    "description": "Prep/readout-leaf DAG with constant LR",
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000,
    "pretrain_epochs": 1,
    "pretrain_key": "pretrain_1ep"
  },
  {
    "affinity_assay_residual_mode": "dag_prep_readout_leaf",
    "affinity_loss_mode": "full",
    "affinity_target_encoding": "mhcflurry",
    "condition_key": "pf07_dag_prep_readout_leaf_constant",
    "description": "Prep/readout-leaf DAG with constant LR",
    "init_checkpoint": "/checkpoints/mhc-pretrain-d32-20260317a-e02/mhc_pretrain.pt",
    "kd_grouping_mode": "split_kd_proxy",
    "lr": 0.001,
    "lr_schedule": "constant",
    "max_affinity_nM": 100000,
    "pretrain_epochs": 2,
    "pretrain_key": "pretrain_2ep"
  }
]
```

## Results

- Aggregated summaries:
  - [`results/condition_summary.csv`](./results/condition_summary.csv)
  - [`results/condition_summary.json`](./results/condition_summary.json)
  - [`results/summary_bundle.json`](./results/summary_bundle.json)
- Pretrain artifacts:
  - [`results/pretrains/`](./results/pretrains/)
- Raw per-run downstream artifacts:
  - [`results/runs/`](./results/runs/)
- Plots:
  - [`results/test_metric_grid.png`](./results/test_metric_grid.png)
  - [`results/test_spearman_ranking.png`](./results/test_spearman_ranking.png)
- Requested GPU: `H100!`
- Observed peak reserved GPU memory:
  - downstream runs: about `23.52 GiB`
- Parameter counts:
  - `pf07_control_constant`: `411,934`
  - `pf07_dag_method_leaf_constant`: `440,758`
  - `pf07_dag_prep_readout_leaf_constant`: `438,356`
- Fresh MHC pretrain checkpoints:
  - `1` epoch: `/checkpoints/mhc-pretrain-d32-20260317a-e01/mhc_pretrain.pt`
  - `2` epochs: `/checkpoints/mhc-pretrain-d32-20260317a-e02/mhc_pretrain.pt`
- Pretrain quality before finetuning:
  - `1` epoch: val loss `0.1732`, val species acc `0.9105`, val class acc `1.0000`
  - `2` epochs: val loss `0.0943`, val species acc `0.9615`, val class acc `1.0000`

| Condition | Pretrain | Val Spearman | Test Spearman | Test AUROC | Test AUPRC | Test RMSE log10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pf07_control_constant` | `0` | `0.7978` | `0.8155` | `0.9263` | `0.8778` | `0.9215` |
| `pf07_control_constant` | `1` | `0.7818` | `0.8002` | `0.9197` | `0.8556` | `0.9290` |
| `pf07_control_constant` | `2` | `0.7846` | `0.8082` | `0.9255` | `0.8852` | `0.9161` |
| `pf07_dag_method_leaf_constant` | `0` | `0.8219` | `0.8376` | `0.9365` | `0.8981` | `0.8850` |
| `pf07_dag_method_leaf_constant` | `1` | `0.8174` | `0.8227` | `0.9284` | `0.8781` | `0.8962` |
| `pf07_dag_method_leaf_constant` | `2` | `0.8199` | `0.8295` | `0.9326` | `0.8880` | `0.8794` |
| `pf07_dag_prep_readout_leaf_constant` | `0` | `0.8203` | `0.8356` | `0.9333` | `0.8877` | `0.8636` |
| `pf07_dag_prep_readout_leaf_constant` | `1` | `0.8194` | `0.8377` | `0.9352` | `0.8881` | `0.8769` |
| `pf07_dag_prep_readout_leaf_constant` | `2` | `0.8151` | `0.8347` | `0.9330` | `0.8815` | `0.8866` |

## Winner

- Best within-sweep test Spearman:
  - `pf07_dag_prep_readout_leaf_constant` with `1` epoch of MHC pretraining
  - test Spearman `0.8376657`
  - test AUROC `0.9352216`
  - test AUPRC `0.8881480`
  - test RMSE log10 `0.8768588`
- Closest competitor:
  - `pf07_dag_method_leaf_constant` with `0` epochs of pretraining
  - test Spearman `0.8376381`
- Important caveat:
  - this did **not** beat the current honest baseline from
    [`2026-03-16_2355_codex_pf07-assay-structured-dag-sweep`](../2026-03-16_2355_codex_pf07-assay-structured-dag-sweep/),
    which reached test Spearman `0.8381589`

## Comparison

- By model family:
  - `pf07_control_constant`: pretraining hurt at both `1` and `2` epochs
  - `pf07_dag_method_leaf_constant`: both pretrain variants underperformed the no-pretrain control
  - `pf07_dag_prep_readout_leaf_constant`: `1` epoch gave a tiny Spearman lift over `0` epochs (`+0.00209`), but with worse RMSE (`+0.01328`)
- By pretrain duration:
  - mean test Spearman at `0` epochs: `0.82958`
  - mean test Spearman at `1` epoch: `0.82020`
  - mean test Spearman at `2` epochs: `0.82412`
- The pretrain checkpoints themselves were healthy:
  - species/class validation accuracy was already strong after `1` epoch and stronger after `2`
  - the transfer problem is downstream usefulness, not failed pretraining
- Relative to the current honest baseline:
  - best this sweep: `0.8376657`
  - current baseline: `0.8381589`
  - difference: `-0.0004932` Spearman

## Interpretation

- Short MHC class/species pretraining does not give a robust downstream gain on this honest 2-allele PF07 affinity contract.
- The effect is architecture-dependent:
  - clearly harmful for the flat PF07 control
  - harmful for `dag_method_leaf`
  - nearly neutral for `dag_prep_readout_leaf`, with a tiny Spearman lift at `1` epoch but no baseline-changing improvement
- Because the pretrain checkpoints were already high-accuracy on their own task, the likely issue is mismatch between the MHC-only pretrain objective and the downstream seq-only affinity ranking problem, not optimizer failure.
- This result does not justify promoting MHC class/species pretraining as part of the default honest PF07 baseline.

## Takeaway

- Keep the current honest baseline unchanged:
  - `PF07`
  - `dag_prep_readout_leaf`
  - no assay-selector inputs
  - no MHC class/species warm start by default
- If MHC pretraining is revisited, the next version should likely test:
  - longer pretraining plus gentler finetuning LR, or
  - a pretrain objective closer to downstream binding-relevant structure than plain class/species classification

## Reproducibility

- Reproducibility bundle: [`reproduce/`](./reproduce/)
- Launcher: [`code/launch.py`](./code/launch.py)
