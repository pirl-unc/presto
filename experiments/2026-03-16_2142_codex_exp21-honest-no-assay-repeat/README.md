# EXP-21 Honest Repeat Without Assay Inputs

- Agent: `codex`
- Source script: `experiments/2026-03-16_2142_codex_exp21-honest-no-assay-repeat/code/launch.py`
- Created: `2026-03-16T21:43:55.992053`
- Status: `completed`
- Result: `the old EXP-21 groove winner does not survive once assay-selector inputs are removed`

## Dataset Contract

```json
{
  "assay_families": [
    "IC50",
    "KD",
    "KD (~IC50)",
    "KD (~EC50)",
    "EC50"
  ],
  "assay_selector_inputs_forbidden": true,
  "legacy_benchmark_contract": true,
  "measurement_profile": "numeric_no_qualitative",
  "panel": [
    "HLA-A*02:01",
    "HLA-A*24:02"
  ],
  "qualifier_filter": "all",
  "source": "data/merged_deduped.tsv",
  "split_seed": 43
}
```

## Training

```json
{
  "assay_input_mode": "none",
  "batch_size": 256,
  "config_version": "v6",
  "content_conditioned": false,
  "epochs": 50,
  "lr": 0.001,
  "optimizer": "AdamW",
  "requested_gpu": "H100!",
  "seed": 43,
  "weight_decay": 0.01
}
```

## Important Scope Note

This experiment intentionally repeats the **legacy** EXP-21 benchmark family under an honest input contract.

- assay-selector inputs are disabled completely
- the old distributional benchmark head family is otherwise preserved
- this isolates the assay-input leak cleanly
- it does **not** make the legacy benchmark output contract canonical

So this is the right experiment for:

- "did the old groove winner still win without cheating?"

It is not the final answer to:

- "what is the best canonical no-assay-input Presto output contract?"

That second question is still better represented by the honest full-output PF07 path.

## Tested Conditions

```json
[
  {
    "assay_input_mode": "none",
    "cond_id": 2,
    "description": "Old EXP-21 best single run repeated with assay inputs disabled.",
    "encoder_backbone": "groove",
    "epochs": 50,
    "model_key": "groove_c02",
    "seed": 43
  },
  {
    "assay_input_mode": "none",
    "cond_id": 1,
    "description": "Closest groove competitor repeated with assay inputs disabled.",
    "encoder_backbone": "groove",
    "epochs": 50,
    "model_key": "groove_c01",
    "seed": 43
  },
  {
    "assay_input_mode": "none",
    "cond_id": 2,
    "description": "Historical positive control repeated with assay inputs disabled.",
    "encoder_backbone": "historical_ablation",
    "epochs": 50,
    "model_key": "historical_c02",
    "seed": 43
  }
]
```

## Final Result

The best honest run in this legacy benchmark family was:

- `groove c02`
- run id: `dist-ba-v6-honest-groove_c02-c02-ai0-e050-s43`
- test Spearman `0.79951388`
- test AUROC `0.92063254`
- test AUPRC `0.86402327`
- test RMSE log10 `0.93834412`

The full ranking was:

| model | run id | test Spearman | test AUROC | test AUPRC | test RMSE log10 |
| --- | --- | ---: | ---: | ---: | ---: |
| `groove c02` | `dist-ba-v6-honest-groove_c02-c02-ai0-e050-s43` | `0.79951388` | `0.92063254` | `0.86402327` | `0.93834412` |
| `groove c01` | `dist-ba-v6-honest-groove_c01-c01-ai0-e050-s43` | `0.79365140` | `0.91787004` | `0.85915500` | `0.95124024` |
| `historical c02` | `dist-ba-v6-honest-historical_c02-c02-ai0-e050-s43` | `0.79320246` | `0.91938734` | `0.85890377` | `0.95560324` |

## Comparison

Relative to the old cheating EXP-21 winner `dist-ba-v6-confirm-groove_c02-c02-cc0-e050-s43`:

- test Spearman: `-0.05462515`
- test AUROC: `-0.02348608`
- test AUPRC: `-0.05359048`
- test RMSE log10: `+0.11967069`

This is a true apples-to-apples repeat of that single-seed benchmark point:

- same encoder backbone: `groove`
- same `cond_id=2`
- same `50` epochs
- same seed `43`
- same row counts: train `15465`, val `1974`, test `1925`

Relative to the current honest full-output PF07 control in [2026-03-16_1621_codex_pf07-output-tying-weight-sweep](../2026-03-16_1621_codex_pf07-output-tying-weight-sweep/):

- test Spearman: `-0.02012604`
- test AUROC: `-0.00820440`
- test AUPRC: `-0.02113754`
- test RMSE log10: `+0.01749867`

So the honest legacy groove benchmark does **not** beat the current honest PF07 main-Presto control.

## Interpretation

- The old EXP-21 `groove c02` result was materially benefiting from assay-selector inputs.
- Once the assay embedding is forced to zero, `groove c02` remains the best run **within that legacy benchmark family**, but it falls from `0.8541` to `0.7995` test Spearman.
- That honest legacy benchmark is now weaker than the honest PF07 full-output Presto baseline at `0.8196`.
- The previous repo-wide claim that EXP-21 `groove c02` was the active honest model to beat is no longer defensible.

The practical conclusion is:

- treat the old EXP-21 benchmark as a useful historical assay-conditioned result
- do **not** use it as the canonical no-assay-input baseline
- use the PF07 untied control as the current honest baseline until a stronger honest model beats it

## Artifacts

- fetch status: `results/fetch_status.json`
- raw run dirs: `results/runs/`
- condition table: `results/condition_summary.csv`
- epoch table: `results/epoch_summary.csv`
- final probe table: `results/final_probe_predictions.csv`
- plots:
  - `results/test_spearman_ranking.png`
  - `results/test_metric_grid.png`
  - `results/training_curves.png`
  - `results/final_probe_heatmap.png`
- reproducibility bundle: [`reproduce/`](./reproduce/)

## Decision

Do not promote any legacy distributional benchmark run from this family as the active honest baseline.

Keep the honest PF07 untied control as the current no-assay-input model to beat.
