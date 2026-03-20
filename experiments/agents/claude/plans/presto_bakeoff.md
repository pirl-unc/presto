# Presto vs Groove Transformer Apples-to-Apples Bakeoff

## Motivation

Current results show Groove A5 val_loss=0.639 vs Presto P01 val_loss=0.888, but these are
confounded by 6+ differences (epochs, warm-start, LR, batch size, params, loss path).
This experiment controls everything except the architectural component under test.

## Standardized Hyperparameters (ALL conditions)

| Parameter | Value |
|-----------|-------|
| epochs | 20 |
| batch_size | 256 |
| lr | 1e-3 |
| d_model / embed_dim | 128 |
| n_layers | 2 |
| n_heads | 4 |
| hidden_dim | 128 |
| alleles | HLA-A*02:01, HLA-A*24:02 |
| probes | SLLQHLIGL, FLRYLLFGI, NFLIKFLLI |
| measurement_profile | numeric_no_qualitative |
| qualifier_filter | all |
| warm-start | None |
| synthetics | Off |
| seed | 42 |

## Condition Matrix

| ID | Name | Script | Key Architectural Delta |
|----|------|--------|------------------------|
| C1 | Groove Transformer | groove_baseline_probe.py | Baseline: 3-segment encoder + single MLP head |
| C2 | Groove + Assay Context (A5) | assay_ablation_probe.py | + assay type/method concat conditioning |
| C3 | Presto probe_only | focused_binding_probe.py | Full Presto encoder + latent DAG, single probe KD head |
| C4 | Presto assay_heads_only | focused_binding_probe.py | + per-type KD/IC50/EC50 routing heads |
| C5 | Presto full loss | focused_binding_probe.py | + multi-loss: probe + unified + per-type |
| C6 | Presto full + score_context | focused_binding_probe.py | + Presto's score_context assay conditioning |

## What Each Transition Isolates

- **C1 -> C2**: Assay context conditioning (A5 style) on groove encoder
- **C1 -> C3**: Total Presto encoder overhead (single-stream encoder + latent DAG + core windows + kinetic decomposition)
- **C3 -> C4**: Per-type assay routing (AssayHeads)
- **C4 -> C5**: Multi-objective loss (adding probe KD + unified KD heads)
- **C5 -> C6**: Presto's score_context assay conditioning
- **C2 vs C6**: Simple assay context (A5 concat) vs rich Presto score_context

## Success Criteria

1. **Primary**: val_loss at epoch 20 (censor-aware, comparable across all conditions)
2. **Probe discrimination**: SLLQHLIGL predicted IC50 for A*02:01 vs A*24:02 (ground truth ratio >= 100x)
3. **Learning dynamics**: Convergence without divergence
4. **"Adds value" definition**: Lower val_loss AND better probe discrimination

## Launcher

```bash
# All conditions
python scripts/benchmark_presto_bakeoff.py

# Subset
python scripts/benchmark_presto_bakeoff.py --condition-ids C1,C3

# Smoke test (1 epoch, single condition)
python scripts/benchmark_presto_bakeoff.py --epochs 1 --condition-ids C1
```

## Follow-Up (if needed)

- If Presto diverges at lr=1e-3: re-run C3-C6 at lr=5e-4
- If Presto matches/beats groove: test with `--binding-direct-segment-mode affinity_residual`
- If groove wins everywhere: consider simplifying Presto's DAG for pure binding affinity
