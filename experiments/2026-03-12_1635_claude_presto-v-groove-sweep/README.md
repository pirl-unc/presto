# Presto vs Groove Fair Head-to-Head (v4)

- **Agent**: Claude Code (claude-opus-4-6)
- **Source script**: `scripts/benchmark_presto_bakeoff.py`
- **Created**: 2026-03-12T16:35:40

## Motivation

Bakeoff-v3 (EXP-11) identified Presto P7 as the winner (488x discrimination vs Groove A2's 249x), but had confounds: 2-allele panel, mixed batch sizes, 20 epochs (overfitting), and only one Groove variant. This experiment fixes all four.

## Dataset Contract

- **Panel**: 7 class-I alleles (A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02)
- **Measurement profile**: `numeric_no_qualitative`
- **Qualifier filter**: `all` (censor-aware loss)
- **Synthetics**: none
- **Probes**: SLLQHLIGL, FLRYLLFGI, NFLIKFLLI

## Training

- **Epochs**: 10
- **Batch size**: 256 (all conditions)
- **Seed**: 42
- **Groove LR**: 1e-3
- **Presto LR**: 2.8e-4
- **Warm start** (Presto only): `/checkpoints/mhc-pretrain-20260308b/mhc_pretrain.pt`
- **Presto positional encoding**: `concat_start_end_frac` for both peptide and groove (P04 winner)
- **Presto extras**: `--affinity-assay-residual-mode shared_base_segment_residual --binding-core-lengths 8,9,10,11 --affinity-assay-mode legacy`
- **Requested GPU**: default Modal GPU (not explicitly set to H100!)

## Condition Matrix

| ID | Type | Function | Architecture | LR | What it tests |
|----|------|----------|-------------|-----|---------------|
| G1 | Groove | `groove_baseline_run` | Single IC50 head (~106K params) | 1e-3 | Simplest baseline |
| G2 | Groove | `assay_ablation_run --variant a2` | Multi-head type-routed (~376K params) | 1e-3 | v3 reference |
| G5 | Groove | `assay_ablation_run --variant a5` | Context conditioning (~284K params) | 1e-3 | EXP-07 winner |
| PA | Presto | `focused_binding_run` | assay_heads_only, warm, segres, multicore (~5.2M params) | 2.8e-4 | v3 winner (P7) |
| PF | Presto | `focused_binding_run` | full loss, warm, segres, multicore (~5.2M params) | 2.8e-4 | v3 P6 comparison |

## Key Comparisons

- **G1 vs G2 vs G5**: How much does Groove complexity help?
- **G5 vs PA**: Groove's best (A5) vs Presto's best (assay_heads_only). Core question.
- **PA vs PF**: Does assay_heads_only still beat full loss on 7 alleles?

## Results

All 5 runs completed 10/10 epochs. Train/val split: 32,855 / 8,194 rows.

### Validation Loss (best epoch)

| Condition | Params | Best Val Loss | Best Epoch |
|-----------|--------|--------------|------------|
| G5 | 284K | **0.6575** | 10 |
| G2 | 376K | 1.0254 | 8 |
| PF | 5.15M | 1.4301 | 10 |
| G1 | 53K | 1.6575 | 10 |
| PA | 5.15M | 1.7986 | 10 |

Note: val losses are NOT directly comparable across architectures (different loss functions/heads). Probe discrimination is the apples-to-apples comparison.

### Probe Predictions at Epoch 10 (IC50 nM)

**SLLQHLIGL** (known A\*02:01 binder, literature ~11 nM):

| Allele | G1 | G2 | G5 | PA | PF |
|--------|-----|-----|-----|-----|-----|
| A\*02:01 | 0.01 | 29.0 | **11.5** | 115.5 | 36.1 |
| A\*24:02 | 0.01 | 8,455 | 23,369 | 16,085 | 13,661 |
| A\*03:01 | 0.02 | 11,520 | 10,163 | 12,613 | 18,124 |
| A\*11:01 | 0.02 | 22,776 | 24,096 | 19,711 | 29,068 |
| A\*01:01 | 0.01 | 8,607 | 28,043 | 20,281 | 32,077 |
| B\*07:02 | 0.00 | 8,754 | 19,211 | 9,874 | 24,316 |
| B\*44:02 | 0.00 | 2,293 | 21,510 | 28,787 | 30,457 |

**NFLIKFLLI** (known A\*24:02 binder, literature ~5 nM):

| Allele | G1 | G2 | G5 | PA | PF |
|--------|-----|-----|-----|-----|-----|
| A\*02:01 | 0.01 | 1,454 | 1,859 | 307 | 9,229 |
| A\*24:02 | 0.05 | **8.8** | **4.8** | 2,035 | 146 |
| A\*03:01 | 0.25 | 8,502 | 9,482 | 11,446 | 34,332 |
| A\*11:01 | 0.49 | 22,864 | 11,099 | 20,015 | 39,029 |
| A\*01:01 | 0.33 | 8,077 | 2,100 | 17,221 | 38,833 |
| B\*07:02 | 0.98 | 2,355 | 8,321 | 8,875 | 21,359 |
| B\*44:02 | 0.86 | 1,941 | 325 | 11,088 | 20,597 |

### Allele Discrimination Ratios (mean non-cognate / cognate)

| Peptide | Expected Binder | G1 | G2 | G5 | PA | PF |
|---------|----------------|-----|-----|-----|-----|-----|
| SLLQHLIGL | A\*02:01 | 1x | 359x | **1,832x** | 155x | 682x |
| NFLIKFLLI | A\*24:02 | 10x | 856x | **1,152x** | 6x | 187x |

### Key Findings

1. **G1 (groove MLP) collapsed**: All predictions converge to sub-0.1 nM regardless of allele/peptide. Zero discrimination. Degenerate model.

2. **G5 (Groove A5 context conditioning) is the clear winner**:
   - Best val loss (0.6575)
   - Strongest allele discrimination (1,832x for SLLQHLIGL, 1,152x for NFLIKFLLI)
   - Closest to literature IC50 values (SLLQHLIGL A\*02:01 = 11.5 nM vs expected ~11 nM)
   - Only 284K parameters

3. **PA (Presto assay_heads_only) underperforms**: Weak discrimination (155x / 6x), and NFLIKFLLI A\*24:02 prediction is **wrong direction** (2,035 nM — predicts weak binding when it should be strong). 5.15M params with pretraining did not help.

4. **PF (Presto full loss) is better than PA** but still behind G5: SLLQHLIGL discrimination 682x vs G5's 1,832x. Gets NFLIKFLLI direction correct (146 nM) but with weaker separation.

5. **G2 (Groove A2) is solid second place**: 359x / 856x discrimination, biologically reasonable predictions. Only 376K params.

### Comparisons to v3 Bakeoff

- v3 had PA (then P7) as winner with 488x discrimination, but only on 2 alleles. On 7 alleles, PA drops to 155x.
- G5 was not tested in v3. Its 1,832x discrimination on 7 alleles is the strongest result seen in any bakeoff.
- The confounds in v3 (batch size, epochs, allele count) materially affected the conclusion.

## Artifacts

- Loss curves: [`val_loss_curves.png`](val_loss_curves.png)
- Probe heatmap: [`probe_heatmap.png`](probe_heatmap.png)
- Discrimination ratios: [`discrimination_ratios.png`](discrimination_ratios.png)
- Full metrics: [`summary.json`](summary.json)
- Raw logs: [`launch_logs/`](launch_logs/)

## Notes

- GPU not explicitly pinned to H100! — no hardware evidence found in logs (only `device=cuda`).
- No per-example validation predictions were saved by this launcher; only aggregate loss and probe predictions are available. This is a limitation — future bakeoffs should save per-example predictions.
- All conditions still improving at epoch 10 (no plateau). Longer training may change rankings.
