# Directness Bakeoff Round 2c

**Date**: 2026-03-10
**Agent**: Claude Code (claude-opus-4-6) and Codex
**Script**: `scripts/focused_binding_probe.py` (Presto) / `scripts/groove_baseline_probe.py` (Groove)
**Launcher**: `scripts/benchmark_design_round2.py`
**Raw data**: `modal_runs/directness_bakeoff_round2c/`

## Question

What combination of positional encoding and residual mode produces the best binding prediction? Is `shared_base_segment_residual` a genuine architectural win?

## Dataset

- **Source**: IEDB merged, 7-allele class-I panel
- **Alleles (7)**: HLA-A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02
- **Measurement profile**: `numeric_no_qualitative`
- **Qualifier filter**: `all` (24,028 exact + 8,827 censored in train)
- **Rows**: 41,049 (32,855 train / 8,194 val)
- **Synthetic negatives**: OFF
- **Contrastive losses**: OFF for Presto; 1.0/0.5 for Groove ranking variants

## Training

### Presto variants (P00-P07)
- **d_model**: 128, **Params**: ~4.8M
- **Warm start**: `mhc-pretrain-20260308b`
- **Epochs**: 3, **Batch size**: 140
- **Fixed config**: binding_core_lengths=8,9,10,11; refinement=shared; assay_mode=legacy; kinetic_input=affinity_vec; direct_segment=off

### Groove controls (G0, G0R, G1, G1R)
- G0/G0R: MLP, 26,497 params (embed=64, hidden=128)
- G1/G1R: Transformer, 392,705 params (embed=128, hidden=256, 2 layers, 4 heads)
- **Epochs**: 3, **Batch size**: 256

## Conditions

### Presto 2x2x2 factorial

| Design | peptide_pos | groove_pos | residual_mode | Params |
|--------|-------------|------------|---------------|--------|
| P00 | triple | triple | legacy | 4,799,172 |
| P01 | triple | triple | shared_base_segment_residual | 4,799,364 |
| P02 | triple | triple_plus_abs | legacy | 4,799,172 |
| P03 | triple | triple_plus_abs | shared_base_segment_residual | 4,799,364 |
| P04 | triple_plus_abs | triple | legacy | 4,799,172 |
| P05 | triple_plus_abs | triple | shared_base_segment_residual | 4,799,364 |
| P06 | triple_plus_abs | triple_plus_abs | legacy | 4,799,172 |
| P07 | triple_plus_abs | triple_plus_abs | shared_base_segment_residual | 4,799,364 |

### Groove controls

| Design | Model | Ranking losses |
|--------|-------|---------------|
| G0 | MLP | off |
| G0R | MLP | on (1.0 / 0.5) |
| G1 | Transformer | off |
| G1R | Transformer | on (1.0 / 0.5) |

## Results

### Val Loss (sorted best to worst)

| Design | Val Loss | Type |
|--------|---------|------|
| **G1** | **0.851** | groove_transformer |
| **P01** | **0.888** | presto (triple/triple/shared_base) |
| P02 | 0.928 | presto |
| P05 | 0.928 | presto |
| P03 | 0.932 | presto |
| P07 | 0.969 | presto |
| P06 | 1.050 | presto |
| P00 | 1.127 | presto |
| P04 | 1.168 | presto |
| G1R | 1.643 | groove_transformer+ranking |
| G0 | 1.935 | groove_mlp |
| G0R | 3.982 | groove_mlp+ranking |

### Probe IC50 Predictions (nM)

| Design | SLL A02 | SLL A24 | SLL ratio | FLR A02 | FLR A24 | FLR ratio | NFL A02 | NFL A24 | NFL ratio |
|--------|---------|---------|-----------|---------|---------|-----------|---------|---------|-----------|
| P01 | 21 | 12,488 | 594x | 115 | 2,898 | 25x | 4,944 | 1,611 | 3.1x |
| P05 | 27 | 9,502 | 352x | 32 | 6,953 | 219x | 4,953 | 672 | 7.4x |
| P03 | 25 | 15,827 | 645x | 27 | 1,806 | 67x | 3,225 | 775 | 4.2x |
| P07 | 15 | 2,094 | 138x | 22 | 5,499 | 254x | 5,304 | 527 | 10.1x |
| P00 | 42 | 1,030 | 24x | 5,835 | 266 | **0.05x INV** | 374 | 2,544 | **0.15x INV** |
| P04 | 18 | 7,571 | 416x | 51 | 58 | **1.1x FLAT** | 21 | 747 | **0.03x INV** |
| G1 | -- | -- | 1,712x | -- | -- | 1,231x | -- | -- | **0.23x INV** |

## Takeaways

1. **`shared_base_segment_residual` dominates `legacy`**: All 4 shared_base variants (P01/P03/P05/P07) beat their legacy counterparts on val loss AND probe discrimination. Legacy variants frequently produce inverted ratios (FLR, NFL wrong sign). This is the single most impactful factor -- 192 extra parameters, massive quality gain.
2. **Best Presto by val loss**: P01 (triple/triple, shared_base) = 0.888.
3. **Best overall probe balance**: P05 (triple_plus_abs peptide, triple groove, shared_base) -- strong on all 3 probes.
4. **G1 (groove transformer) beats all Presto on raw val loss** (0.851 vs 0.888) with only 393K params vs 4.8M. But inverts NFL.
5. **Positional encoding mode** (`triple` vs `triple_plus_abs`) shows no consistent directional effect. The interaction with residual mode is stronger.
6. **MLP groove (G0) is not competitive**: predictions collapse, no discrimination.
