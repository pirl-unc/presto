# Presto vs Groove Bakeoff Results

## Loss Summary

| Cond | Name | Epochs | Train Loss | Val Loss |
|------|------|--------|------------|----------|
| C1 | Groove Transformer | 20/20 | 1.6157 | 1.8198 |
| C2 | Groove + A5 Context | 20/20 | 0.2267 | **0.7919** |
| C3 | Presto probe_only | 16/16 | 3.5826 | 4.3930 |
| C4 | Presto assay_heads | 10/17 (NaN) | 4.0394 | 4.3677 |
| C5 | Presto full loss | 11/17 (NaN) | 3.9191 | 4.7232 |
| C6 | Presto full+score_ctx | 10/10 | 3.6579 | 4.1450 |

## Probe Discrimination (SLLQHLIGL)

Ground truth: A*02:01 is a strong binder (~6 nM), A*24:02 is a non-binder (~10,000+ nM).

| Cond | A*02:01 IC50 (nM) | A*24:02 IC50 (nM) | Ratio |
|------|-------------------|-------------------|-------|
| C1 | 7.6 | 182.4 | 24x |
| C2 | 46.5 | 25,060 | **539x** |
| C3 | 3,790 | 3,790 | **1.0x** |
| C4 | 1,668 | 1,668 | **1.0x** |
| C5 | 810 | 810 | **1.0x** |
| C6 | 1,566 | 1,566 | **1.0x** |

## Key Findings

### 1. Presto produces completely degenerate predictions (C3-C6)
All four Presto conditions output the **exact same IC50** for every peptide x allele combination.
Zero allele discrimination, zero peptide discrimination. The model has collapsed to a constant
function at the probe prediction level.

### 2. Groove + A5 Context (C2) is the clear winner
- Best val_loss: 0.79 (vs Presto best of 4.15)
- Best probe discrimination: 539x SLLQHLIGL ratio (ground truth direction)
- Strong peptide differentiation: FLRYLLFGI=26 nM, NFLIKFLLI=5,716 nM, IMLEGETKL=30 nM for A*02:01
- IMLEGETKL correctly predicted as non-binder for A*24:02 (34,443 nM)

### 3. Plain Groove (C1) shows partial discrimination
- 24x ratio for SLLQHLIGL -- right direction but insufficient
- No peptide differentiation for FLRYLLFGI/NFLIKFLLI (both ~0.5 nM for both alleles)

### 4. Presto training instability
- C4 (assay_heads_only) diverged to NaN at epoch 11
- C5 (full loss) diverged to NaN at epoch 12
- C3 and C6 ran but never learned anything discriminative

## Diagnosis: Why Presto Fails

The constant-output collapse in C3-C6 points to:
1. **Information bottleneck**: Presto's latent DAG, kinetic decomposition, and core window mechanism
   prevent gradient flow from the IC50 loss to the MHC representation
2. **Over-parameterization**: 4.8M params (vs 53K-284K for groove) with lr=1e-3 on ~15K rows
   likely causes the encoder to memorize training batches while the probe head sees no gradient
3. **Probe architecture mismatch**: The probe evaluates via IC50 head which derives from kinetic
   latents (koff, kon) that are themselves derived from the latent DAG -- too many indirections
   for signal to reach the encoder

## Recommendations

1. **For pure binding affinity**: Use Groove + A5 architecture. It's simpler, faster, and dramatically
   more accurate.
2. **Presto's DAG may still have value** for multi-task (presentation, immunogenicity) but is
   counterproductive for single-task affinity prediction at this scale.
3. **If Presto is retained**: The information bottleneck between MHC tokens and IC50 output must be
   fixed (direct segment bypass, residual connections from groove to output).
