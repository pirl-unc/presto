# Assay Head Ablation (A1-A8)

**Date**: 2026-03-11
**Agent**: Claude Code (claude-opus-4-6)
**Script**: `scripts/assay_ablation_probe.py` via `scripts/train_modal.py::assay_ablation_run`
**Launcher**: `scripts/benchmark_assay_ablation.py`
**Raw data**: `modal_runs/assay_ablation/`

## Question

Which Presto binding head components (kinetics, residual corrections, assay context, blend) add value on top of a shared GrooveTransformerModel encoder?

## Dataset

- **Source**: IEDB via `merged_deduped.tsv`
- **Alleles (7)**: HLA-A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02
- **Measurement profile**: `numeric_no_qualitative` (IC50 + KD + EC50, no qualitative)
- **Qualifier filter**: `all` (exact + censored; 30,112 exact + 10,937 censored)
- **Rows**: 41,049 total (32,855 train / 8,194 val), split by peptide
- **Synthetic negatives**: OFF
- **Allele distribution (train)**: A\*02:01=13,441; A\*03:01=5,149; A\*11:01=4,601; A\*01:01=3,109; B\*07:02=3,019; A\*24:02=2,069; B\*44:02=1,467
- **Assay type distribution**: KD(~IC50)=12,989; IC50=8,923; KD(~EC50)=8,542; KD=2,097; EC50=304 (note: KD is severely underrepresented at ~14/batch for typed variants)

## Training

- **Encoder**: AblationEncoder (shared transformer, 2 layers, 4 heads, embed_dim=128, ff_dim=128)
- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Epochs**: 20, **Batch size**: 256, **Seed**: 42
- **Gradient clipping**: max_norm=1.0
- **Loss**: censor-aware regression; typed variants (A2, A3, A7) route IC50/KD/EC50 to separate heads
- **Consistency weight** (A7 only): 0.1
- **GPU**: A100 (Modal default)

## Conditions (8 variants)

| Variant | Description | Params | Architecture |
|---------|------------|--------|-------------|
| A1 | Single IC50 MLP (reference) | 277,505 | encoder -> MLP -> log10(IC50) |
| A2 | Multi-type routed heads | 376,323 | 3 parallel MLPs (IC50, KD, EC50), routed by type |
| A3 | Shared base + residual | 326,917 | Shared KD head + bounded softsign residuals for IC50/EC50 |
| A4 | Kinetic decomposition | 253,957 | 3 kinetic heads (koff, kon_int, kon_chap) -> derived KD -> IC50 |
| A5 | Assay context conditioning | 284,001 | A1 + assay type/method embedding (16-d each) concatenated to input |
| A6 | Probe + kinetic blend | 278,661 | Dual-path (direct MLP + kinetic KD), sigmoid-blended (init ~0.75 probe) |
| A7 | Multi-head + consistency | 376,323 | A2 + Huber penalty on |KD - IC50| > log10(2) |
| A8 | Type indicator feature | 278,401 | A1 + 7-d one-hot measurement type prepended to MLP |

## Results

### Val Loss

| Variant | Best Val Loss | @ Epoch | Final Train | Final Val |
|---------|--------------|---------|-------------|-----------|
| **A5** | **0.639** | 8 | 0.206 | 0.648 |
| **A8** | **0.748** | 14 | 0.236 | 0.764 |
| A3 | 0.770 | 13 | 0.280 | 0.827 |
| A2 | 0.792 | 6 | 0.224 | 0.817 |
| A6 | 0.800 | 15 | 0.257 | 0.819 |
| A7 | 0.801 | 15 | 0.229 | 0.829 |
| A1 | 0.802 | 13 | 0.265 | 0.807 |
| A4 | 0.874 | 17 | 0.339 | 0.895 |

### Probe Discrimination (Epoch 20)

**SLLQHLIGL** (known HLA-A\*02:01 binder):

| Variant | A\*02:01 IC50 (nM) | Best non-target (nM) | Selectivity ratio |
|---------|--------------------|--------------------|-------------------|
| A6 | 6.7 | 42,540 | 6,796x |
| A8 | 26 | 21,447 | 1,253x |
| A1 | 55 | 7,848 | 510x |
| A4 | 38 | 2,247 | 469x |
| A5 | 64 | 1,305 | 366x |
| A2 | 68 | 1,995 | 390x |
| A3 | 189 | 11,626 | 101x |
| A7 | 270 | 1,721 | 108x |

**NFLIKFLLI** (known HLA-A\*24:02 binder):

| Variant | A\*24:02 IC50 (nM) | Best non-target (nM) | Selectivity ratio |
|---------|--------------------|--------------------|-------------------|
| A5 | 4.3 | 181 (B44!) | 42x |
| A7 | 4.8 | 773 (B44!) | 161x |
| A1 | 14 | 13,616 | 943x |
| A3 | 16 | 1,096 (B44!) | 69x |
| A6 | 18 | 6,103 | 330x |
| A2 | 26 | 900 (B44!) | 35x |
| A4 | 31 | 509 | 16x |
| A8 | 40 | 2,338 (B44!) | 58x |

**Cross-reactivity concern**: Multiple typed variants (A2, A3, A5, A7) predict low NFLIKFLLI IC50 for B\*44:02 (181-1,096 nM). This is likely a false positive from multi-head routing leaking signal.

### A6 Mix Weight Trajectory
- Epoch 1: 0.747 (probe-dominant as initialized)
- Epoch 20: 0.697 (slowly shifting toward kinetics, but probe path remains dominant)

### A7 Consistency Penalty
- Epoch 1: 0.098
- Epoch 20: 0.032 (heads converging but not fully aligned)

## Takeaways

1. **A5 (assay context conditioning) is the clear winner on val loss** (0.639 vs 0.802 baseline). Knowing what assay type/method produced each measurement gives a huge advantage. This is a ~20% reduction in loss.
2. **A8 (type indicator)** is second best (0.748). Even a simple one-hot of measurement type helps substantially -- the minimal version of context conditioning.
3. **A3 (shared base + residual)** third (0.770). The `shared_base_segment_residual` pattern helps but less than context awareness.
4. **A4 (kinetic decomposition) is worst** (0.874). Imposing the physics constraint without kinetic supervision is net-negative -- the encoder can't learn meaningful kinetic parameters from IC50 data alone.
5. **A2, A6, A7 cluster near A1** (0.792-0.801 vs 0.802). Multi-head routing and consistency reg don't meaningfully beat the single-head baseline.
6. **Key insight**: Assay metadata is far more valuable than architectural complexity for binding prediction. A5's context conditioning should be prioritized for integration into the Presto DAG.
7. **Caution**: Typed/multi-head variants show B\*44:02 cross-reactivity leakage for NFLIKFLLI. This pattern doesn't appear in A1 or A6.
