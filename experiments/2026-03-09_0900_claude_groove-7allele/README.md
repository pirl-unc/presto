# Groove 7-Allele Experiments

**Date**: 2026-03-09 (expA-E, mlp, larger), 2026-03-10 (exact-ic50)
**Agent**: Claude Code (claude-opus-4-6)
**Script**: `scripts/groove_baseline_probe.py` via `scripts/train_modal.py::groove_baseline_run`
**Raw data**: `modal_runs/groove_7allele/`

## Question

How does the GrooveTransformerModel perform on 7-allele class-I binding prediction? What is the effect of contrastive loss, synthetic negatives, curriculum learning, model size, and data filtering?

## Dataset

### "All" filter (expA-E, mlp, larger)
- **Source**: IEDB merged, all qualifier types
- **Alleles (7)**: HLA-A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02
- **Rows**: 41,049 (32,855 train / 8,194 val), split by peptide
- **Allele distribution (train)**: A\*02:01=13,441; A\*03:01=5,149; A\*11:01=4,601; A\*01:01=3,109; B\*07:02=3,019; A\*24:02=2,069; B\*44:02=1,467

### "Exact IC50" filter
- **Rows**: 10,193 (8,166 train / 2,027 val), IC50 measurements with exact qualifiers only
- A\*02:01 dominates (6,989 rows), B\*44:02 tiny (92 rows)

## Training

- **Optimizer**: AdamW (lr=1e-3, weight_decay=0.01)
- **Epochs**: 20, **Batch size**: 256, **Seed**: 42
- **GPU**: A100 (Modal)

## Conditions (8 experiments)

| Experiment | Model | Params | Training mode | Synth |
|-----------|-------|--------|--------------|-------|
| expA Baseline | transformer | 106,241 | regress | none |
| expB Contrastive | transformer | 106,241 | regress+contrastive | none |
| expC Synth | transformer | 106,241 | regress+synth | peptide_scramble+random |
| expD Synth+Contr | transformer | 106,241 | regress+synth+contrastive | peptide_scramble+random |
| expE Curriculum | transformer | 107,144 | 5ep classify -> 10ep regress -> 5ep synth+contr | peptide_scramble+random |
| MLP | mlp | 26,497 | regress | none |
| Larger | transformer | 392,705 | regress (embed=128, hidden=256) | none |
| Exact IC50 | transformer | 106,241 | regress (IC50-only, exact qual) | none |

## Results

### Val Loss

| Experiment | Best Val | @ Epoch | Final Val (e20) |
|-----------|---------|---------|-----------------|
| **Larger** | **0.769** | 18 | 0.807 |
| expC Synth | 0.776 | 9 | 0.838 |
| **expA Baseline** | **0.797** | 12 | 0.819 |
| expE Curriculum | 0.846 | 15 | 1.359 |
| Exact IC50 | 1.046 | 19 | 1.079 |
| expB Contrastive | 1.233 | 18 | 1.343 |
| expD Synth+Contr | 1.216 | 17 | 1.225 |
| MLP | 1.554 | 19 | 1.559 |

Note: Contrastive variants have inflated val loss (val measures regression only, not contrastive reward).

### Probe Discrimination (Epoch 20)

| Experiment | SLLQHLIGL A02 (nM) | A24 (nM) | Ratio | NFLIKFLLI A24 (nM) | A02 (nM) | Ratio |
|-----------|--------------------|---------|----|-------|------|------|
| **Larger** | **8** | 25,173 | 3,147x | 38 | 16,308 | 429x |
| expD Synth+Contr | **11** | 32,693 | 2,972x | **1,337** | 22,882 | **17x FAIL** |
| expA Baseline | 35 | 30,785 | 880x | **10** | 16,020 | 1,602x |
| expC Synth | 46 | 38,383 | 834x | 66 | 32,971 | 500x |
| expB Contrastive | 72 | 30,078 | 418x | **8** | 7,286 | 911x |
| expE Curriculum | 48 | 13,366 | 278x | 37 | 5,979 | 162x |
| Exact IC50 | 61 | 7,561 | 124x | 27 | 1,988 | 74x |
| MLP | 0.001 | 0.001 | **1x FAIL** | 0.001 | 0.001 | **1x FAIL** |

## Takeaways

1. **Baseline transformer (expA) produces the most balanced discrimination**: strong binding (35 nM, 10 nM) with clean 800-1600x separation. Simple regress-only training works best.
2. **Larger transformer gets best val loss (0.769) and SLLQHLIGL binding (8 nM)** but 3.7x more params for modest gains.
3. **MLP is a complete failure**: all predictions collapse to 0.001 nM. Self-attention is essential for groove-peptide interactions.
4. **Synth+contrastive combination is toxic**: expD catastrophically fails NFLIKFLLI (1,337 nM, only 17x ratio).
5. **Synthetic negatives alone mildly hurt probe quality**: NFLIKFLLI weakens from 10 to 66 nM.
6. **Curriculum adds complexity without benefit**: classification saturates in 2 epochs (trivial task).
7. **Exact IC50 filtering (4x less data) weakens discrimination**: ratios drop from 800-1600x to 74-124x.
