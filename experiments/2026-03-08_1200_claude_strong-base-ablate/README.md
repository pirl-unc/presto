# Strong Base Ablation

**Date**: 2026-03-08
**Agent**: Claude Code (claude-opus-4-6)
**Script**: `scripts/focused_binding_probe.py`
**Raw data**: `modal_runs/strong_base_ablate/`, `modal_runs/strong_base_ablate_pull/`

## Question

Which binding architecture variant is the strongest overall: legacy single-head, score_context multi-head, multi-core groove, or peptide-rank contrastive?

## Dataset

- **Source**: IEDB class-I, exact IC50 only
- **Alleles (7)**: HLA-A\*02:01, A\*24:02, A\*03:01, A\*11:01, A\*01:01, B\*07:02, B\*44:02
- **Rows**: 10,193 (8,166 train / 2,027 val)
- **Warm start**: `mhc-pretrain-20260308b`

## Training

- **Epochs**: 12, **Batch size**: 140 (20 slots per allele), balanced batches
- **No synthetic negatives, no contrastive (except peprank variants)**
- **GPU**: A100 (Modal)

## Conditions (5 variants)

| Variant | assay_mode | groove_pos | core_lengths | peprank |
|---------|-----------|------------|-------------|---------|
| legacy_baseline | legacy | sequential | [9] | 0.0 |
| score_context | score_context | sequential | [9] | 0.0 |
| legacy_peprank | legacy | sequential | [9] | 0.5 |
| **legacy_m1** | **legacy** | **triple** | **[8,9,10,11]** | **0.0** |
| legacy_m1_peprank | legacy | triple | [8,9,10,11] | 0.5 |

## Results

### Probe IC50 (at best epoch)

| Variant | SLL A02 | SLL A24 | Ratio | FLR A02 | FLR A24 | NFL A24 | NFL A02 | NFL ratio |
|---------|---------|---------|-------|---------|---------|---------|---------|-----------|
| legacy_baseline | 37 | 28,405 | 764x | 36 | 3,562 | 5.6 | 5,790 | 1,034x |
| score_context | 446 | 11,700 | 26x | 195 | 5,639 | 10.4 | 6,605 | 635x |
| legacy_peprank | 48 | 25,307 | 527x | 39 | 18,080 | 27 | 14,451 | 535x |
| **legacy_m1** | **37** | **29,588** | **800x** | **33** | **24,346** | **6.1** | **5,749** | **942x** |
| legacy_m1_peprank | 29 | 24,179 | 834x | 102 | 6,589 | 2.2 | 16,585 | 7,539x |

### Val Loss (not directly comparable across assay modes)

| Variant | Best Val Loss | @ Epoch |
|---------|--------------|---------|
| score_context | 0.764 | 4 |
| **legacy_baseline** | **1.045** | 12 |
| **legacy_m1** | **1.090** | 6 |
| legacy_peprank | 1.398 | 5 |
| legacy_m1_peprank | 1.477 | 6 |

Note: score_context val loss is lower due to different loss function terms, not better fit.

## Takeaways

1. **`legacy_m1` promoted as winner**: Best FLR A02 binding (33 nM), strong and balanced across all 3 probes, correct polarity on all.
2. **score_context is 5x weaker** on probe IC50 predictions (446 nM SLL vs 37 nM). The extra multi-head assay machinery dilutes the binding signal.
3. **Peptide-rank contrastive** helps on some peptides but badly hurts NFLIKFLLI calibration (A02 inflated to 14-17k nM). Over-separates peptides.
4. **Multi-core triple groove** (M1 upgrade) gives legacy_m1 an edge over legacy_baseline on FLR discrimination (24,346 vs 3,562 A24 IC50).
5. **Key conclusion**: Simple legacy assay head + multi-core groove encoding is the strongest configuration. Architectural complexity (score_context) and contrastive losses (peprank) are net-negative.
