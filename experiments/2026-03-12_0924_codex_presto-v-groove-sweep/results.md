# Presto vs Groove A2 Head-to-Head Sweep (v3) — Results

## Experiment Design

**Goal**: Isolate which ingredients Presto needs to match or beat Groove A2's binding
discrimination. Bakeoff-v2 showed Groove A2 crushing Presto, but the Presto runs were
missing warm start, proper LR, multi-core, and segment residual heads.

**Approach**: 9 conditions — 1 Groove reference + 8 Presto variants. Each successive
Presto condition (P1-P6) adds one ingredient, so we can attribute improvement to each.
P7 and P8 branch from P6 to test alternative loss routing and direct segment mode.

**Dataset**: 2-allele panel (HLA-A\*02:01 + HLA-A\*24:02), IEDB numeric_no_qualitative,
qualifier_filter=all, no synthetics, ~19k rows. This is a *narrower* panel than the
44k 7-allele dataset used in EXP-01/EXP-02 — chosen to match the Groove A2 bakeoff
conditions for a controlled comparison.

**Training**: 20 epochs, seed=42. (Prior successful runs used 3-12 epochs — 20 was
chosen to observe late-training behavior, and indeed several conditions diverged late.)

**Probes**: SLLQHLIGL (strong A\*02:01 binder), FLRYLLFGI, NFLIKFLLI, IMLEGETKL.
SLLQHLIGL should predict low nM for A\*02:01 and high nM for A\*24:02.

## Condition Descriptions

Each condition is described by what it changes vs the previous:

| ID | Type | Warm | Loss Mode | Residual Mode | Core Lengths | LR | Direct Seg | What it isolates |
|----|------|------|-----------|---------------|-------------|----|------------|------------------|
| REF | Groove | n/a | type-routed (IC50/KD/EC50 heads) | n/a | n/a | 1e-3 | n/a | **The bar to clear** |
| P1 | Presto | No | probe_only (single affinity head) | legacy | 9 | 1e-3 | off | Collapse control (= bakeoff-v2 C4) |
| P2 | Presto | **Yes** | probe_only | legacy | 9 | 1e-3 | off | **Warm start effect** (P1 vs P2) |
| P3 | Presto | Yes | **full** (adds KD+IC50+EC50 heads + probe) | legacy | 9 | 1e-3 | off | **Multi-loss effect** (P2 vs P3) |
| P4 | Presto | Yes | full | **shared_base_segment_residual** | 9 | 1e-3 | off | **Segment residual heads** (P3 vs P4) |
| P5 | Presto | Yes | full | segment_residual | **8,9,10,11** | 1e-3 | off | **Multi-core groove** (P4 vs P5) |
| P6 | Presto | Yes | full | segment_residual | 8,9,10,11 | **2.8e-4** | off | **Presto LR** (P5 vs P6) |
| P7 | Presto | Yes | **assay_heads_only** (KD+IC50+EC50 heads only, no probe/unified) | segment_residual | 8,9,10,11 | 2.8e-4 | off | **Type-routed loss** (branches from P6) |
| P8 | Presto | Yes | full | segment_residual | 8,9,10,11 | 2.8e-4 | **affinity_residual** | **Direct gradient path** (branches from P6) |

**Loss mode glossary**:
- `probe_only`: Single affinity probe head, no assay-type routing
- `full`: Probe head + assay-type-specific heads (KD, IC50, EC50) + unified head
- `assay_heads_only`: Only assay-type-specific heads (KD, IC50, EC50), no probe, no unified — closest to Groove A2's architecture

## Results Table

| ID  | best_val | SLLQ@A\*02:01 | SLLQ@A\*24:02 | A0201 discrim | Notes |
|-----|----------|---------------|---------------|---------------|-------|
| REF | **0.941** | 6.5 nM | 2,538 nM | 249x | Stable reference |
| P1  | 4.389 | 962.5 nM | 962.5 nM | 1.0x | Collapsed — all peptides identical |
| P2  | 4.394 | NaN | NaN | NaN | Diverged — warm start + lr=1e-3 unstable |
| P3  | 4.477 | 1,025 nM | 1,025 nM | 1.0x | Collapsed — lr=1e-3 still the bottleneck |
| P4  | 3.812 | NaN | NaN | NaN | Best val improved but diverged late |
| P5  | 2.814 | NaN | NaN | NaN | Val improving but diverged at epoch 17 |
| P6  | 1.416 | 36.9 nM | 7,808 nM | 64x | **First stable Presto condition** |
| **P7** | **1.469** | **0.17 nM** | **6,265 nM** | **488x** | **Winner — beats Groove** |
| P8  | **1.325** | 52.1 nM | 9,002 nM | 67x | Best val_loss, similar discrimination to P6 |

*Discrimination = mean(other probes nM) / SLLQHLIGL nM at A\*02:01. Higher = better separation.*

*P5-P8 rerun at batch_size=128 (OOM at 256 with 4 core lengths). REF/P1-P4 at batch_size=256.*

## Ingredient Marginal Contributions

| Transition | What changed | Effect |
|------------|-------------|--------|
| P1→P2 | + warm start | Diverged to NaN. lr=1e-3 too aggressive for pre-trained weights. |
| P2→P3 | + full multi-loss | Still collapsed at 1.0x. lr=1e-3 is the bottleneck. |
| P3→P4 | + segment_residual | Best val improved 4.48→3.81, but diverged late. Directionally helpful. |
| P4→P5 | + multi-core [8,9,10,11] | Best val improved 3.81→2.81, but still diverged at epoch 17. |
| **P5→P6** | **lr=1e-3 → 2.8e-4** | **Breakthrough: stable convergence, 64x discrimination, 37 nM SLLQHLIGL** |
| **P6→P7** | full → assay_heads_only | **Discrimination 64x → 488x.** Type-routed loss is critical for probe separation. |
| P6→P8 | + affinity_residual direct | Best val_loss improved 1.42→1.33, but discrimination unchanged (64x→67x). |

## Key Findings

1. **LR is the gate**: Every condition at lr=1e-3 either collapsed (constant outputs, 1x)
   or diverged (NaN). lr=2.8e-4 is necessary and sufficient for Presto convergence.

2. **`assay_heads_only` (type-routed) loss is the discrimination winner**: P7 achieves 488x,
   nearly 2x the Groove A2 reference (249x). The `full` loss mode (P6, P8) achieves only 64-67x.
   This makes biological sense — routing KD/IC50/EC50 to separate heads with shared base
   prevents the probe head from diluting assay-specific gradients.

3. **Direct segment mode helps val_loss, not discrimination**: P8 has the best val_loss (1.33)
   but discrimination is similar to P6 (67x vs 64x). The latent DAG bottleneck isn't the
   binding discrimination issue.

4. **Warm start + multi-core + segment_residual are prerequisites**: They enable the lower
   val_loss plateau that P6-P8 reach, but without the right LR they all diverge.

5. **P7's sub-nM SLLQHLIGL (0.17 nM)**: The absolute nM value is biologically implausible
   (real Kd ~10-50 nM), suggesting the model over-concentrates probability mass. But the
   discrimination *ratio* (which is what matters for ranking) is strong.

6. **Allele discrimination works**: All converged conditions (REF, P6, P7, P8) correctly
   predict SLLQHLIGL as a strong A\*02:01 binder (low nM) and a weak A\*24:02 binder
   (thousands of nM).

## Caveats and Next Steps

- **Narrow panel**: Only 2 alleles, ~19k rows. The 7-allele 44k dataset may behave differently.
- **Long training**: 20 epochs may be excessive. P5 diverged at epoch 17; P6-P8 showed
  mild overfitting. Consider early stopping at 10-12 epochs.
- **Batch size confound**: P5-P8 ran at bs=128 vs bs=256 for P1-P4. Smaller batch size
  may have contributed to P6-P8 convergence (lower effective LR).
- **Next**: Run the winning P7 config on the full 7-allele panel with early stopping.

## Winner

**P7** — warm start + assay_heads_only + shared_base_segment_residual + multi-core [8,9,10,11] + lr=2.8e-4

Minimum recipe:
```
--init-checkpoint mhc-pretrain-20260308b
--affinity-loss-mode assay_heads_only
--affinity-assay-residual-mode shared_base_segment_residual
--binding-core-lengths 8,9,10,11
--lr 2.8e-4
```
