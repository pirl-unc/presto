# Claude Ideas

Use this file for rough experimental hypotheses and brainstorms.

- Keep items short.
- Promote actionable items into `todo.md`.
- Promote non-trivial items into `plans/`.

## Dual-head architecture: independent quant + qual heads with shared encoder

**Date**: 2026-03-13
**Context**: Emerged from distributional BA heads v1/v2 experiments (EXP-11). Fine-grained distributional bins (HL-Gauss, Two-Hot) don't beat MHCflurry regression. The insight: instead of distributing over continuous IC50 bins, separate quantitative and qualitative measurements into independent heads.

**Architecture**:
```
encoder(pep, mhc) → h (shared latent)
h → quant_head (+ quant_assay_emb) → IC50       [trained on quantitative data]
h → qual_head  (+ qual_assay_emb)  → pos/neg    [trained on qualitative data]
```

**Key points**:
- Qualitative assays (ELISPOT, multimer staining) aren't "imprecise IC50" — they're fundamentally different measurements needing their own assay embeddings and decoder
- Both losses backprop through shared encoder; qualitative task acts as auxiliary regularizer enriching `h` with abundant qualitative data
- No explicit dependency between predictions — coupling is through shared encoder
- Quant → qual is thermodynamically correct (IC50 determines binding status), but independent-given-h is cleanest and avoids DAG complexity
- Qualitative data is abundant — this could significantly increase effective training data
- Follow-up variant to test: qual logits passed as features to quant decoder (qual → quant coupling), useful if encoder is small

**Relation to v1/v2 findings**:
- D1-affine collapsed because target depended on parameters (self-referential optimization)
- D2-logit works but doesn't beat regression (~0.015 Spearman gap)
- K=16 bins is the sweet spot for distributional heads — very coarse bins work as well as fine ones
- MHCflurry additive regression remains best point predictor (Spearman ~0.81)
- Gaussian and Quantile heads are competitive (~0.795 and ~0.792) and provide uncertainty for free
