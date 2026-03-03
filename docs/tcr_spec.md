# TCR Encoder Specification (Future Feature)

This document specifies the planned optional TCR pathway and its intended
integration into Presto.

**Canonical status (current):** TCR-conditioned recognition/matching is not active
in canonical training or inference. TCR inputs are reserved for future work.

Model architecture is in `design.md`. Training strategy is in `training_spec.md`.

---

# 1. Planned Behavior

When enabled in a future release, TCR support will:
1. Encode TCR alpha/beta chains into a shared representation (`tcr_vec`).
2. Optionally modulate recognition latents via gated cross-attention.
3. Compute a TCR-specific pMHC match score (`match_logit`).
4. Support optional chain/cell auxiliary supervision.

Current canonical behavior:
- recognition remains peptide-only,
- no TCR-conditioned matching output is used,
- no TCR objective is active in canonical training.

---

# 2. Planned TCR Input Sequence

```
[TCR_CLS] [TRA] alpha_1 alpha_2 ... [SEP] [TRB] beta_1 beta_2 ... [SEP]
```

When only one chain is available (planned):
```
# beta-only
[TCR_CLS] <MISSING> [SEP] [TRB] beta_1 ... [SEP]

# alpha-only
[TCR_CLS] [TRA] alpha_1 ... [SEP] <MISSING> [SEP]
```

The TCR token stream is separate from the pMHC token stream (`design.md` S3.1).

---

# 3. Planned Per-Token Embedding

```python
tcr_token_repr[i] = aa_embed[i] + tcr_segment_embed[i] + tcr_position_embed[i]
```

`aa_embed` is shared with the pMHC encoder.

## 3.1 TCR Segment Embedding

| Segment | Tokens |
|---------|--------|
| `SEG_TCR_ALPHA` | `[TRA]`, alpha residues |
| `SEG_TCR_BETA` | `[TRB]`, beta residues |
| `SEG_TCR_GLOBAL` | `[TCR_CLS]`, `[SEP]` |

## 3.2 CDR-Annotated Positional Encoding (Planned)

CDR boundaries from IMGT numbering (ANARCI or explicit annotation), with
region embeddings and CDR3 dual-terminus encoding.

---

# 4. Planned TCR Encoder Architecture

Small transformer over TCR tokens only:
- layers: `N_tcr` ~ 3-4,
- dimension: same `d_model` as pMHC path,
- attention: full self-attention within TCR tokens,
- style: pre-norm transformer.

---

# 5. Planned Integration into Recognition Latents

## 5.1 Gated Residual Cross-Attention (Planned)

Recognition starts from peptide-only latent computation; a gated residual TCR
contribution is optionally added when TCR is available.

## 5.2 Gating Rationale

TCR presence changes semantics from population-level recognizability to
TCR-specific matching. The planned gate handles this transition.

---

# 6. Planned TCR-pMHC Match Score

Canonical planned anchor is `pmhc_vec` (from `design.md` S9.7):

```python
pmhc_embed = Linear_proj_pmhc(pmhc_vec)   # (d_model,)
tcr_embed = Linear_proj_tcr(tcr_vec)      # (d_model,)
match_logit = cosine_similarity(pmhc_embed, tcr_embed) / temperature
match_prob = sigmoid(match_logit)
```

Supervision source (future): known TCR:pMHC pairs from VDJdb/McPAS.

---

# 7. Planned Contrastive Objective

Optional InfoNCE objective in shared TCR-pMHC embedding space:

```python
pmhc_embeds = Linear_proj(pmhc_vec)
tcr_embeds = Linear_proj(tcr_vec)
L_contrastive = infonce(pmhc_embeds, tcr_embeds, temperature=0.07)
```

Not active in canonical training.

---

# 8. Planned Missing-Chain Handling

| Situation | Planned strategy |
|-----------|------------------|
| No TCR | Skip TCR pathway; recognition remains peptide-only |
| Single-chain TCR | Use `<MISSING>` in absent chain segment |
| Paired TCR | Full TCR encoding |

---

# 9. Planned Multi-TCR Bag Support

Future MIL semantics mirror multi-allele Noisy-OR:

```python
per_tcr_match_probs = [sigmoid(match_logit_i) for tcr_i in tcr_bag]
bag_match_prob = 1 - prod(1 - p for p in per_tcr_match_probs)
```

---

# 10. Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| TCR pathway in canonical model/training | Not active | Reserved future feature |
| Standalone TCR utility modules | Prototype-only | Kept for future integration work |
| TCR-conditioned recognition/matching outputs | Not active | No canonical guarantees yet |
| Contrastive TCR-pMHC objective | Not active | Planned |
| Multi-TCR bag training/inference | Not active | Planned |
