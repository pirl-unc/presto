# Why Affinity Isn't Learning Quickly — Root Cause Analysis

## Executive Summary

There are **7 distinct problems** preventing fast affinity learning, ranked by severity.
The top 3 are architectural/training design issues that compound multiplicatively.
Together they explain why the SLLQHLIGL probe shows near-zero allele discrimination
even after the refactored stack is numerically healthy and loss is decreasing.

---

## Problem 1: KD gradient path is extremely indirect (CRITICAL)

The main KD prediction flows through:

```
interaction_vec (512-dim)
  → BindingModule: 3 × Linear(512, 1)     # 3 scalars: log_koff, log_kon_intrinsic, log_kon_chaperone
  → derive_kd(): log_koff - log10(10^kon_i + 10^kon_c) + 9
  → binding_logit_from_kd_log10(): sigmoid-style calibration
  → kd_from_binding (inverse of above)
  → + kd_bias (from binding_affinity_vec via tanh * softplus)
  → smooth_upper_bound → final KD_nM
```

**The problem**: The interaction_vec (512-dim, rich with allele-specific information from 8 cross-attention queries over MHC tokens) is immediately compressed to 3 scalars by BindingModule. That's 512 → 3 — a **170× compression** at the first hop. The 3 scalars then go through transcendental functions (`pow(10, ...)`, `log10(...)`) that further distort gradients.

Meanwhile, `binding_affinity_vec` (derived from interaction_vec via a separate 512→256→256 MLP) is only used for:
- `kd_bias`: a **residual correction** gated by `tanh * softplus(0.5)` ≈ ±0.47 log10 units max
- `binding_affinity_probe_kd`: an auxiliary shortcut loss at weight 0.3

So the **main gradient pathway** (KD loss → interaction_vec) must flow backward through the kinetic derivation (nonlinear) and the 3-scalar bottleneck (extreme compression). The `binding_affinity_vec` path, which could provide a richer gradient, only contributes a small bias term.

**Impact**: Gradients arriving at the interaction_vec are weak, noisy, and allele-insensitive because the 3-scalar bottleneck destroys the allele-specific pattern before loss evaluation.

---

## Problem 2: Loss weight dilution (HIGH)

Default aggregation is `sample_weighted` (line 86 of train_synthetic.py). The formula:

```python
task_weight = base_weight * supervised_loss_support[task_name]
# where support = number of labeled samples in batch for that task
```

Then all terms are summed and divided by `total_weight`.

In a typical full-data batch (batch_size=512, balanced sampler):
- **Binding tasks** (binding + binding_kd + binding_ic50 + binding_ec50 + binding_affinity_probe): ~5-15 samples with labels → support ≈ 5-15
- **Elution task**: ~100+ samples with labels → support ≈ 100+
- **T-cell tasks** (tcell + immunogenicity + 5 context tasks): ~50+ samples → support ≈ 50+

With uncertainty weighting on top, the effective binding weight can be 10-50× smaller than elution. The optimizer predominantly serves elution/presentation gradients.

**Impact**: Even if binding gradients are clean, they're swamped by other tasks in the parameter update.

---

## Problem 3: No allele-discrimination loss (HIGH)

The contrastive MIL loss (E1 in the learning refactor) creates pressure to distinguish alleles within elution bags — but this is a **presentation-level** signal, not a **binding-level** signal. There is no loss term that directly says "this peptide binds A*02:01 better than A*24:02."

The binding loss only says "this peptide-allele pair has KD ≈ X nM." For the model to learn allele discrimination from this, it needs many (peptide, allele_A, KD_A) vs (peptide, allele_B, KD_B) contrastive pairs. But:
- Binding data is mostly single-allele measurements (not paired)
- The model sees each (peptide, allele) independently and must learn relative preference from absolute KD values
- Absolute KD prediction is hard when the main gradient path is a 512→3 scalar bottleneck

**Impact**: The model can minimize binding loss by learning the **average KD distribution** without learning allele-specific preferences.

---

## Problem 4: BindingModule hard clamps kill gradients (MEDIUM-HIGH)

```python
log_koff = torch.clamp(self.head_log_koff(pmhc_vec), min=-8.0, max=8.0)
```

Three independent hard clamps at [-8, 8]. If any head output reaches ±8.0, its gradient is exactly zero. Early in training with random initialization, head outputs can easily reach these bounds. The `derive_kd` function then also clamps `log_kd_nM` to [-3, 8].

**Chain of clamps**: interaction_vec → Linear → clamp[-8,8] → pow(10,...) → clamp[1e-10,1e10] → log10 → clamp[-3,8] → calibration → clamp[min=-3] → smooth_upper_bound

Each clamp is a potential gradient-zero wall. If koff and kon outputs are both near 8.0 (or -8.0), the model has zero gradient to escape.

**Impact**: Can trap the model in a flat region for many steps, especially early in training.

---

## Problem 5: 41% data loss from MHC filtering (MEDIUM)

From Codex's Modal run: "resolved coverage before filter: 58.64%". The `--filter-unresolved-mhc` flag (default True) drops all rows where the MHC allele can't be looked up in the index.

For binding specifically, the loss is probably less severe (binding data uses well-characterized alleles like HLA-A*02:01), but elution data suffers badly: 1.7M rows dropped. This shifts the batch composition even further away from binding.

**Impact**: Shrinks the total training set and skews task proportions. The binding fraction of the surviving data is probably similar, but the overall signal diversity is reduced.

---

## Problem 6: Groove fallback truncation degrades MHC signal (MEDIUM)

When groove extraction fails, `prepare_mhc_input` with `allow_fallback_truncation=True` doesn't drop the sample — it crudely truncates:
- Class I: first 45 + next 25 = 70 residues
- Class II: alpha first 32, beta first 34

This truncation can cut off critical groove residues, turning the MHC input into near-random sequence. The model still sees these samples and trains on them, but the MHC signal is degraded.

**Impact**: Noisy MHC signal dilutes the allele-specific learning that does occur.

---

## Problem 7: Synthetic negative expansion dilutes batches (LOW-MEDIUM)

Synthetic elution negatives (1.07M added) expand the dataset substantially. The balanced sampler tries to correct, but the raw count imbalance remains:
- ~53K binding samples total (real + synthetic)
- ~3.4M elution instances (real + synthetic + cascaded)
- 64:1 ratio before MIL expansion

The MIL bag expansion makes this worse. Each elution sample with 6 alleles becomes 6 instances, all scoring the same peptide against different alleles. The Noisy-OR loss only requires "at least one is positive" — no incentive to pick the right allele.

**Impact**: Batches are dominated by elution instances. Binding gets a small fraction of each batch even with balanced sampling.

---

## Architectural Corrections (Not Bugs, But Subagent Errors)

The subagent reports contained several factual errors I need to correct:

1. **"Single query binding bottleneck"** — WRONG. `binding_n_queries=8` (default). The interaction_vec is 8×64=512 dims, not 256. This is reasonable capacity.

2. **"pmhc_interaction depends only on processing"** — PARTIALLY WRONG. The `_binding_latent_query` method **does see MHC tokens directly** via cross-attention. The processing dependency in `LATENT_DEPS` means the processing latent is injected as an extra KV token, but the main KV stream includes raw MHC-a and MHC-b tokens. The binding latent has full access to MHC sequence.

3. **"Presentation latent has no token access"** — CORRECT but misleading. Presentation gets `interaction_vec` (512-dim) + `groove_vec` (256-dim) + `processing_vec` (256-dim). It doesn't cross-attend to tokens, but it gets the full interaction vector before scalar compression.

4. **"interaction_vec compression 256→64 (4×)"** — WRONG dimensions. It's `d_model(256) → pmhc_interaction_token_dim(64)` per query, but there are 8 queries, so the total is 512-dim. The compression is 256→64 per query token, then 8 tokens are concatenated to 512.

---

## Recommended Fixes (Priority Order)

### Fix 1: Add direct KD regression from interaction_vec (bypasses scalar bottleneck)

The `binding_affinity_probe` already does this at weight 0.3, but it reads from `binding_affinity_vec` (a 256-dim projection of interaction_vec). **Increase its weight to 1.0** or add a second direct regression head that reads the full 512-dim interaction_vec. This gives the interaction_vec a clean, direct gradient signal for KD prediction without going through the kinetic derivation.

```python
# Already exists at line 1841:
probe_kd = self.binding_affinity_probe(binding_affinity_vec)
# Current weight: 0.3. Increase to 1.0.
```

Alternatively, add a new head:
```python
self.direct_kd_head = nn.Sequential(
    nn.Linear(self.pmhc_interaction_vec_dim, d_model),
    nn.GELU(),
    nn.Linear(d_model, 1),
)
```

### Fix 2: Switch loss aggregation to `task_mean`

Change default from `sample_weighted` to `task_mean`. This gives each task equal weight regardless of sample count. Binding gets 1/N_tasks weight instead of proportional-to-count weight.

```python
# scripts/train_synthetic.py line 86:
"supervised_loss_aggregation": "task_mean",  # was "sample_weighted"
```

### Fix 3: Replace hard clamps with soft clamps in BindingModule

```python
# Instead of:
log_koff = torch.clamp(self.head_log_koff(pmhc_vec), min=-8.0, max=8.0)

# Use:
log_koff = smooth_clamp(self.head_log_koff(pmhc_vec), min=-8.0, max=8.0)
# where smooth_clamp = softplus-based: preserves small gradient near bounds
```

The `smooth_upper_bound` and `smooth_lower_bound` functions already exist in the codebase (used for KD capping). Apply them here too.

### Fix 4: Add binding-specific contrastive loss

For each binding sample in a batch, find another sample with the same peptide but different allele (if one exists). Create a margin loss:
```
if KD_true(allele_A) < KD_true(allele_B):
    loss += max(0, KD_pred(allele_A) - KD_pred(allele_B) + margin)
```

This directly trains allele discrimination without relying on absolute KD accuracy. Even a small number of contrastive pairs per batch would help.

### Fix 5: Log and monitor binding-specific metrics

Add per-batch logging of:
- Number of binding-labeled samples in each batch
- Binding loss magnitude relative to total loss
- Uncertainty weight for binding task
- Mean/std of BindingModule head outputs (to detect clamp saturation)

This is diagnostic, not a fix, but it's necessary to know whether the other fixes are working.

---

## What NOT to Change

- **BindingModule kinetic architecture**: The koff/kon/KD physics is biologically sound and should stay. The issue is not the kinetic model itself but the gradient path through it.
- **Core window enumeration**: Works correctly. 8 queries provide enough capacity.
- **Interaction_vec dimension (512)**: This is fine. The bottleneck is at the 512→3 scalar step, not at the interaction_vec itself.
- **Groove extraction**: The algorithm is correct. The fallback truncation is a reasonable safety net.
