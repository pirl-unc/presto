# Affinity Output Consistency Regularization (future, 2026-03-16)

## Goal

Test whether weak output-side consistency regularization can stabilize related affinity outputs while preserving the canonical sequence-only input contract.

## Modeling Constraints

- No assay-selector inputs.
- No assay type / assay method / assay prep / assay geometry / assay readout tensors may be used as predictive inputs.
- If assay structure is added, it must be output-side only and driven by shared sequence-derived latents.
- Supervised assay-family losses remain primary; any agreement loss is secondary and low-weight.

## Motivation

- The corrected PF07 rerun will expose whether `KD_nM`, `IC50_nM`, `EC50_nM`, `KD_proxy_ic50_nM`, and `KD_proxy_ec50_nM` disagree on supported peptide / allele pairs.
- The user wants two kinds of soft coupling:
  - `IC50` outputs from different assay contexts such as cellular / purified / lysate crossed with radioactivity / fluorescence should stay similar
  - proxy and non-proxy affinity outputs should also stay similar

## Proposed Structure

### 1. IC50 context family

- Introduce a shared context-agnostic `IC50_anchor_nM` output.
- Add optional output-side IC50 context residual heads for populated assay buckets.
- Recommended factorization:
  - prep bucket: `cellular`, `purified`, `lysate`, `other`
  - readout bucket: `radioactivity`, `fluorescence`, `other`
- Recommended output form:
  - `IC50_ctx(prep, readout) = IC50_anchor + small_residual(prep, readout)`
- Regularizer:
  - weak Huber / smooth-L1 agreement in `log10(nM)` between each context-specific IC50 output and `IC50_anchor`
  - optionally a low-weight variance penalty across all context-specific IC50 outputs for the same sample

### 2. Direct vs proxy KD family

- Keep or add a shared `KD_anchor_nM`.
- Tie:
  - `KD_nM`
  - `KD_proxy_ic50_nM`
  - `KD_proxy_ec50_nM`
- Preferred regularization:
  - weak Huber / smooth-L1 in `log10(nM)` from each output to `KD_anchor_nM`

### 3. Cross-family proxy ties

- Optional and weaker than the within-family ties:
  - `IC50_nM <-> KD_proxy_ic50_nM`
  - `EC50_nM <-> KD_proxy_ec50_nM`
- These should be treated as soft compatibility losses, not equality constraints.

## Weighting Rules

- Start small.
- Agreement weights should be materially lower than the supervised assay losses.
- Initial sweep should stay in a low range such as:
  - `0.0`
  - `0.005`
  - `0.01`
  - `0.02`
- If the agreement loss starts pulling supported heads into visibly wrong values, reject it immediately.

## Preferred Loss Form

- Use `log10(nM)` space.
- Prefer anchor-based penalties over full pairwise all-to-all penalties.
- Prefer Huber / smooth-L1 over plain L2.
- Keep all consistency losses off by default and explicit in configs / launchers.

## Gating Before Launch

- Wait for the corrected PF07 sequence-only all-head rerun to finish.
- First confirm where disagreement exists on supported peptides and held-out rows.
- Do not use unsupported probes such as `SLLQHLIGL` as primary evidence for the regularizer.

## Acceptance Criteria

- Held-out primary metrics improve or at least do not regress materially.
- Supported probe peptides remain biologically plausible.
- The regularizer reduces pathological head disagreement without erasing genuine assay-family differences.
- The implementation remains compatible with the outputs-only assay modeling contract.
