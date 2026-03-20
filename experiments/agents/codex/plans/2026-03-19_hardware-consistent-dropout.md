# Hardware-Consistent Dropout

## Goal

Replace the temporary MPS `zero_dropout` safety path with real dropout that stays numerically stable on Apple Silicon and keeps the same dropout contract across CPU and MPS.

## Starting Point

- Current `--mps-safe-mode auto` in `scripts/focused_binding_probe.py` zeroes both `nn.Dropout.p` and `nn.MultiheadAttention.dropout` on `mps`.
- That made MPS stable, but it changed the training contract relative to CPU/CUDA.
- New targeted debugging on the same first train batch shows:
  - baseline MPS train-mode forward is non-finite
  - zeroing only `nn.Dropout` fixes the first-batch NaN
  - zeroing only `nn.MultiheadAttention.dropout` does not
  - replacing `nn.Dropout` with a manual Bernoulli-mask implementation also fixes the first-batch NaN with finite gradients

## Hypothesis

The broken backend op is ordinary `nn.Dropout` on this path, not attention-weight dropout. So we can preserve the intended dropout rate by swapping `nn.Dropout` modules for a backend-neutral manual implementation while leaving `nn.MultiheadAttention.dropout` unchanged.

## Planned Changes

1. Add a small backend-neutral dropout module:
   - manual Bernoulli mask
   - same `p` semantics as PyTorch dropout
   - no hidden device-specific branches inside the math itself
2. Add a recursive helper that replaces `nn.Dropout` modules in the focused model with that manual implementation.
3. Update `--mps-safe-mode` behavior:
   - `auto` on `mps` -> `manual_dropout`
   - keep `zero_dropout` as an explicit fallback option
   - leave CPU/CUDA unchanged unless explicitly asked
4. Preserve runtime summary metadata so experiment artifacts record which safeguard was actually used.

## Verification

- Unit tests:
  - manual-dropout replacement preserves `p`
  - `auto` on `mps` now applies `manual_dropout`
  - `off` still leaves the model untouched
- Local experiments:
  - registered matched CPU-vs-MPS smoke with manual dropout
  - registered matched CPU-vs-MPS short multi-epoch mini-train with manual dropout
- Compare:
  - completion vs divergence
  - CPU vs MPS terminal metrics
  - absence of `non_finite_train_loss`

## Exit Criteria

- Manual dropout is stable on MPS for the matched validation contracts.
- `auto` no longer needs to disable dropout entirely.
- The repo docs and experiment log say clearly that the current local MPS path uses backend-neutral manual dropout rather than native `nn.Dropout`.
