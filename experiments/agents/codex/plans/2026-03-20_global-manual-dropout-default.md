# Global Manual-Dropout Default

## Goal

Make seeded manual dropout the default on CPU, CUDA, and MPS for the focused PF07 path, so dropout implementation no longer varies silently by backend.

## Why

- The user explicitly wants to avoid subtle hardware-dependent bugs caused by different default dropout implementations.
- The previous split default was:
  - CPU/CUDA: native `nn.Dropout`
  - MPS `auto`: manual dropout
- That made the runtime contract hardware-dependent even when model/data settings were the same.

## Required Changes

- Redefine `--mps-safe-mode auto` to use seeded `manual_dropout` on all backends.
- Keep:
  - `off` for native backend dropout
  - `zero_dropout` as explicit MPS fallback/debug mode
- Update tests and docs so the default contract is described honestly.

## Confirmation

- Run a small registered CPU-vs-MPS confirmation on the default path:
  - honest all-class-I PF07 downstream condition
  - reduced `max_records`
  - default `auto` mode, no explicit manual override
- Success criterion:
  - both runs complete
  - both report `mps_safe_mode_applied = manual_dropout`
  - the experiment README and canonical log state that manual dropout is now the default, not a special MPS-only path
