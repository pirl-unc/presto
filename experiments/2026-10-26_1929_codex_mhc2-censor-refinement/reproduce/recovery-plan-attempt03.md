# Restore the exact original environment before recovery

2026-10-26, Codex / GPT-6. Attempt02 terminated before training because the
canonical resume contract rejected an omitted environment override. The complete
recursive comparison in `recovery-attempt02-contract-diff.json` contains exactly
one difference: saved `PYTORCH_ENABLE_MPS_FALLBACK="0"` versus absent. Runtime,
data, model, optimizer recipe, scales, numerical backend and all other fields match.

Preserve attempt02 code, controller state, logs and receipts. Start a distinct
attempt03 controller with `PYTORCH_ENABLE_MPS_FALLBACK=0`, exactly as in the
original launch. Reuse the saved100-step snapshot only after verifying its hash
still equals interruption01. Freeze separate helper/controller copies; their
only changes are attempt identifiers and log/state destinations. Require full
execution-contract equality before launch. No gate is weakened, source modified,
checkpoint patched or model update added. Original resource floors still apply.
Stop on any new error. The original bounded rollback-overhead disclosure remains.
