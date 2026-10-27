# Shared candidate transfer and flank benchmark

Status: preparing; no predictive fits or profiles yet. Codex / GPT-6.
The [prospective plan](reproduce/plan.md) defines the v4 architectural contract,
all ten conditions, original IC50 data/qualifier/split policy, training/loss settings,
metrics, unsupported-class interpretation, local resource limits and closure.

[Live local dashboard](http://127.0.0.1:54495/) is running in Firefox. It reads only
local generated state. Earlier converged results are separate reference values.

Software checks: 272 common-model/training tests and four family harness fixtures
pass. No new biological result follows from these checks. Old v2/v3 checkpoint
inference is preserved; current training uses shared-v4 only.

Admission attempt1 refused before processing because available RAM was below6GiB
while software tests ran. Failure receipt is retained in reproduce/receipts/.
After tests ended, measured available RAM recovered to9.13GiB; the manual second
admission attempt retains all original resource thresholds. No fit retry or policy
relaxation. Raw derived inputs/checkpoints will be in ../../artifacts/ under this
family ID; exact paths/hashes are written by admission and launch freezing.

Required selected validation/test metrics, prediction dumps, timings and plots are
pending. PR131's scientific qualification remains open.
