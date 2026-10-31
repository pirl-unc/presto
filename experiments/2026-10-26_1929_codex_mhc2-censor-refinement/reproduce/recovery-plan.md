# Explicit recovery after the available-memory guard stopped the censor fit

2026-10-26, Codex / GPT-6. The original controller is terminal, exit1; its state
and failed fit receipt remain unchanged. The censor action stopped after126.502s
on the10GiB available-RAM floor, with2,776,088,576bytes peakRSS. The exact arm
completed and its fixed-final predictions are already saved. Older jobs remain
live and are not restarted or modified.

The durable censor snapshot is in epoch1 at update100, position6,400, with
6,400IC50 exposures and32optimizer state entries. Preserve an exact copy and
hash of that snapshot before any recovery. Current available RAM has recovered
to about20GiB; do not lower the original13GiB startup or10GiB runtime floors.

Use canonical `fit_index(..., resume=True)` with the original index, model,
training recipe, parameters, scaler, optimizer ordering and RNG state. Its
execution-contract and initial-artifact checks must pass. The final trajectory
still ends at update1,476; do not add epochs or retune. Up to49uncommitted updates
may be repeated after rollback to the last50-step snapshot. Report that recovery
overhead and all attempt runtimes; equal logical update budgets do not imply
equal executed compute after this interruption.

Run a new, separately frozen controller: resume censor, score its fixed final
checkpoint, then run the original independent comparison verifier. Do not rerun
the completed exact condition or overwrite the original controller/failure state.
Stop on any new error; no automatic retries. Keep other source scans terminal
before relaunching, and defer memory-intensive work during this recovery.

The canonical resume/fixed-update software checks already passed in the46-test
suite. No original runtime, launcher, comparator, metric gate, source file or
holdout policy changes. Archive recovery code/invocation and terminal receipts.
