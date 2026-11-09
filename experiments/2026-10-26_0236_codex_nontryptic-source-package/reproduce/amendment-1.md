# Amendment 1: diagnose panel-length rejection

Previous goal turn: progress (shared compiler, full software regression, and an
original-source run produced a concrete failure). The intervening checkpoint
retention fix is complete locally and separate from this experiment.

Attempt verify-1 is terminal with ValueError: Panel sequence length differs,
2.427 seconds, 274,776,064 bytes peak RSS and no resource violation. Preserve
its source snapshot, request, logs and failed package staging directory.

Before changing semantics, run an inspect-only stage under unchanged guards:
independently read every S1 sequence and length using the literal source reader
and openpyxl, retaining original type/formula/value and calculated string length.
Write exact contradictory coordinates. No fit, target selection, new retrieval,
source mutation or hidden correction. Diagnose other original numeric cell types
only if the recorded error requires it. Then specify a source-faithful correction
and regression fixture before a fresh frozen compiler/request/package attempt.
