# Source-layout correction before retry

The original launch failed before publishing a source package. Both original
summary.txt tables have three rows: deposited RAW, experiment subtotal and Total.
The fixture covered only the raw row. Select the run by the exact README RAW link;
retain all original rows and explicitly validate subtotal identities and blank
Experiment fields. Reject extra runs, duplicate subtotals or mismatched RAW identity.

No quantity, membership, quality threshold, candidate or target changed. Preserve
attempt-1 source snapshots, receipt/diff and launch log; its empty output parent is
retained under data/measured_ms1_source_admission/2026-10-24_1635_attempt1.
The regression test adds authentic subtotal structure and rejects a second run.
