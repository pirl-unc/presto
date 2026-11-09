# Comparator correction, after the preserved first execution

The first replay completed all checks and failed at final closure with 1,242
`context_literal_experiment` disagreements. All are the same existing projection
convention: context strings trim surrounding whitespace, while raw source-row
arrays retain the original cells. The v1 package already used that convention.
No source value, production code, identity rule or eligibility rule changes.

Preserve the first launcher, execution receipt, package, replay.json,
disagreements.json and failure log. A separate closure verifies every failure
against both package versions and the frozen independent raw rows, requires
exactly whitespace-only differences, and verifies every new context projection.
All other original checks must have passed. This corrects the audit comparison;
it does not silently mark the original execution successful. Bound closure to
30 seconds / 1 GiB, no rebuilding, training, or source repair.
