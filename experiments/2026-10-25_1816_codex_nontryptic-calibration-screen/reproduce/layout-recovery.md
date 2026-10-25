# Dose-layout diagnosis after the preserved failed analysis

The first semantic analysis terminated at its complete-dose-grid assertion on
2026-10-25. Its source, isotope inventory, log, and exit receipt remain unchanged.
Before changing interpretation, inspect every S3A/B block's original A/B cells,
row boundaries and blank response slots under the same resource guard. Inspect
S6 identity/acquisition metadata as already authorized. Write a diagnosis with
coordinates, without fitting or changing the original support requirement.

Use Python's existing read-only OOXML/openpyxl route because literal stored-cell
and absent-cell distinctions are required and were independently verified before
this diagnosis. Do not rewrite/export the source workbooks, add data, impute a
missing dose, infer injection identity from ordinal columns, or access new sources.
Register any supported layout correction separately before running it.
