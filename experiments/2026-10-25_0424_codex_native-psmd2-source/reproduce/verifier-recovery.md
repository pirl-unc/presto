# Independent verifier recovery

The first verifier reached the Excel annotation pass after regrouping quantities,
then failed because openpyxl emits EmptyCell objects without coordinates for
missing cells. Its original code and log remain preserved. No source or result
was changed and no model ran. Skip cells without coordinates, then rerun this
verifier to a separate log. Retain all numerical comparisons and source contracts.
