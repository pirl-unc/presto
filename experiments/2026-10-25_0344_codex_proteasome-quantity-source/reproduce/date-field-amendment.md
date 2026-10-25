# Source date fields: preserve corruption, verify stored representation

The first complete-cell audit found no literal value disagreements, but stopped
because openpyxl rendered13 numeric cells as dates. Every one is in the original
`Gene names` column H of the Dataset7 peptide sheet. The raw OOXML values/styles,
rendered dates and coordinates are preserved in the failed inventory. This is a
source semantic defect, not a reason to rewrite any gene name from a date.

Verify the workbook epoch and explicit cell style, then independently reconstruct
each date from the stored Excel serial. Require all13 and no additional cells;
the original literal numeric values remain unchanged. Combine this receipt with
the original all-other-cells equality result. Do not relabel the original audit
as passed or replace its files. Report fidelity as qualified with date-shaped
gene annotations; these are unusable gene labels until independently resolved.
Peptide sequence and all protein-accession alternatives remain source evidence.
No model, data selection, assay number or quantity transformation changes.
