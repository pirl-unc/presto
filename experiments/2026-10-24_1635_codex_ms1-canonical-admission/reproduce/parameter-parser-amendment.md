# Parameter multiplicity correction before retry

Original parameters.txt repeats Decoy mode, Special AAs and Include contaminants
with identical values in both archives. Preserve every original row. Extract only
Version and Fixed modifications with explicit unique-value resolution; conflicting
required values fail. Other source parameter assertions remain lossless rows.

Both archives were inspected for duplicate peptide/form IDs, mismatched raw-file
claims, empty quantity/PEP cells and aggregate/channel discrepancies: none found.
No quantities or admission gates changed. Attempt-2 code/metadata and failure log
remain preserved, along with its empty output parent. The focused regression tests
exercise identical repeated values and conflicting required values.
