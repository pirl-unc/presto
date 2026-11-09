# Source-supported semantic correction

The guarded diagnosis finds S3A A16:B18 reports GLFDQHFRL[+7] only at
1, 10 and 100 fmol. The next peptide begins at A19. There is no source row
for 0.1 fmol for this peptide. Every other S3A block has the four anticipated
doses; S3B has five doses per block. Preserve the missing dose as an absent
record, never create a source coordinate or impute zero. The original support
criterion remains at least 12 sequences with at least three positive doses.
The complete four-dose assertion was a parser error, not a prospective gate.

Freeze a new semantic script, leaving analyze.py and its failed receipt intact.
Allow exactly this recorded omission and retain all other structural assertions.
Verify every emitted response/dose cell independently against raw OOXML, including
the blank response cells. Keep the 3 response columns as ordinal worksheet slots
with unknown acquisition joins. Do not call them known injection replicates.

Also audit S6B parenthesized isotope sites and reported mass shifts against S6A.
S6A A10:C10 reports V / 13C3,15N1 / 4.007, while S3A and S6B use nominal +6.
Do not choose one chemical identity or repair source values. Record every affected
isotopologue and original coordinates. Training remains unadmitted until shared
Hitlist curation defines a supported identity/metadata contract. This is source
qualification, with no fits, model scoring or held-out biological access.
