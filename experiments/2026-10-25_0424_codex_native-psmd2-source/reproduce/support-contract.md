# Descriptive support before any modeling

Tabulate support at one, two and three positive acquisitions per condition; these
are descriptive strata, not a chosen fitting threshold. Do not average quantities
or compute control/KD effects. A strict-origin stratum has no source reverse or
contaminant flag, PEP <= 0.01, one distinct complete parent sequence, only reported
human-endogenous reference origins and no I/L-only candidate spans. This stringent
stratum does not imply the remaining observed peptides are biological negatives.

Build dependence components on all 10,647 source peptides before applying those
strata. Join every candidate reference sequence and every source-declared parent;
reference accessions with identical full sequences share a node. Retain unresolved
declared parents as nodes. Report components, qualifying parent groups with at
least two peptides, and reference-contrast capacity (n_peptides - 1 per group).
No split, model, reference-peptide selection or training recipe is fixed here.
