# Version internal excision-site scoring — Presto #88

## Problem and scope

The existing internal missed-cleavage score counts K/R before proline for ordinary
Trypsin although its junction rule and the source search definition exclude that
boundary. Trypsin/P permits it (Cox Lab primary Andromeda documentation:
https://cox-labs.github.io/coxdocs/andromeda_enzymes.html). The previous PR separates
those machinery categories but deliberately preserves old checkpoint predictions.
This follow-up corrects internal site counting under an explicit scoring version.
It does not introduce measured labels, change mixture components or claim that
rule penalties estimate true cleavage probabilities.

## Design

New models default to `excision_scoring_version=2`; the head counts a site only
when the current and immediately following peptide tokens are present and the
next residue does not block that machinery's rule. The final residue and padding
cannot become internal sites. Trypsin/P has no proline exclusion; ordinary
Trypsin, chymotrypsin and GluC use their already declared exclusions. LysC is
unchanged. The version-1 function remains exactly the historical computation.

Checkpoint configs record the scoring version independently of machinery
vocabulary. Any historical checkpoint without this field loads version 1,
including eight-category checkpoints produced by 0.1.6. Never silently upgrade
scoring at load or resume. Unknown versions fail; trainer resume validates the
contract before mutating weights or optimizer state. Bare state dicts do not
encode scoring semantics and therefore receive the historical version unless
saved in a self-describing checkpoint after intentional new-model construction.

## Verification and publication

Test ordinary Trypsin versus Trypsin/P at internal K-P/R-P, allowed boundaries,
protein termini, padding, other blocked proteases and both scoring versions.
Retain exact historical checkpoint weight/output parity; demonstrate corrected
new checkpoints round-trip and incompatible trainer resumes fail before mutation.
Run format/lint/full tests, strict docs and CI on frozen code. Version 0.1.7;
separate reviewed PR, merge, clean-main distribution and public hash verification.
Canonical #70 fitting follows this correction and records both contract versions.
