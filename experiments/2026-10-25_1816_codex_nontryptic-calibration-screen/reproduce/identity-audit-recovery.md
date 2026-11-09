# Missing isotope assignment is a source limitation

Reconcile attempt 1 wrote the source-dose summaries, then its independent audit
failed because S6B includes labeled methionine while S6A omits methionine from its
isotope-assignment table. Preserve that failure and the partial output. A separate
audit must represent missing S6A assignments as unknown, not as zero shift or a
guessed isotope. Complete the original-cell verification and audit all 72 reported
isotopologues. This changes the diagnostic's missing-metadata handling only.
No isotope identity is selected, no training is admitted, and no support threshold
or model objective changes.
