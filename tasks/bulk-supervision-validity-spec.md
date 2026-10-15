# Remove unsupported bulk depth supervision — #70, first PR

The ordinary `load_bulk_peptides` path has no verified cross-protocol comparison
contract or first-seen depth. Remove its fraction-depth/replicate score and leave
`detectability_label=None` on every imported observation and generated control.
Retain observed-product excision proxies and separately identified generated
wrong-enzyme controls. Do not relabel every observed peptide as detectability 1.

Remove the unused numeric depth helper/constants. Preserve nullable replicate
metadata without coercing unknown counts to zero. Record the absence of
detectability support and its reason in loader statistics. A caller-supplied
detectability label on a manually constructed record has unknown provenance;
observation alone must not make it a measured or depth-derived label.

Update current source/learning documentation, the output contract and coverage
expectations so the default optional bulk path advertises only the objectives
it actually supervises. Preserve frozen experimental records; add a current
limitation statement instead of rewriting their historical results.

Prove the old score generated labels from missing/incomparable depth; test the
corrected raw loader, nullable metadata, dataset collation masks and complete
coverage preflight. Existing manually supplied labels remain accepted with
explicit unknown provenance. Run formatter, lint, the full suite and CI before
calling this stage validated. Bump the patch version.

The next PR under #70 consumes Hitlist's verified search-scoped candidate
bundle with source/reference/protocol/occurrence provenance and scoped observed
labels. #71 preserves training lineage; #48/#53 gate supported endpoints and
real-data quality. This first PR removes invalid supervision permanently; it
does not claim the replacement dataset or the full readiness tracker is done.
