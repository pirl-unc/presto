# Match-scoped parent sequence correction

Initial cohort.py failed before any cohort quantities, groups or estimates were
written because it required a sequence for every reference inventory entry.
The canonical source intentionally supplies sequences for all10,497matched parent
accessions and leaves15,261unmatched inventory entries explicitly null. All52,277
occurrences reference parents with sequences; no matched parent is missing one.
This is a consumer assumption error, not an upstream omission affecting the cohort.

Preserve cohort.py, its snapshot and failure log. cohort_v2.py accepts only explicitly
unmatched null inventory entries, ignores them when constructing sequence lookups,
and still requires every occurrence to resolve to a nonnull correct full sequence.
The empty model-inputs directory may be reused; no output file is overwritten.
All source measurements, mappings, cohort gates, splits, precursor aggregation and
model hypotheses remain unchanged. Record the scope check and continuation hashes.
