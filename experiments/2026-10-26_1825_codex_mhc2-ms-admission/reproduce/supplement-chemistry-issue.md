Presto's MHC-II ligand admission found that supplementary imports cannot retain
explicit peptidoform metadata, even when the input CSV supplies it.

Verified against current Hitlist commit
`13b91424295e01286db46e8e9818fcc2fe816834` (2.0.42), not just an older export:

- [`scan_supplementary`](https://github.com/pirl-unc/hitlist/blob/13b91424295e01286db46e8e9818fcc2fe816834/hitlist/supplement.py)
  constructs its output record without `has_ptm` or `peptide_modifications`.
- An artificial two-row CSV with these columns supplied as False/unmodified and
  True/explicit modification returns both peptide rows but **neither column**.
  The check uses `classify_source=False`, explicit local entries and
  `allow_download=False`; no biological measurements were generated.
- The current [Stražar CSV](https://github.com/pirl-unc/hitlist/blob/13b91424295e01286db46e8e9818fcc2fe816834/hitlist/data/supplementary/strazar_2023_hla2.csv)
  contains only peptide, class and restriction. Its registry provides no
  source-backed peptidoform policy. This is a separate curation question from
  preserving fields that are already supplied.

In the published `human-cohort-corrected-v1` training table (Hitlist1.78.0;
Parquet SHA256 `f902341e90c1c57defc93f4bc540ffcbea1b41061afb4492ddd8aaa7642c65bd`),
all **293,282 class-II `curated_ligand_table` rows** have null `has_ptm`.
These are export counts, not a claim that the current source was regenerated.
Our consumer leaves that chemistry unknown and defers those rows under its
explicitly unmodified input contract. Alphabetic peptide strings alone do not
establish an unmodified measured peptidoform.

Requested acceptance:

- Preserve supplied modification text and parse the explicit PTM indicator
  losslessly through supplementary scanning, contributors, observation and
  training exports. Distinguish False, True and unknown; do not apply truthiness
  to string `"False"`.
- Retain unknown status when the source lacks the information. Any manifest-wide
  unmodified assertion needs a cited source/filtering contract; do not fill all
  blanks with False to increase training coverage.
- Audit original Stražar and MAPTAC sources for available modification/filtering
  evidence and retain the original row/sheet/field locator. State explicitly when
  the source cannot resolve the peptidoform.
- Add fixtures for supplied unmodified, modified and unknown states, and ensure
  distinct source peptidoforms are not silently merged under sequence-only IDs.

The independent fixture reports output_rows=2,
output_has_ptm_column=False and output_peptide_modifications_column=False.
The inspected module SHA256 is
`fb4bae5cb051f5f2817e890567da1a7736318bd6c07b5734f37290dc61e480a9`.
This issue concerns chemistry preservation; MAPTAC condition membership remains
separately tracked in #853.
