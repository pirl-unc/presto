# Quantitative anchors while canonical validation waits

Purpose: identify a concrete, independently quantified source that could constrain
peptide-specific MS response or precursor-relative peptide amounts beyond the nine
selected PXD024809 standards. This supports the original availability objective;
it does not replace native/APC validation with synthetic dose-response success.

This is a read-only literature/source screen, not a numerical experiment. No raw
MS downloads, source parsing, new model fit, held-out access or resource-limit
change. Current source family0959 and its frozen execution contract remain intact.

## Fixed screen

- Search existing Hitlist issues first to avoid duplicating previous data asks.
- Read primary papers/repositories for purified synthetic standards with assigned
  amounts and uncertainty, direct native peptide amounts, or quantitative MHC
  measurements with independently reported spike/recovery/precursor context.
- Distinguish instrument response from digestion/extraction, MHC capture and
  biological processing. Classify every measurement by actual observed grain.
- Record public accession, processed-table availability, standard chemistry,
  amount assignment, spike stage, matrix, acquisition, dilution design, physical
  replication and any recovery/precursor pairing. Do not infer unreported fields.
- Prefer broad sequence support and protocol-matched transfer. A small targeted
  panel or MHC-selected abundance can supply a restricted anchor, not free APC
  concentrations or an unconditional negative/candidate universe.
- Compare each candidate with the already qualified PXD021013 crude-pool and
  PXD004886 selected-standard limitations. Keep independent calibration and
  biological-transfer requirements explicit.

## Action and closure

If a source meets a materially missing requirement, add a concrete, cited Hitlist
issue or extend its existing owner; describe exact data/metadata and acceptance
criteria. If no source qualifies, record the limiting evidence without claiming
that all published data have been exhausted. Any later source retrieval or
numerical audit requires its own registered family and unchanged resource gates.

## Checklist

- [x] Search existing issues and primary literature.
- [x] Verify candidate roles and record processed-data access status.
- [x] Record an evidence-based admission direction and unresolved access limits.

## Review — 2026-10-25

Extended the existing [Hitlist calibration request](https://github.com/pirl-unc/hitlist/issues/762#issuecomment-6396605227),
instead of creating a duplicate issue. The accession search found no separate
owner. The requested qualification prioritizes PASS01163, with PASS01200 context
and PASS05890 as a potential comparator. Primary publication evidence is linked
in the comment. No processed dataset, source schema or trainable coverage is yet
verified: browser retrieval of PASSEL failed and a specific public Panorama
folder was not established. This is an access-route limitation, not a declaration
that the data are unavailable.

The admission requirement is sharper: obtain separate absolute heavy/light peak
areas and source-backed amount/acquisition joins. Under a shared isotope-pair
response coefficient, a ratio cancels that coefficient. Ratio-only curves cannot
identify peptide-specific absolute response. Selected MRM calibration also does
not establish transfer to other acquisition methods or native APC quantities.
The follow-up requires complete planned observations, missingness/censoring,
standard characterization, material/protocol lineage and independent numerical
reconciliation before any fit. No new source download, model or biological result
is claimed. The read-only screen is closed; do not keep broadening this search
while the already registered canonical replay is pending.

Current exact heads now pass hosted CI: Presto d92606a / 40184204117 and Hitlist
1b3ad56 / 40183613623, with Hitlist docs build 40183613700. Both PR descriptions
were updated and exactly read back. Hosted checks do not replace local source
execution. At 10:40 UTC the recorded free-plus-speculative memory was 82,100,224
bytes, below both the 2.5 GiB local-suite and 3 GiB source startup guards. No local
stage was launched. Disk reserve passes; the primary checkout's working and
staged fingerprints remain unchanged.

Receipts: `tasks/modular_model/quantitative-anchor-hitlist-readback.json`,
`dilution-canonical-ci-completed.json`, `dilution-canonical-ci-publication.json`
and `dilution-canonical-resource-recheck.json` (all under `tasks/modular_model/`).
