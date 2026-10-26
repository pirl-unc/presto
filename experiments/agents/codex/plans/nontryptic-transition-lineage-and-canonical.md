# PXD024917 transition lineage before shared calibration ingestion

Previous goal turn: progress. Family1816 is closed and published at717ed59, with
nominal dose/raw-response support but unresolved isotope identities and C/D/E run
joins. Hitlist849 specifies those gaps. This task retrieves the two already named
deposited exports and prepares a source-faithful shared implementation. It does not
restart broad source discovery or reuse the closed screen's exhausted budget.

The full goal stays precursor-relative APC peptide amounts, including short and
unobserved products, machinery-conditioned processing/clearance and biological
exact-versus-discrete comparisons. Protocol-specific MS calibration is a dependency,
not a substitute for that goal. PyPI2.0.41 publication is separately awaiting explicit
user approval; do not retry that rejected action or interpret goal continuation as
authorization. Scientific source work can continue independently.

## Fixed assets and bounded retrieval

Register a new source-only family before any requests. Use the public PRIDE
PXD024917 project and file-list APIs to locate exactly the S7-reported
`Iso18_DirectSpike_TransitionResults.xlsx` and
`Iso18_hipMHCspiikein_TransitionResults.xlsx`. Preserve literal spelling. If deposited
names differ, require source-backed inventory correspondence and report ambiguity.
No other studies, broad searches, raw spectra, credentials, authors, private routes,
source-executable downloads or shared-environment changes.

At most eight inclusive network operations, counting failed requests/retries and
any tools outside the fetch helper. At most4MiB per metadata response,16MiB per
processed asset,40MiB total. Connect10s/read30s/request120s; register each URL before
fetching. Only official ebi.ac.uk/ftp.pride.ebi.ac.uk HTTPS endpoints. Keep original
responses/receipts/checksums, including failures; do not overwrite/retry silently.
If pagination is necessary it consumes the same budget. Stop discovery at the
ceiling; do not claim global source absence from a narrow access failure.

Existing original S1/S3/S6/S7/article/appendix hashes from1816 remain input pins.
No duplication of those files; all assets remain once under repo-level data.
Before numeric export inspection, freeze file hashes and the exact inspection
questions in a separate request. Inspect schemas and original values without
training, fitting calibration, or opening any old biological held-outs.

## Scientific verification questions

1. Are original peptide/form, dose, acquisition and selected transition identities
   reported explicitly in the named exports? Preserve label positions, charge,
   isotope, product m/z, area units and original strings separately.
2. Does the export distinguish quantification versus trigger transitions, raw
   versus normalized signal, missing versus zero, and excluded/interfered ions?
   Source availability is not proof that the reported S3 sum can be reconstructed.
3. Can exact source identifiers and reported aggregation reproduce S3A/B cells?
   An intensity-nearest match, fitted scale, or row order is not an acquisition
   identity. If values coincide, record that numerical agreement separately from
   the evidence for acquisition linkage and ambiguity counts.
4. Do original annotated isotope forms resolve S6A's valine discrepancy, or provide
   competing source claims? Keep each assertion and its provenance. S6B describes
   labeled M even though S6A omits it; never call that globally absent.
5. Preserve original sample/preparation/matrix/spike-stage/amount assignment and
   analytical settings. Targeted MS2 signal is not native MS1 abundance. S1 affinity
   predictions are metadata, not binding supervision; S3C transition counts are
   not intensity. Technical ordinal columns are not known independent samples.

## Shared implementation after structural qualification

Use existing Hitlist package verification, immutable snapshots, Arrow schemas,
catalog fields/relations, material-context graph and EvidenceStore query surface.
Write a concrete implementation spec in an isolated feature checkout before code.
Retain original assets and addressed cell records, panel standards, reported forms
and conflicting assertions, nominal amounts, observation protocols, measurements,
acquisition/export inventories and transition/measurement lineage. Reuse field
names when meanings agree. Do not add a Presto-only ingestion path or overwrite
the existing study/sample curation. Preserve unknowns and queryable exclusions.

Freeze any consumer recipe separately. A potential first empirical-response recipe
would restrict to free post-purification 1H measurements and use peptide-grouped
dependence, reported nominal dose and the selected-panel protocol. Exact isotope
chemistry or acquisition-level claims require stronger support. Group all doses
and ordinal slots for each peptide when assessing sequence generalization; a
replicate split would leak peptide identity. Specify constant, dose-only and small
sequence controls before fitting. An empirical signal model does not resolve
native response transfer, extraction or APC concentration identifiability.

## Verification, resources and closure

LocalCPU/one numerical thread;20min/stage,3GiB startup free+speculative memory and
process-tree RSS,1GiB disk reserve+32MiB admission margin,50ms polling and OS child
peak checks. Disk is near capacity: only small processed files, no corpus copy or
large simultaneous tests. Keep failures immutable and correct causes before new
attempts. Verify exact raw cells and relationships independently, then close the
source family in README and experiments/experiment_log.md. No predictive split,
loss, synthetic labels or predictions are applicable to this source-only stage.
Model quality and the full biological goal remain unproven.
