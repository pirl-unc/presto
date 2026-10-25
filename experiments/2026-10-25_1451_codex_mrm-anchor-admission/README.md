# MRM calibration audit: supporting metadata verified, Skyline access pending

2026-10-25; Codex / GPT-6. **No model was trained.** This bounded source audit
verified calibration metadata from two studies and identified a small processed
response-curve document. That document's contents remain unverified because the
published server fails TLS hostname verification. Neither source is admitted as
raw amount–signal training pairs. The full APC availability goal remains active.

The preceding [nine-standard dose diagnostic](../2026-10-25_1423_codex_dilution-dose-diagnostic/README.md)
favored a fixed unit slope. We sought broader sequence/amount support without
extending that model or reopening native outer partitions.

## Results

| Source | Retrieved and independently checked | Available support | Missing for training |
|---|---|---|---|
| [Michaud2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6123701/), PASS01163/PASS01200 | Eight XLSX files, ten sheets, DOCX descriptions; 60,825 stored-cell positions | 500 unique standards with stock concentration, purity, adjusted concentration, heavy/light m/z; limits and assay summaries | Separate isotope areas with exact run/dose/form joins and dispositions |
| [Sobsey2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11812018/), PASS05890 | Both XLSX files, five sheets; 14,507 stored-cell positions; PDF assay table | Transition settings, assay summaries and48 matched assay/peptide links | Contents and semantics of the listed response-curve Skyline document |

All eleven Office files match official inventory sizes/MD5 and local SHA256
receipts. ZIP integrity passes. Independent XML parsing and openpyxl agree at
all75,332 represented cell positions, with no unsupported/date-conversion cells.
Counts include blank/styled positions; they are not independent observations.
Original bytes and cached values are unchanged; no formulas were recalculated.
See [primary inspection](results/supplement-inspection.json),
[fallback inspection](results/fallback-inspection.json), and
[admission summary](results/source-admission-summary.json).

All500 primary stock corrections were checked as H×I/100 versus J.498 agree at
1e-12 relative/absolute tolerance. Rows115 and255 differ by0.000016 and0.0004
pmol/µL, consistent with rounding to three decimal places. Preserve reported
values and precision; these are stock solution concentrations, not injected
amounts or certified exact molarity.

The primary supplement called raw data contains inferred plasma concentrations,
not raw instrument areas. Other tables contain LLOQ, CV, response-curve slopes,
stability and protein estimates. The fallback's Trigger Ratio is an acquisition
setting, not a measured isotope ratio. Sheet layout, values and descriptions
support this distinction; it is not a column-name-only absence claim.

The fallback title reports53 assays; TableS3 contains48 assay/peptide/URL triples.
pypdf and Poppler independently agree on the ordered48 triples; rendered pages4–5
were visually checked. The panel difference remains unresolved, not five invented
negative observations. pypdf emitted cross-reference recovery warnings; Poppler
provided the independent check. [Exact links](results/fallback-assay-links.json).

## Access findings and next asset

PASS05890's public inventory explicitly lists **CPTAC_Exp1_Curves_CS.sky (7.8M)**.
This small processed document is the next asset to qualify. Its listing alone does
not prove that it contains the needed separate areas or amount/run mappings.
Only that standalone document was requested; no spectra, cache or large archive
was downloaded.

- The exact primary and fallback Panorama pages require login. No restricted
  alternate API or private user session was used.
- Complete PeptideAtlas HTML initially failed our UTF-8 assumption. Reversible
  Latin-1 inspection recovered the public listings; original failed receipts remain.
- The Europe PMC supplement ZIP timed out after73,268 bytes. That partial was not
  parsed. Individual PMC bin URLs returned HTML instead of Office files.
- The legacy NCBI OA lookup returned404. Current documentation confirms its
  August2026 retirement. The official anonymous [S3 distribution](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/)
  supplied all verified supplements. Published-article metadata records CC BY for
  the primary and CC BY-NC-ND for the fallback. Source workbooks are not republished.
- An explicit amendment permitted only the repository-published public dataset
  FTP login, read from archived HTML without logging its password. The server
  rejected cleartext FTP with421 requiring TLS. Explicit TLS then failed hostname
  verification for ftp.peptideatlas.org. **Zero Skyline bytes downloaded.**
  A no-login TLS probe of the official web hostname timed out and resolved to a
  different address. Certificate verification was never disabled.

See [TLS transfer receipt](results/pass-skyline-tls-retrieval.json) and
[compact retrieval receipts](results/retrieval-summary.json). These are specific
route limits, not proof of global data absence. Next obtain a repository-supported,
verifiable route or mirror for the exact Skyline document; do not retry the retired
OA API or disable TLS checks. [Hitlist#762 update](https://github.com/pirl-unc/hitlist/issues/762#issuecomment-6399609459) records the exact verified fields, access limits and acceptance requirements.

## Curation and modeling decision

[Field/source contract](results/curation-contract.json) maps available and pending
fields into the existing Hitlist evidence/catalog/query/material-context and
calibration-applicability system. Preserve study/accession, asset/checksum/cell,
sequence/exact form/isotope/charge/transition, stock/purity/assignment, injected
amount, run/preparation/replicate, matrix/species roles, digest/spiking stage,
instrument/LC/acquisition, selection/integration and all intended observations.
Unknowns, zeros, absent/unintegrated peaks, bounds and rejected observations must
remain distinct. No Presto-only raw workbook training adapter was added.

These are selected targeted assays. Post-digestion synthetic standards do not
label intracellular proteolysis. Plasma/BSA/cell-lysate matrices and inferred
protein concentrations are distinct from native APC free-peptide abundance.
Ratios or LLOQ alone cannot identify absolute peptide response. MRM coefficients
must not be applied directly to DDA, DIA, MAPP or MHC elution.

After access succeeds, freeze the actual Skyline field/run/dose mapping, reconcile
separate isotope areas and the intended query universe independently, then add a
shared Hitlist source/recipe. Register a simple held-peptide response experiment
and protocol-transfer test before returning to matched exact/RL availability fits.
Native biological validation and concentration/precursor-relative calibration
remain required for the full goal.

## Reproducibility and resources

Source commit `22241241daaae1b223d4ed6e0543f8fa1afcad9a`, dirty with preserved
unrelated task work and this registered audit. [Prospective plan](reproduce/plan.md)
predates retrieval. Immutable request sets, amendments, executed source snapshots
and per-attempt `*-launch.json` files preserve exact argv/configuration/env.
The original [launch configuration](reproduce/launch.json) and `launch.sh` remain;
later invocations identify their versioned launchers. Non-overwrite checks
intentionally refuse to rerun into existing artifacts.

Local CPU only; no Modal/GPU, cloud computation or environment changes.20 HTTP
metadata and15 processed HTTP requests received2,481,660 bytes including failed
payloads/partials.22 bounded stages:14 passed,8 failed; total measured stage time
111.49s; largest child RSS112.19MiB. All stages are terminal. Verification stages
passed; access/parser failures remain in the record.

Guards stayed20min/stage,3GiB process-tree RSS,3GiB startup free+speculative,
1GiB disk reserve plus32MiB startup margin,50ms monitoring and OS peak. BLAS thread
overrides were1. Asset/family byte limits256/512MiB were not raised. Runtime
versions: openpyxl3.1.5, pypdf6.21.2, requests2.34.2, psutil7.2.2.

Raw inputs and full receipts remain once under
`data/mrm_anchor_sources/2026-10-25_1451_codex_mrm-anchor-admission/`.
Some raw HTML/error logs embed repository-published login details. Publication uses
sanitized summaries and original hashes; those raw files/logs remain local.

No pretraining, training, synthetic examples, loss, model output, validation/test
split or model winner applies. No predictions or binding-threshold metrics exist
because this is source admission. Earlier native outer data and predictive-quality
findings remain unchanged. This audit does not establish APC concentrations.
