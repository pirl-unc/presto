# Measured peptide MS1-response source qualification

Codex / GPT-6, October24,2026. Registered before original source retrieval.
[Prospective plan](../agents/codex/plans/measured-ms1-source-qualification.md).
Existing shared owner: [Hitlist762](https://github.com/pirl-unc/hitlist/issues/762).

**Outcome: original nontryptic peptide MS1 measurements and their intended pool
membership are accessible and verified.** A bounded shared response experiment is
now feasible. These are crude synthesis pools with estimated injected amounts;
they do not independently calibrate true peptide molarity or intracellular amounts.
No model was trained in this source qualification.

## Why this source, and what it measures

The2023 [MS1-response reanalysis](https://doi.org/10.1016/j.csbj.2023.07.027)
points to PXD004732/PXD010595/PXD021013. The inspected original source is
PXD021013, described in the2021 [ProteomeTools nontryptic study](https://doi.org/10.1038/s41467-021-23713-9).
Its synthetic human peptide standards were physically synthesized and measured;
they are experimental observations, not artificial numerical labels. HLA-I/HLA-II
library origin describes sequence selection, not MHC binding or restriction
measured in these MS runs. No MHC pull-down or APC processing occurs in the
synthetic-pool measurement itself. In-silico unspecific cleavage is a search
setting, not an experimental digestive enzyme.

The original Methods describe crude synthesis products and estimated200or500fmol
injections per peptide. Per-peptide purity/yield and per-run choice between these
estimated amounts are not established by the inspected tables. Thus a model of
these signals can learn a conditional synthesis/handling/LC/MS-response prior;
calling it independently calibrated ionization efficiency would overstate it.
The2021 correction changes figure cross-references, not these source methods.

The2023 reanalysis removes zero-intensity, high-PEP, contaminant and reverse rows,
merges repeated sequences using median intensity, and selects low-CV peptides.
Its final model further selects specific-search tryptic sequences. Neither its
reported accuracy nor its final training distribution qualifies nontryptic APC
detector transfer. Its supplement contains artificial proof-of-concept and model
summary tables, not a replacement for original pool-level quantities.

## Verified source coverage and new findings

Official directory:1,865 raw files,1,864 downloadable search ZIP entries and one
search item exposed as a directory ending`.zip/`, plus README/SDRF. The original
README lists1,865 RAW/1,865 SEARCH records. Its nominal ZIP-versus-directory
discrepancy is retained. Two file API queries returned empty HTTP200 bodies;
the actual public directory and README recovered the file identities.

The community SDRF covers1,008 of1,865 raw-file names. All its rows label material
as`cell`, instrument as`Orbitrap Fusion`, acquisition asDDA/HCD, and assay as
`run 1`. The source protocol instead describes synthetic pools, Fusion Lumos and
several acquisition methods. These annotations must not override the primary
source or become asserted biological specimen/replicate identities. Three repeated
modification columns are preserved by column position, not collapsed into a dict.
Both inspected runs join to exact original RAW names through summary.txt and SDRF.

The first named HLA-I/HLA-II DDA archives were chosen before inspecting quantities.
Their original five relevant tables contain3,649 peptide rows and3,738 molecular-
form rows. Literal tab cells, intensity-column equality, peptide/form IDs and raw
file joins independently verify. Zeros and identification failures remain source
states, not invented negative examples.

The complete original Supplementary Data1 was recovered from the public EuropePMC
bundle after a prospective resource-cap amendment. All610,670 populated mapping
cells independently match direct OOXML decoding and openpyxl. There are305,331
sequence-to-pool rows across421 pools:168,688 HLA-I,73,464 HLA-II,31,744 AspN and
31,435 LysN. These are mapping rows, not a claim of globally unique sequences.
Styled empty rows inflate sheet dimensions to~0.8–0.9million; they are not data.
The HLA-II legend says73,465; the actual populated sequence table has73,464.

| Sampled source pool | Planned sequences | Observed sequence rows | Observed and planned | Planned, positive, PEP<0.01, no reverse/contaminant | Corresponding unmodified positive quality form rows |
|---|---:|---:|---:|---:|---:|
| TUM_HLA_1 |1,000|1,961|860|759|677|
| TUM_HLA2_1 |500|1,688|491|484|470|

Crucially,1,101 and1,197 observed sequences are outside their respective planned
pool lists. Their identities must not inherit the estimated amount of an intended
standard. They may include synthesis-related products, other source matches or
contaminants; this audit does not assign a cause to every off-list identification.
There are140 and9 planned sequences absent from the peptide tables. Their absence
cannot distinguish synthesis failure, solubility, chromatography, MS response or
identification failure. This is not a pure detector-negative label.

## Decision and smallest useful next experiment

Source existence and two-sample intended-product joins pass. Quantitative training
admission remains dependent on a canonical Hitlist source/query recipe, explicit
form aggregation and context resolution. No Presto production curator was added.
The next recipe should retain original per-pool/run/form values and every
disposition, selecting intended full-length products with explicit ID quality.
Record estimated load and unknown purity separately from measured concentration.
Use original run names and README mappings; keep SDRF assertions with provenance
and conflict status. Preserve repeated acquisitions and sequence overlap for splits.

The first meaningful model test should predict **relative MS1 signal among observed,
intended products within a run**, using sequence features against constant and
length-only controls. Group pools, repeated sequences and related acquisitions
before splitting. This avoids pretending a source-wide response gain or individual
true molarity is known. It remains a synthesis-plus-observation prior until
independent transfer is demonstrated. No intensity-CV selection on held-out outcomes,
no unobserved-peptide zeros and no treating in-silico search enzyme as digestion.
Fit this observer with direct differentiation; there is no discrete processing
step here that requires RL. Later exact/sampled processing comparisons must share
the same independently evaluated observation module and a new biological cohort.

The PXD003472 known-protein-load lead remains uninspected in this bounded family;
it could supply a complementary digestion/response test but is not declared
equivalent to known peptide loads. The original APC availability goal is unchanged.

## Reproduction and verification

Registered at Presto2bf8ca1f0a5ee5ace7599920b6a43528be66e604, with task dirt and new
audit files. Original bytes and retrieval/extraction receipts are under
`data/measured_ms1_source_qualification/2026-10-24_1546_v1`.
Reproduce/launch.json and launch.sh pin initialization, source and environment;
per-asset receipts preserve each actual retrieval argv. Sources and both prospective
plan versions are retained. Retrieval helpers refuse overwriting original files.

The initial100MiB bundle cap stopped retrieval. The documented amendment permitted
streaming this asset up to400MiB/180s and total transfer up to1GiB, including the
first partial attempt. The232,332,843-byte bundle completed in50.03s. Original
workbook bytes63,188,883 and ZIP CRC/hash verify. Preserve all failed routes:
publisher challenge, CDN DNS, EuropePMC direct binary403 and OA-index404. These are
access results, not evidence of missing scientific data. No access control bypass,
author contact, raw-MS download, cloud compute or environment change occurred.
The original amendment heading rounded its minute forward; its caption correction
and executed helper remain preserved, with request UTC15:58:38 establishing timing.

Phase1 verifies369,548 literal source cells using independent CSV/pandas readers
in0.28s/0.224GiB. Planned-product verification adds610,670 populated workbook cells
using independent OOXML/openpyxl,26.58s/0.436GiB. Every sampled peptide row has a
source-linked disposition. The first index assertion failed because one nominal
ZIP is exposed as a directory; the original code/failure is retained and the
qualified inventory preserves that distinction. No source quantity was repaired.
`source-summary.json` and `pool-samples.json` preserve the earlier phase before
the intended-product table was recovered; `planned-product-verification.json`
and `qualification-decision.json` supersede only that membership-availability gap.

This is a source audit: no optimizer/pretraining/loss, model outputs, validation/test
split or prediction dumps apply. No existing test outcomes were reopened. Numeric
source conservation is not a new predictive result or proof of APC concentrations.

**Status:** bounded source qualification complete; shared recipe and training pending.
**Next step:** canonical intended-product relative-response admission underHitlist762,
then a separately registered pool/sequence-grouped predictive experiment.
**Open questions:** molecular-form response interpretation, synthesis uncertainty,
per-run load mapping and transfer to native non-MHC and MHC-elution measurements.
