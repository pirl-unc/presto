# APC-derived endolysosomal source qualification

Codex / GPT-6, October 24, 2026. Registered before supplement/deposit retrieval.
[Prospective plan](../agents/codex/plans/endolysosomal-source-qualification.md);
[upstream owner](https://github.com/pirl-unc/hitlist/issues/812).

Question: are original quantitative peptide time courses publicly available to
constrain processing before MHC selection? Source is Öztemiz2026,
DOI10.3389/falgy.2026.1936335, an isolated murine dendritic-cell-derived preparation.
This is not an intact APC concentration cohort. Linked PeptideStudio synthetic
examples are excluded. No new source curator, model fit or held-out evaluation.

**Decision: no quantitative peptide cohort admitted from the inspected public
assets.** The five official supplements contain scientific figures, including one
aggregate protein half-life table. They do not provide the original numeric
peptide-by-sample measurements needed for this model. This is a limited source
qualification result, not evidence that the authors lack those measurements.

## Source and inspection contract

Registered at Presto `bf7b3e64e484fd280ce311dffe2c15a93e6c4aca` with existing task
dirt and new audit files. Source: Öztemiz et al., DOI10.3389/falgy.2026.1936335,
murine JAWSII-derived isolated endolysosomal preparations, Bet v 1 and Aes h 1.
Primary article: <https://www.frontiersin.org/journals/allergy/articles/10.3389/falgy.2026.1936335/full>.
Original HTML/XML, public publisher asset metadata, all five listed version1 PDFs,
and the linked repository at commit18dd99da4a619abd8a28c7622e0809eb2c03e0ff were
inspected. Original bytes remain under
`data/endolysosomal_source_qualification/2026-10-24_1521_oztemiz-v1`.

| Official supplement | Evidence present | Quantitative peptide admission |
|---|---|---|
| S1 | Reporter-activity curves | No original peptide measurements |
| S2 | Protein half-life graph and aggregate n/mean/SD/SEM table | Different assay; no peptide-product labels |
| S3 | Bet v 1 peptide span map with intensity color bins | No original numeric sample table |
| S4 | Aes h 1 peptide span map with intensity color bins | No original numeric sample table |
| S5 | Aes h 1 peptide intensity time-course graph | No original numeric sample table |

All five pages were rendered and visually reviewed. Independent Poppler checks
confirm one page and zero embedded attachments in each PDF. S2's eight literal
summary values independently match: Bet v 1/Aes h 1 n=7/4, mean=1/16 hours,
SD=0.28/3.3 and SEM=0.1/1.6. These remain aggregate protein-depletion statistics;
they were not converted into peptide quantities or independent replicate rows.
No plotted curve or color value was digitized. Pypdf hit a form-XObject extraction
limit on S3/S4; its text is not claimed complete. Poppler extraction and rendered
visual review supply complementary evidence.

The official supplemental-data endpoint resolves all five download URLs. The
publisher's separate, read-only Figshare metadata query returned HTTP404; the
traceback and a posthoc failure receipt are retained. It does not prove there is
no deposit. Initial sandbox DNS failure is also retained, followed by successful
public retrieval. No access controls were bypassed and no authors were contacted.

The complete, nontruncated repository tree has61 entries. Its only workbook is
`sample_data/Example_degradome.xlsx`; the pinned README explicitly identifies
every example intensity as artificial. That workbook was excluded without
downloading or training on it. Repository availability does not establish
availability of the study's experimental measurements.

## Metadata needed for a future shared recipe

Original peptide/form-by-sample intensities, time points, preparation and replicate
lineage, input/construct identity, identification/aggregation filters and missing
versus zero semantics remain necessary. Reported MS search length limits exclude
short products; missing short peptides cannot be labeled nonproduced. The source
describes intensity as semi-quantitative. Reporter activity normalization is not
peptide-specific molar calibration. Reduction/alkylation changes observation and
must remain separate from the biological processing context. Even original
numeric files would label an isolated preparation, not intact APC concentrations.

The shared ingestion/query owner remains Hitlist812. No Presto production curator
was added. Aggregate protein-depletion evidence can be retained as its own assay
family; it does not satisfy the peptide-product admission gate.

## Verification, reproduction and scope

`results/independent-source-verification.json` verifies15 successful asset hashes,
14,805,139 PDF bytes, five pages, zero embedded files, eight literal table values
and61 repository entries. Original retrieval time sums to14.97s, excluding the
failed Figshare request and interactive discovery. PDF inventory/rendering took
17.61s with0.440GiB peak process memory; independent verification took0.43s and
0.047GiB. These are local CPU source-inspection costs, not model benchmarks.

`reproduce/launch.json` and `launch.sh` preserve the initial invocation;
`reproduce/asset-invocations.json` reconstructs subsequent retrieval argument
vectors from receipts and lists the executed inspection helpers. Frozen sources
and plan are included. Reproduction uses a fresh copy of the registered directory
layout: the launchers intentionally refuse overwriting original assets/results.
Raw publisher HTML/JS and the full Nuxt state remain local, with hashes and paths,
rather than publishing irrelevant frontend configuration. Original extraction
texts and launch logs are preserved losslessly in the accompanying archives.

New verification code passes isolated Ruff. Frozen retrieval scripts retain two
preexisting UP017 timezone-alias style findings; the remaining isolated checks
pass with that rule excluded. Their executed source hashes are unchanged.

There is no training, optimizer, pretraining, synthetic biological data, loss,
assay-to-model output mapping, validation/test split or prediction dump: this is
source qualification before data admission. Predictive metrics and binding
thresholds therefore do not apply. No existing biological model or held-out
partition changed. The full APC availability objective remains unachieved.

**Status:** source qualification and independent verification complete.
**Next step:** preserve the concrete source gap in Hitlist812 and qualify an
independent measured MS-response corpus under Hitlist762.
**Open question:** availability of original peptide/sample quantities beyond the
inspected public assets, without assuming author-held data are absent.
