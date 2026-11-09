# HLA-DO quantitative source qualification

Codex / GPT-6, October24,2026; registered at Presto e8d5daf with existing task dirt. **Source access gap: no quantitative training cohort admitted.** This is an independent candidate from the original architecture plan, not a rerun of exposed MAPP/macrophage/HAP1 tests.

## Source and intended use

Nanaware2019, PMID30573663, DOI10.1074/mcp.RA118.000956, PMC6398211, MSV000082570/PXD010301. [Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6398211/) and [repository](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?accession=MSV000082570). The source reports human LG-2 control/DO-deficient clones and mouse B-cell comparisons, per-peptide and predicted-core summaries, selected isotope-standard amounts and binding measurements. These are possible downstream presentation/observation constraints; DO is loading/editing context, not an upstream cleavage rate. Predicted class-II cores are derived annotations, not measured registers. The nine standards were outcome-selected; their generality is limited. MHC-bound amount is not free intracellular concentration.

The source's current Hitlist legacy arm metadata explicitly records an axis mismatch: human and mouse identities are distinguishable in IEDB, but DO status is not. Existing species-context assertions do not resolve quantitative sample/clone joins. No source or legacy evidence was rewritten by this audit.

## Retrieved evidence

| Evidence | Result |
|---|---|
| Official ProteomeXchange descriptor | PXD010301/MSV000082570 and source replicate-name crosswalk retrieved |
| Complete embedded MassIVE inventory | **328 files**, asserted total and unique descriptors independently verified |
| File classes |87 raw,69 result,69 ccms_result,69 peak,26 ccms_peak,4 parameters,4 statistics |
| First WT processed mzTab |3,385 PSM rows and612 protein rows; no quantitative headers |
| Same acquisition mzIdentML |4,800 identification items /4,751 spectrum results; no intensity/abundance/quantity terms in its complete CV/user field inventory |
| Three deposited MSF files |286–294MB each; all named parental LG-2, not the WT/KO comparisons; not retrieved under the100MiB/asset cap |

The two inspected result files are identification exports, not the reported quantitative tables. Their absence of quantity fields is **not** an exhaustive claim about the remaining repository results. No RAW acquisition, unobserved-negative generation, imputation, synthetic biology or training-label conversion occurred.

## Supplement access and payload validation

- Europe PMC fullTextXML: HTTP500. Preserve as retrieval failure.
- Legacy PMC OA API: HTTP404. Current official [documentation](https://pmc.ncbi.nlm.nih.gov/tools/oa-service/) says the service was retired in August2026. This is not source absence.
- Documented [PMC Cloud Service](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws): article-specific listing complete, zero objects; the public README is pinned.
- Europe PMC supplementaryFiles: HTTP200 **XML error**, explicitly stating this article is not open access; not a ZIP and not a successful supplement retrieval.
- PMC supplemental index: HTTP200 **reCAPTCHA page**, not the source index. No attempt to defeat it.
- Publisher DOI: public redirect page only. Guessed public publisher media path:mmc1.pdf returns404, which does not establish that publisher supplements are absent. Browser publisher fulltext access also returned403.

Original S10–S12 quantitative tables, S13 selected isotope standards and S14 selected binding data were not recovered through these routes. Source-listed filenames are recorded in the upstream request. No email, private/reviewer credential, author outreach, access-control bypass or raw-data reanalysis was used. Raw downloaded redirect pages remain local because they contain incidental server/client metadata.

## Reproduction and verification

`reproduce/plan.md`, launch.json/sh and stage-specific manifests/scripts preserve prospective scope, original requests and environment. All executed sources are under `reproduce/source/`; first-acquire source retains one broad-exception Ruff style finding, while the reusable retrieval and final audit scripts pass isolated Ruff. Every successful HTTP payload has a receipt/hash; successful HTTP alone is not semantic admission.

Run the frozen acquisition stages in order, then:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /Users/iskander/code/shared-virtual-env/bin/python experiments/2026-10-24_1818_codex_do-source-qualification/analysis/audit.py
```

Originals: `data/do_source_qualification/2026-10-24_1818_v1`. Inventory and schema summaries are in `results/`. Independent HTML-script/JSON extraction, CSV versus literal TSV splitting, and ElementTree versus lxml reconstruction agree. **10 asset hashes /14,866,858 bytes /328 inventory entries** verify. Audit2.66s/0.207GiB peakRSS; local CPU, no Modal/GPU or shared-environment changes. Acquisition times are preserved individually, not a training benchmark.

No model, pretraining, optimizer, loss terms, assay-head mapping or predictive split was created. No validation/test metrics or prediction dumps apply; source admission precedes fitting. Existing biological baselines remain unchanged. Full APC peptide availability/concentration objective remains unachieved.

## Decision

Request canonical retrieval/curation of original quantities, clone/species/replicate/acquisition identity, normalization/form/selection and exact-versus-surrogate protein pairing. Keep this access limit explicit. Qualify the independent2024 afferent-lymph/DC source next; it could provide native non-MHC peptide measurements alongside a separate downstream presentation dataset, but its pairing and quantitative support are not yet established.

Shared source owner: [Hitlist816](https://github.com/pirl-unc/hitlist/issues/816), posted and exact-body verified.
