# Dendritic-cell source retrieval: quantitative admission remains pending

October24,2026; Codex/GPT-6. Presto c7d538c, with existing task dirt and
registered source-retrieval files; Hitlist fda1eff/2.0.18 inspected unchanged.
The original 2010 DC quantitative supplements were not retrieved. No source rows
were manufactured, no canonical adapter was added, and no model was trained.

## Exact access results

Source PMID20484733 / PMC2938112 / DOI10.1074/mcp.M900566-MCP200.
13 retained retrieval attempts include three sandbox DNS failures and a separate
PyPI delivery check. Ten HTTP bodies have independently checked size/SHA256.
All per-attempt URLs, commands, timing, HTTP statuses and hashes are in results/.

| Route | Observed result |
|---|---|
| EuropePMC full-text XML | HTTP500 |
| Legacy PMC OA service | HTTP404; official documentation confirms retirement |
| Current PMC public archive | HTTP200, zero matching keys, untruncated listing |
| EuropePMC article metadata | HTTP200; supplements present, OA flag N |
| EuropePMC supplementaryFiles | HTTP200 containing an explicit non-OA error XML, not a ZIP |
| DOI landing page | Redirects to publisher; landing HTML retrieved |
| Publisher article and linked PDF | HTTP403 |
| Hitlist2.0.17 official PyPI JSON | HTTP404; deployment still unverified |

Do not interpret HTTP200 alone as source availability or failed routes as proof
that the supplements do not exist. No CAPTCHA or access control was bypassed.
The [official PMC instructions](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/) explain
the retired archive service and current public object listing. Public-source
retrieval through that service is not cloud training.

## Biological contract still required

The [primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2938112/) describes
MHC-associated DC peptides under concurrent LMP7/MECL1 loss, with a separate
beta2m background control. Its processing includes a predicted MHC motif filter.
For undetected dKO peptides, reported ratios use a15000 intensity detection limit
and carry inequality qualifiers; these must not become exact observations or zero.
RNA comparisons and changes in other machinery components remain distinct from
measured precursor protein inputs. These claims guide admission, not training rows.

Hitlist798 stays open. This bounded retrieval assessment is complete; quantitative
admission is not. An accessible original supplement or repository deposit is still
needed. No author messages were sent. Existing Hitlist726 offers an independent
next action: recover quantitative HAP1 machinery-panel evidence. HAP1 is not a
professional APC and cannot replace the requested final APC validation.

## Reproduction, evaluation and verification

Run each exact recorded `command` through reproduce/launch.sh, using new attempt
names and a fresh source root to preserve existing receipts. The launcher is frozen
in reproduce/source/launch.py; config and prospective plan are in reproduce/.
Raw HTTP bodies remain under the source_root in reproduce/launch.json. They are
not duplicated in the Git report. `analysis/close.py` verifies all body hashes,
receipt copies, metadata identity, empty archive listing and explicit API error.
`results/access-summary.json` records the independently checked decision.

HTTP time totals5.59s; local retrieval/metadata work only, no GPU/Modal/shared
environment changes. No fitting/pretraining, loss, synthetic data, assay-output
mapping, split, validation/test predictions or nM/binder metrics apply to this
source-access assessment. No predictive winner or concentration claim.
