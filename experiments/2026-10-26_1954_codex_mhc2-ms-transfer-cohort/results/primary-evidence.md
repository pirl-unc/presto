# Primary evidence and retrieval ledger

Codex / GPT-6, 2026-10-26; reviewed before MS model fitting.

- **DP35251023:** [original Methods](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2022.831822/full) describe defined paired HLA-DP K562 populations and DP-targeted capture. The described MS-list creation uses identification-confidence filtering; no binding-prediction filter was identified in that section. This supports scoped detections and reported pairs, with engineering/arm metadata still under-annotated. It does not establish isolated processing.
- **DQ834433824:** [original Methods](https://www.nature.com/articles/s41467-021-25404-x) describe DQ capture from9033 BLCL and an MS confidence-filtered list; binding predictions are subsequently descriptive in the reviewed section. Spliced/modified peptides are also analyzed, so source unmodified status does not establish linear origin. Methods specify the alpha family; exact paired tokens here come from IEDB and retain that separate provenance.
- **Atlas33858848:** [original article](https://pmc.ncbi.nlm.nih.gov/articles/PMC8054196/) distinguishes detections from predictor-derived named associations, permits several donor alleles per peptide and retains predicted nonbinders in analyses without specified associations. Sample QC also uses binding predictions. Donor typing must remain separate from computational restrictions; exact native association joins remain necessary.

Verified Atlas XML:
`/private/tmp/presto-mhc2-20261026/ms-transfer-cohort/primary-attempt02/PMC8054196.xml`,
167,469bytes, SHA256
`1d901f85a12c3772d785688b84473044d34c699d8d339af005b43d97c6dd7edb`.
Article IDs match PMID/DOI. The retrieval receipt pins URL/response.

For25502872,30573663 and33592498, snippets, curator notes and earlier repository
audits are available, but full Methods were not retrieved in this stage. Their
complete source policies remain unresolved. Preserve existing DO quantity issue817
and current CIITA ambiguity corrections; do not duplicate them.

## One primary retrieval budget

Original budget:16 logical discovery/open operations and20minutes. After operation16,
a prospective amendment allowed24 total within the same time window because six
PMC pages returned challenges and the Atlas publisher returned403. Retrieval ran
approximately19:55–20:03UTC and stopped at24.

| Operations | Action | Outcome |
|---|---|---|
| 1–4 | Search35251023,33858848,25502872,30573663 | Primary discovery |
| 5–7 | Focused Atlas, CIITA and Tran searches | Primary discovery |
| 8–13 | Six selected PMC opens | Browser challenges |
| 14–16 | DP Frontiers, Atlas JITC and Tran Nature | DP/Tran retrieved; JITC403 |
| 17–19 | DP/Tran text windows, including one incorrect window | Methods read |
| 20–23 | Four EuropePMC XML requests | Sandbox DNS failures |
| 24 | Explicit network-enabled Atlas XML request | HTTP200; article IDs/hash verified |

No subsequent primary retrieval belongs to this family. Snippets are not complete
Methods. Original failed and successful XML launchers/receipts remain separate.
No author outreach or permission-controlled data was requested.

GitHub ownership was separate follow-up: three initial searches/#702 readback,
then #593/#31 readback and focused prediction-provenance search. Issues855/856
were created by CLI after the connector's first create returned403; both bodies
were read back. Those API actions are not counted as primary-paper retrievals.
