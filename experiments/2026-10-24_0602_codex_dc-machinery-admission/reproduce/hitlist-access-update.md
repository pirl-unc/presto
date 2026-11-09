Verified access update (October24,2026): quantitative admission remains pending.

- Official EuropePMC metadata confirms `hasSuppl=Y`, `isOpenAccess=N`.
- Its documented supplementaryFiles endpoint returns HTTP200 with an XML error explicitly declining non-OA distribution. It is not a downloaded archive.
- The current PMC public S3 prefix listing has zero matching article-version objects and is not truncated. The legacy OA API has been retired, so its404 is not evidence that the study lacks supplements.
- The DOI resolves to MCP/Elsevier; the publisher article and its metadata-linked PDF return403 in this environment. No access controls were bypassed and no author outreach was sent.

The primary article adds two admission requirements: retain its predicted-MHC-motif search filtering as ascertainment metadata, and retain lower-bound WT/dKO ratios derived using the reported15000 detection limit as inequalities rather than exact values or KO zeros. Source reference: https://pmc.ncbi.nlm.nih.gov/articles/PMC2938112/ .

All13 retrieval attempts,10 hashed response bodies and independent receipt verification are preserved in Presto `experiments/2026-10-24_0602_codex_dc-machinery-admission`. No original quantitative table, training rows or adapter has been admitted. This issue remains open for the original supplements or an authorized repository copy. Continuing existing#726 HAP1 quantitative admission while this access gap remains; HAP1 does not replace professional-APC validation.
