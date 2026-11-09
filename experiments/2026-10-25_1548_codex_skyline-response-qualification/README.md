# PASS05890 processed response-curve qualification

Date: 2026-10-25. Agent/model: Codex / GPT-6. Source-access experiment, closed
with an unresolved external access dependency; no model was trained.

The publicly listed `CPTAC_Exp1_Curves_CS.sky` remains a candidate for separate
heavy/light response areas. Its contents and dose joins have **not** been inspected.
The HTTPS upload browser returns the same 15-file inventory as the dataset page:
the 7.8M Skyline filename is plain text, with no download link. Independent
stdlib HTMLParser and BeautifulSoup extraction agree on the inventory and links.
The no-login FTPS diagnosis also finds both hostname mismatch and certificate
expiry. Neither a successful HTML request nor a listed filename admits training data.

| Condition | Outcome | Runtime | Peak RSS |
| --- | --- | ---: | ---: |
| Published HTTPS upload-browser route | HTTP 200, 26,747 bytes, inventory only | 9.584 s | 43.30 MiB |
| Independent inventory/hash replay | Pass; 15 entries, zero links to target document | 0.193 s | 40.28 MiB |

The FTPS diagnostic predates registration and is preserved in
`results/prior-ftps-certificate-diagnostic.json`; it is not included in the two
stage timings. Its `s_client` exit code zero is not TLS verification success.

Official software/documentation discovery yielded no supported alternative file
route. See `results/documentation-discovery.json` for a protocol deviation: two
search queries and four tool-mediated opens/clicks occurred outside the guarded
fetcher. Together with the one guarded request, this exceeds six operations under
an inclusive reading of the request ceiling; browser-tool byte/time receipts are
unavailable. Do not claim the entire discovery phase met the frozen network bounds.
No further retrieval is scheduled here. No private credentials, author outreach,
TLS-validation bypass or vendor spectra download was used.

## Dataset and interpretation

Fixed source: [public PASS05890 upload inventory](https://db.systemsbiology.net/sbeams/cgi/PeptideAtlas/PASS_View?identifier=PASS05890&browseArea=upload&path=/).
Prior [family1451](../2026-10-25_1451_codex_mrm-anchor-admission/README.md)
contains the primary study/supplement audit and exact Panorama routes that returned
sign-in pages. Its 48 linked versus 53 titled assays remain unreconciled. Known
stock amounts, LLOQs, ratios and inferred plasma concentrations are not replacements
for separate raw areas. No source units, censor dispositions, isotope aggregation,
run/dose joins or amount conservation inside the Skyline document are verified.

No pretraining, training, synthetic data, losses, assay-label/output mapping,
validation/test splits, prediction dumps or predictive metrics apply: this family
checks public access only. Native held-out outcomes remain closed. The winner is
**none**. No gain in biological prediction or APC concentration accuracy is claimed.

## Reproduce and artifacts

Starting commit: `d40f51d00d68d928e342d2ce9c937b61f3c63b77`; tracked tree dirty
because unrelated task edits were preserved and this source qualification was
registered. Source hashes and exact stage invocations are in `reproduce/launch.json`
and the per-attempt launch files. From the worktree root:

```sh
experiments/2026-10-25_1548_codex_skyline-response-qualification/reproduce/launch.sh fetch --attempt 1 --requests experiments/2026-10-25_1548_codex_skyline-response-qualification/reproduce/requests/metadata-1.json
experiments/2026-10-25_1548_codex_skyline-response-qualification/reproduce/launch.sh verify --attempt 1 --requests experiments/2026-10-25_1548_codex_skyline-response-qualification/reproduce/requests/verify-1.json
```

These commands intentionally refuse to overwrite existing attempts; replay in a
fresh artifact location with reviewed path changes. Local CPU, Python 3.12.6,
requests 2.34.2, psutil 7.2.2; verification additionally uses BeautifulSoup.
One BLAS thread, no Modal/GPU, no shared-environment changes. Both guarded stages
pass unchanged 20-minute / 3-GiB process-tree / 3-GiB startup free+speculative /
1-GiB disk reserve limits, with 50-ms polling and OS peak accounting. Total guarded
runtime 9.777 seconds; maximum 43.30 MiB. Browsing discovery is excluded as noted.

The original HTML is stored once at
`data/skyline_response_sources/2026-10-25_1548_codex_skyline-response-qualification/pass05890-upload-index.html`,
SHA256 `db6c9ea079bf615f14364337cc96e555055341ece2d0e5e58bc4c72c99f8d40c`.
It is not redistributed; receipts, safe inventory, runtime summaries and frozen
source are in this directory. `results/verification.json` is the semantic limit
record; `results/summary.json` contains runtime counts. Original scientific
Skyline bytes retrieved: **zero**.

## Handoff

**Status:** access qualification closed; semantic admission remains unperformed.
**Next step:** obtain a repository-supported HTTPS link or restored verified FTPS
for the exact standalone document, then resume the frozen signal/dose qualification
through shared Hitlist curation. Do not retry these unchanged routes or train on
summary/ratio substitutes. Existing Hitlist #762 owns the dependency.
**Open questions:** separate raw isotope areas, intended query dispositions,
source-backed doses/preparation, 48/53 membership, and transfer from targeted MRM
standards to native/APC measurements. Full precursor-relative APC availability,
short/unobserved products, machinery effects and matched biological exact/RL
validation remain unachieved.

Software checks: tracked-checkout lint/format pass, and all six new Python files compile. Whole-directory traversal reports style errors only in an unrelated untracked historical `data/erap1_source_qualification/.../executed-launch.py`, preserved unchanged. Optional isolated lint reports import ordering, broad exception handling and regex aliases in the already-executed fetch/launcher snapshots; these diagnostics are retained in `results/software-checks.json`. Repository policy excludes frozen experiments from lint rewriting. The new verifier passes isolated lint and format. No production training code changed.
