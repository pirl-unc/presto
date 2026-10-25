# Original Methods qualify an unmodified-search interpretation

The original PXD058121 study explicitly reports “No additional modifications
were included” in its ClipsMS search. Independent ElementTree and minidom parses
agree on the original Methods paragraph (article.xml#s15, XML paragraph43).
This supports a **method-reported search assignment**, not proof that a molecule
has no modifications. Original F3 missing cells remain missing. Strict family1655
stays closed with zero admitted examples; its policy was not retroactively changed.

Primary study: https://pmc.ncbi.nlm.nih.gov/articles/PMC11874349/,
DOI10.1073/pnas.2419607122, section ID-MS Data Analysis and Statistics.
The same paragraph specifies ClipsMS,10ppm, minimum4aa, UniProtP37840 and b/y/by
assignments. Original article SHA25634c2d374b62afe17c48e04bdfd75a5f91e23786db1d537035d632c0f7f411bea;
paragraph SHA256557facde6e07c61a824c7e71d00b1025847dd578d1ac63c24d0c6680f79f7282.

The [original software paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8174100/)
and [official repository](https://github.com/loolab2020/ClipsMS) document optional
modifications. Pinned commit8ad829bdb5d2b644a290551442a8728b3b8ac5b5 (2022-07-22)
contains a `None`/zero-mass no-modification CSV and a function installing it.
Static AST/CSV inspection verifies that convention and the commit's Git-blob hash.
**No downloaded source code was executed.** Neither the exact study software
version nor its CSV-to-workbook blank conversion is established. The original
Methods declaration, not a presumed export roundtrip, is the interpretation basis.

## Reproduction and resources

Codex/GPT-6;2026-10-25; Presto b8150780ede2272908cde3fac828e35bab240973 with
pre-existing unrelated task dirt plus the registered plan. Sources, request hashes,
configuration and exact invocations are under `reproduce/`; qualification evidence
is in `results/qualification.json`. Raw external assets remain once under
`data/intact_search_contract_sources/2026-10-25_1735_codex_intact-search-contract`. The original article remains
under `data/intact_degradomics_source/2026-10-24_2338_codex_intact-degradomics-source`.

```sh
reproduce/launch.sh fetch --attempt 1 --requests reproduce/requests/fetch-1.json
reproduce/launch.sh inspect --attempt 1 --requests reproduce/requests/inspect-1.json
```

Run these from this experiment directory. Python3.12.6, localCPU, one numerical
thread, PYTHONSAFEPATH=1, no bytecode writes; exact overrides are in launch.json.
No Modal/GPU. Two guarded stages passed in 1.266s combined;
maximum process-tree/OS peak 96.30MiB. Guards:20min/stage,
3GiB startup free+speculative/RSS,1GiB disk reserve+32MiB margin,50ms polling.
Eight inclusive discovery operations were used, including four pinned downloads
(710,944bytes total). Browser discovery has no measured local
byte/time guard; the report does not claim it does. No additional source discovery
is authorized by this family. Requests, receipts and completed ledger are preserved.

## Scope and decision

No quantitative F3 outcomes, model scores, training or pretraining were queried.
No synthetic labels, loss terms/weights, assay-to-model targets, validation/test
split or predictions exist in this source-only family; affinity/binder metrics are
inapplicable. There is no model winner.

Use a separately named shared recipe with a typed reported-method assertion,
source/paragraph hashes and explicit scope. Permit a blank unlocalized field only
when reported localized/terminal fields are both zero and this Methods evidence
is present; retain all other eligibility rules. Preserve original raw nulls and
both strict recipes. Register a new frozen-model comparison before numeric
admission. Family1747 does that with the unchanged21arms/31input hashes and gates.
This qualification does not resolve concentration calibration or APC transfer.
