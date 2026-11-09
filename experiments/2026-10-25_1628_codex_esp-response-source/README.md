# ESP source screen: response scores are not concentration-calibration targets

2026-10-25; Codex / GPT-6. Registered at commit
`e176f7d220360431e5c2002506710f71765ae28e`, with preserved unrelated task edits
and new experiment-local registration. [Plan](reproduce/plan.md),
[configuration and frozen hashes](reproduce/launch.json).

**No calibration dataset is admitted.** The original article is available and
independently verified; raw isotope amount–signal pairs are not recovered. No
model was fitted or biological held-out outcome opened.

The [primary ESP paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2753399/)
uses digest-derived training responses. It also describes selected isotope-labeled
MRM validation, whose underlying raw calibration records remain unqualified here.
Those are separate data contracts. The former cannot supply independent detector
calibration merely by relabeling its measured or predicted response score.
This does not prove that useful raw validation data are absent elsewhere.

## Retrieval and verification

Six discovery operations were used across both tools: exact-title search,
challenged PMC page, unsupported web-tool EuropePMC view, local EuropePMC request,
official PMC object inventory and its exact JSON metadata key. The local EuropePMC
request failed with HTTP500 and a150-byte JSON error; it is retained as failure.
The official distribution then supplied the111,323-byte original article XML.
One of four permitted processed assets was downloaded; no numerical supplement.
The inventory has article XML/TXT/JSON keys; it is not a supplement completeness claim.

| Check | Result |
| --- | --- |
| Official article MD5 / source identity | Pass |
| ElementTree versus minidom paragraph extraction | All82 agree |
| Frozen verification source/configuration and syntax | Pass |
| New verifier lint/format | Pass |
| Raw amount–signal records / model fits | 0 / 0 |

[Source decision](results/source-admission.json) retains hashed paragraph
coordinates and five supplement references without republishing article text.
The S3 metadata identifies an author manuscript under `TDM`, not PMC open access;
original article bytes stay local and are not redistributed. Five guarded stages
include one HTTP failure and four passes; all retain original resource limits.
Total guarded runtime **2.242s**, peak OS/polled RSS
**44.81MiB**; received **113,273 bytes**
including the failed response. Web discovery is separately recorded and has no
local byte/time receipts. No browser-derived timing is included in these numbers.

## Modeling and data boundary

No pretraining, training, synthetic labels, losses, assay-label/output mapping,
validation/test split, prediction dumps or predictive metrics apply to article
admission. IC50 and binder thresholds are irrelevant. No shared Hitlist source
or recipe was added because its quantitative records are unqualified. No external
teacher scores become target labels. Native outer partitions remain closed.
This negative screen changes candidate eligibility, not the biological baseline.

Full availability still requires separately supported processing, observation,
precursor/initial scale, unobserved products and machinery effects, then a matched
exact/discrete comparison on prospective biological evaluation. A model can be
numerically correct or predict a digest signal without identifying those quantities.

## Reproduction and retained artifacts

The exact immutable request sets are under `reproduce/requests`. From the worktree
root, invoke `reproduce/launch.sh` in this directory with the following arguments
in order (existing attempts refuse overwrite):

```text
fetch --attempt 1 --requests experiments/2026-10-25_1628_codex_esp-response-source/reproduce/requests/metadata-4.json
fetch --attempt 2 --requests experiments/2026-10-25_1628_codex_esp-response-source/reproduce/requests/metadata-5.json
fetch --attempt 3 --requests experiments/2026-10-25_1628_codex_esp-response-source/reproduce/requests/metadata-6.json
fetch --attempt 4 --requests experiments/2026-10-25_1628_codex_esp-response-source/reproduce/requests/processed-1.json
verify --attempt 1 --requests experiments/2026-10-25_1628_codex_esp-response-source/reproduce/requests/verify-1.json
```

Per-stage launch snapshots, environment, receipts and logs are preserved. Raw
inputs live once under `data/esp_response_sources/2026-10-25_1628_codex_esp-response-source/`.
Article SHA256: `3b4834e44fe949675881e8aa1afd349d5dc8e6ffa46aee4d9bc9c88b48e7333f`.
Use a fresh artifact location and review absolute path changes for replay.
Python3.12.6/macOS arm64 CPU, requests2.34.2, psutil7.2.2, one numerical thread;
20min/stage,3GiB RSS and startup free+speculative,1GiB disk reserve plus32MiB margin,
50ms polling plus OS peak. No GPU/Modal, shared-environment changes or raw-MS downloads.

## Handoff

**Status:** bounded article-source screen closed; no predictive experiment.
**Next step:** qualify actual original calibration records if a working route or
local authorized copy is supplied; the existing PASS05890/Hitlist762 dependency
remains. Do not expand the same native response-only feature/optimizer search.
**Open questions:** raw measurements, molecular amounts/forms, acquisition and
selection/disposition metadata, and protocol transfer. The optional Panorama/local
file access question is still unanswered. This screen does not add a duplicate
Hitlist comment: its existing requested fields and next action are unchanged.
The full APC availability/concentration objective remains active and unachieved.
