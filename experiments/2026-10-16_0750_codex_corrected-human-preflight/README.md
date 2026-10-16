# Corrected human corpus preflight

Agent/model: Codex (GPT-6). Prospective data-only qualification of the corrected
[Hitlist human cohort](https://github.com/pirl-unc/hitlist/releases/tag/human-cohort-corrected-v1),
native manifest `b180eba804532f633a6d61bb350584dbe24f8df4b4b96f02c6ec492c60144372`.
The fixed contract is in [the plan](../agents/codex/plans/corrected-human-preflight.md).

Released Presto **0.1.13** and Hitlist **1.78.0** are the installed producers.
This recipe PR versions the repository as 0.1.14; that is not the experimental
producer. [Producer pins](reproduce/producer.json) bind the public wheel and main
commit. The launcher refuses dirty recipes and records the exact recipe commit.

The source contains 4,071,463 human-source/HLA observations. Both MHC classes are
eligible. The new cohort uses explicit reservoir selection (binding/elution caps
5,000 each, kinetics/stability 1,000 each, seed 42) and relative_deficit_v2 whole
peptide-component 80/10/10 allocation with split seed 42. Original baseline inputs,
membership and results remain immutable; this changes both source and policy, so
it is not a causal comparison of individual corrections. Native assay endpoints,
units, censor bounds, source IDs and contributor lineage remain preserved. Strict
MHC sequence resolution is enabled. Only the exact existing mhcseqs 2.6.13 catalog
is reused through its unchanged datacache recipe and explicit four-file hashes;
no prior cohort or partition assignment is reused. All shared products use datacache.

No pretraining, fitting, model constructor, optimizer, predictions or performance
metrics are allowed in this family. Train/validation/test membership and full output
evidence coverage will be measured. All augmentation/generated-negative ratios and
regularization weights are zero. Positive elution has no observed negative class;
unknown origins, donors, specimens and acquisitions do not establish independence.

[candidate-support.json](reproduce/candidate-support.json) fixes a prospective IC50
requirement before launch: direct binding:IC50 evidence from IEDB/CEDAR, at least
50 observations, 50 distinct peptides and 50 exact observations in every split.
The canonical preflight remains undeclared-output reporting. A later manifest may
bind these fixed claims to actual membership, but it is not an executed gate or a
trained-output claim. Thresholds will not be relaxed after seeing counts.

Reproduce from the recorded clean recipe commit with
`bash experiments/2026-10-16_0750_codex_corrected-human-preflight/reproduce/launch.sh`
from the repository root. The launcher
fetches the released producer through datacache. CPU: eight, RAM: 64 GiB, scratch:
512 GiB, GPU: explicitly none, Torch/BLAS threads: one, loader workers: zero,
function/work limits: 50/45 minutes. Runtime, runner-only RSS, all logs, pip check,
versions, catalog, membership, funnel, coverage and leakage evidence are retained.

A first local prelaunch used the tooling interpreter, which lacked filelock. The
wheel fetch failed; the hash guard then rejected the empty path before a remote
app ran. The unchanged log is [preserved](results/prelaunch-failure.log). #107 fixes
the shell wrapper to propagate a fetch failure before invoking Modal. The actual
retry uses the shared-environment interpreter; no shared packages were modified.

## Actual result and fixed support decision

The run succeeded from clean recipe commit
`e2e1e827f7b677f7c4f4d37dfc4ca9f386c58ffb`, preserved at
`experiment-corrected-human-preflight-actual-20261016`. Modal app:
`ap-abhFjm8vPCTQjJqnvNooK3`. Runtime: **985.768 seconds**; peak runner RSS:
**11,749,728,256 bytes**, excluding children. GPU remained none. All 48 run archive
members independently match their inventory; no checkpoint or prediction file
exists. The exact four-file MHC catalog was reused and hash-verified.

The source funnel admitted 13,190 binding / 3,909,515 elution / 19 kinetics / 741
stability records before caps. The actual cap mode is **reservoir**, independent
Algorithm R streams per family, each seeded 59 (data seed 42 plus the canonical
source offset). Caps retain 5,000/5,000/19/741; strict MHC filtering leaves
**4,492/3,856/19/686**, or **9,053** observations/model samples. There are no generated
samples, missing observation IDs or records lost between roots and model membership.
MHC sequence coverage reports 17,754 resolved allele occurrences, not 17,754 samples.

Whole-component allocation is explicitly `relative_deficit_v2`, seed 42,
80/10/10 requested fractions. Frozen membership:

| Split | Observations | Direct IC50 observations | IC50 peptides | IC50 exact | IC50 right-censored | IC50 class I / II |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 7,241 | 2,074 | 1,031 | 1,599 | 475 | 877 / 1,197 |
| Validation | 906 | 265 | 126 | 210 | 55 | 99 / 166 |
| Test | 906 | 275 | 126 | 214 | 61 | 101 / 174 |

The fixed candidate minima pass without adjustment. `assays.IC50_nM` is in
**log10(nM)** and admits only direct `binding:IC50` from IEDB/CEDAR for this claim.
No left-censored IC50 observations occur in this cohort. The frozen loader policy
treats absent inequality as exact and preserves its original empty value. Qualifier
-1 means `<` (left censoring / upper bound), +1 means `>` (right censoring / lower
bound). Legacy `split_support.csv` names these directional counts `censored_lower`
and `censored_upper`; consult the explicit inequalities or canonical output census
rather than interpreting those legacy names as bound types.

All **221 canonical output declarations remain undeclared** in the executed
preflight. [Candidate counts](results/candidate-support-counts.json) are an evidence
screen, not an executed manifest gate. The separately pinned
[prospective manifest](results/prospective-supported-outputs.json) binds the fixed
claim to actual model/data/membership fingerprints for the **next** fit. Its SHA-256
is `13fe8bd4dab158bd06ae31fc7996fba98f73b3944fa68c3f57463beec34e7cb0`.
That fit must pass the unchanged canonical gate before model construction.

## Inspect observations and limits

Every selected observation joins one-to-one to the verified native source, with
identical peptide. Public `selected-source-contexts.tsv` carries all 9,053 roots,
partition, citable reference and assay IDs, cell line/type, MHC sets/genotype and
restriction basis, conditions/attribution, measured values/units/inequalities and
lineage status. Join model-resolved MHC sets from `membership.tsv` using
`evidence_row_id`. Start with [context counts](results/source-context-counts.tsv)
and [the cited preview](results/source-context-preview.tsv); the preview is the
first three rows per split/evidence-kind/MHC-class after the documented sort,
not a representative sample. Full original and resolved allele sets remain in
the archive's frozen membership, alongside assignments and parent graph.

Peptide, observation, pMHC and known experimental-origin overlap are zero across
splits. Shared studies are **159 train/validation, 153 train/test, 109
validation/test**. Known experimental origins cover only 78/6/3 train/validation/test
observations; 8,966 of 9,053 remain unknown. Every donor, specimen and acquisition
identity is unknown. Zero observed reuse on an unknown axis does not establish
independence. pMHC identity is unresolved for 1,449/179/174 observations and study
identity for 1,044/120/123; those unknowns also remain explicit.

Direct KD has 375/35/33 observations but only 144/6/5 exact values. Off-rate has
19/0/0; no held-out kinetics support is established. Elution is positive-only
(3,088/394/374); no measured processing or T-cell/TCR response records are present.
Human-source foreignness is one-class. The old allocation-only audit used different
selected roots; its task counts do not transfer here. No all-output claim follows.

## Artifacts and reproducibility

[Corrected preflight release](https://github.com/pirl-unc/presto/releases/tag/corrected-human-preflight-v1)
contains 23 public assets, each independently downloaded and hash/size-verified
through datacache. Public verification also checks every one of the 48 archive
members, all 9,053 source/partition joins, fixed IC50 counts and future-manifest
fingerprints. [Verification receipt](results/public-verification.json). `run.tar.gz` is 7,308,268 bytes, SHA-256
`ce8d55cf1e7a14fb898f997918133f708526d145c65f01f97c37e2b0df6db0c8`;
`inventory.json` is 6,870 bytes, SHA-256
`a249f213f4c61c70eb3c139f6fbe0ffeb9e03a937266f898a0b6230c87bfdf36`.
The complete output census, regularization/config, source manifest, versions/pip
check, original log, exact assignments, model membership, parent graph and MHC
catalog/reports are inside. Standalone summaries, native source-context tables,
future manifest, recipes and inventories accompany it. All acquisition, analysis
materialization and publication use datacache.

Membership semantic SHA-256:
`12ed561ae2c8e670290b8dd16826c9834aa7b40e7dfff7132d254588c550fbd2`.
Assignment SHA-256:
`07d4744c10d7f5d825880bd0bdf3efcb47bd97e8d8725c623916d84b71bbdf1b`.
Parent graph SHA-256:
`2dba2460212ed052dd5ec586ce7bef6c469472833ae0cb819f9c9c99a976c450`.

Local invocation used `PYTHONPATH=/private/tmp/hitlist175-wheel-import` and PATH
with `/Users/iskander/code/shared-virtual-env/bin` before the isolated tooling
bin, then the exact reproduce/launch.sh. These are local orchestration overrides;
the remote installed producers and dependency receipt are explicit. No shared
environment packages were changed. Analysis recipe and public metadata joins were
written after launch and do not change the executed recipe or candidate thresholds.

Format/lint/strict docs passed. The full local suite passed **2,276 tests with four
explicit skips**; the actual-recipe CI passed 2,277 tests with three explicit skips, strict docs,
distribution validation and installed-wheel checks.

Status: actual preflight, registry and independent public artifact verification
complete. Final PR checks, merge and clean-main 0.1.14 distribution remain.
Next Step: publish and verify this family, then run the separately prospective
[bounded IC50 fit](../agents/codex/plans/corrected-human-fit.md).
Open Questions: predictive quality and evidence-backed biological independence.
#48/#53 and Hitlist #678/#680 remain open.
