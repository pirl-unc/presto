# Canonical affinity target-policy audit

Status: completed label audit. Agent: Codex (GPT-6). This family checks the candidate Presto
0.1.16 census against the published 2,614 corrected-human IC50 observations.
See [the prospective plan](../agents/codex/plans/affinity-target-policy.md).

The canonical affinity policy is `log10(clip(nM, 0.001, 50000))` with unchanged
censor direction. Original values remain original measurements/bounds. This
family constructs label-only samples, retaining identity, class, allele sets and
source strings, without MHC sequences, a model, generated data or optimization.
It does not qualify full model inputs, output performance or a supported-output
manifest. Validation and test rows are audited but have no predictions because
no model is constructed. Donor/specimen/acquisition identities remain unresolved.

The input table preserves citations, assay types, cell context and conditions:
[public original/effective target audit](https://github.com/pirl-unc/presto/releases/tag/corrected-human-target-policy-audit-v1).
The [source preflight](https://github.com/pirl-unc/presto/releases/tag/corrected-human-preflight-v1)
retains the complete 9,053-row cohort and partition/leakage evidence. No existing
artifact is changed. This audit compares direct IC50 observations, not independent
experiments. Repeated objective/panel exposures must not be added together.

Run `reproduce/launch.sh` from the clean recorded recipe revision. It uses local
CPU, one Torch/BLAS thread, no GPU/workers, batches of 128 and a ten-minute alarm.
Inputs and outputs use datacache under `~/.hitlist/presto-affinity-target-policy`.
The script records runtime and versions; results and download pins are retained
in this family. There are no fitted conditions, loss weights or baseline winners.

## Observed result

The canonical census matches all 2,614 effective targets and censor directions
from the independent released-0.1.13 audit. All 211 clipped source identities
match, including 54 exact values and 157 right-censored bounds; none are below
the minimum. Every clipped direct observation is traceable to its source ID.

| Partition | Observations | Exact | Clipped | Clipped exact | Clipped right-censored |
|---|---:|---:|---:|---:|---:|
| Train | 2,074 | 1,599 | 157 | 39 | 118 |
| Validation | 265 | 210 | 31 | 8 | 23 |
| Test | 275 | 214 | 23 | 7 | 16 |

Actual producer: clean candidate Presto 0.1.16 source revision
`ca9e883034a8842b99394deb11b8155abd83b597`, preserved by tag
`experiment-affinity-target-policy-20261016`. This is a candidate checkout, not
an already released wheel. The receipt records the imported module and all
analysis versions: Python 3.12.6, Torch 2.14.1, NumPy 2.5.3, pandas 2.3.3,
PyArrow 25.0.1, mhcseqs 2.6.12 and datacache 1.27.1. No MHC catalog or sequence
resolver was used. The local arm64 CPU audit used one thread, no GPU or workers,
and took 2.014 seconds including input acquisition, with 271,433,728 bytes peak
RSS. These are observed process metrics, not a throughput benchmark.

`affinity_target_audit.json` and `counts.tsv` enumerate applicable objective and
panel cells, including zeros. `affinity_target_changes.tsv` contains every
clipped objective exposure. `clipped-source-contexts.tsv` retains all 86 source
and derived audit fields for the 211 distinct direct observations, including
cell/condition fields and explicit unknown biological lineage. The complete
2,614-row input is pinned by URL/hash in `summary.json`; no field is inferred
from a missing cell. `receipt.json` binds every output and its datacache path.

The [downloadable audit](https://github.com/pirl-unc/presto/releases/tag/affinity-target-policy-audit-v1)
contains the outputs, producer receipt, recipe, plan and hashes. Use
`reproduce/launch.json` for the exact invocation and `reproduce/source/audit.py`
for executed source. A fresh reproduction uses the recorded clean revision and
an equivalent import environment; its absolute cache/checkout paths may differ.

Decision: the canonical clipping audit agrees with the frozen source evidence.
The policy can now be bound in a new prospective fitting family; this audit does
not approve the historical raw-target fitting plan or silently migrate its
manifest. Source values, partition membership and the stopped attempt remain
immutable. No output is declared scientifically ready by this result.
