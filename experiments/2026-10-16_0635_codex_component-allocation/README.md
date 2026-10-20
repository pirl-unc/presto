# Frozen component-allocation audit

Completed audit and independently verified public artifacts; software release checks remain. Agent/model:
Codex (GPT-6). [Prospective contract](../agents/codex/plans/component-allocation.md).

The label-blind relative-deficit policy reduces the measured support imbalance
without splitting components or changing source observations. Direct KD now has
406/71/33 train/validation/test observations, versus 506/0/4 originally. Both policies
produce exactly 7,200/900/900 total roots. This is a proposed partition audit, not a
model fit or training-readiness demonstration.

## Reproducible contract

Clean actual source `83431016f30f9bae1ba47601ce4b2346db28cd52`, tag
`experiment-component-allocation-20261016`; Presto 0.1.13. Original source is the
[public human preflight](https://github.com/pirl-unc/presto/releases/tag/human-corpus-preflight-v1),
archive SHA-256 `444ac92e859da9afd3fcdf3c27ee3ae282f70a6134ac9c42f3c528cdcc1dae80`.
Its native Hitlist 1.76 bundle is pinned by manifest
`039efc7854098951b5c10b3fb5b38294452eeb425e367ed1fafdeebddf34b38d`.
Curation fixes in subsequent producers do not retroactively change these inputs.

Exactly 9,000 original selected observations; same source admission, historical
seeded uniform-list caps, strict MHC filtering and source labels. Both MHC classes,
human-reported peptide source. No re-cap/re-filter/model/pretraining/optimizer or
predictive outcome. All original endpoints, units, censoring, cell context, source
citations and MHC roles remain in observation_contexts.tsv and the source bundle.
No synthetic controls or new targets. Affinity/kinetics/stability labels remain
measured; elution positives remain presentation evidence, not detectability labels.

The original 5,843 intact components (maximum 31 observations) are recovered from
verified assignments. V1 exactly reproduces every original assignment before V2
runs. Both sort components by descending size and seeded digest; seed 42, fractions
80/10/10. V1 chooses largest absolute remaining row capacity. V2 chooses largest
remaining fraction of each partition's requested capacity. The nonempty-partition
guard and deterministic tie order remain. No labels or assay types enter either
allocator. 3,060 roots move partitions under V2; every component remains intact.

Run `bash reproduce/launch.sh --cache-dir /shared/datacache/path` from the original
tag. Optional `--local-inputs paths.json` reuses existing datacache-managed inputs;
all URLs/hashes/sizes are pinned in reproduce/inputs.json. The launcher imports
exact checkout bytes via a temporary package link without changing the environment.
Runtime 7.35 s, peak RSS 769,818,624 bytes on local macOS CPU, no GPU; source scans bound
to 65,536 rows. Hitlist 1.78.0 audit implementation, datacache 1.27.1, pandas 2.3.3,
numpy 2.5.3, pyarrow 25.0.1, Python 3.12.6; complete source/environment receipt retained.
Inputs and reusable outputs use shared datacache.

## Actual support

| Evidence | Original train/val/test | Relative-deficit train/val/test |
| --- | ---: | ---: |
| All observations | 7,200 / 900 / 900 | 7,200 / 900 / 900 |
| Direct KD | 506 / 0 / 4 | 406 / 71 / 33 |
| IC50 family including KD(~IC50) proxy | 3,390 / 166 / 174 | 2,996 / 340 / 394 |
| EC50 family including KD(~EC50) proxy | 225 / 11 / 14 | 210 / 13 / 27 |
| Kinetics (off rate) | 19 / 0 / 0 | 15 / 1 / 3 |
| Stability (half-life plus temperature) | 613 / 44 / 29 | 570 / 57 / 59 |
| Elution positives | 2,447 / 679 / 679 | 3,003 / 418 / 384 |
| Elution negatives | 0 / 0 / 0 | 0 / 0 / 0 |

`endpoint_support.tsv` separates exact original endpoints/proxy endpoints, units
and censoring; `class_support.tsv` retains both MHC classes and evidence families.
Missing endpoint labels on supplement rows remain missing, not inferred from an
assay-group label. No thresholds or strata were changed after seeing results.

Both root audits pass peptide-disjoint policy with zero cross-partition peptides,
observation IDs, pMHC keys or known experimental origins. Studies still overlap:
V2 train/test 143, train/val 131, val/test 93 (original 161/162/117). All 9,000 donor,
specimen and acquisition identities remain unknown. Only 79 origins are resolved;
V2 distributes 54/24/1 resolved observations across train/val/test. Lower study
overlap is descriptive, not evidence of study independence. The same reviewed
registry limitations apply to both policies.

Decision: prefer V2 for future canonical Hitlist allocations because it reduces
systematic component-size concentration. Keep Python preparation and scoped-search
V1 defaults explicit; retain pinned external assignments and historical results.
V2 does not promise every target adequate evidence: kinetics still has only one
validation observation, elution has no negatives, and biological independence is
unresolved. No fitted model, predictions, validation/test performance, regression
or 500 nM classification metrics exist in this audit. #48/#53 and Hitlist #678/#680
remain open; the corrected corpus needs its own prospective qualification.

## Download and inspect

[Public artifact set](https://github.com/pirl-unc/presto/releases/tag/component-allocation-audit-v1)
contains both original/proposed assignment TSV/parquet, observation context,
per-assay/endpoint/class/censor counts, component sizes, root leakage audits,
versioned policy/source pins and runtime receipt. The archive is 3,727,118 bytes,
SHA-256 `f7a66f2a87e0d8b343730240c41f94e61cd649096f30c861504d102fb9d3994d`.
All 17 members and complete component/observation identity invariants verify.

Start with support-comparison.tsv and qualification-summary.json. Original counts
and unknowns are retained alongside the new policy; assignment files are proposed
root membership, not evidence of a completed new canonical fit. Recompute small
reports with `analysis/summarize.py /path/to/downloaded/run-files`. Public inventories
and SHA256SUMS pin each payload; results/audit.json retains original shared paths.

Independent public retrieval verified all 12 asset hashes and sizes through
datacache. `results/public-verification.json` records durable URLs and pins;
re-running the analyzer against those downloaded bytes verifies all 17 archived
files and the unchanged observation/component identities. The full software
suite passes 2,276 tests with four explicit skips (525.13 seconds).
