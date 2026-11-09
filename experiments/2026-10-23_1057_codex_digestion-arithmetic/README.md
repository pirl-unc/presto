# All-substrate digestion arithmetic replay

**All17 deposited conditions reproduce numerically. Four conditions also reveal source-inserted time-zero baselines, which must not be treated as observations.** Agent/model: Codex / GPT-6. Registered and executed2026-10-23; local CPU only, no Modal/GPU.

## Source and exact scope

Immutable source `/private/tmp/hitlist-v2-digestion-208-candidate`, manifest SHA256 `b3cfffc444f922803625735bd1e712a2f734513fc4bd8e83f8fd83759f55410c`; original Soh2024 MSV000092631/PXD044463 processed archive, workbook and constructs. This family consumes original nested CSVs plus independently decoded original RData layers, without importing Hitlist. Prior family1018 independently checked source typing. The reference aSPIRE1.0 code is pinned to `c3d6c9112b2fd88d7e9453b1f23dc338f52c172b`; it is not asserted to be the deposition's exact producer.

Recompute distinct precursor rows, original assignment joins, missing area/background treatment, raw peptide/run sums, reported technical and biological means, I/L aggregation, technical3/4→1/2 mapping, baseline maximum/global median, background subtraction, forced time0 zeros, later missing/censored values and technical averaging. Compare every final slot, including repeated times and missing values. The original synthesis-error/search filtering itself is not independently validated. RF1 raw-time arithmetic can match while its original workbook/quantity conflict remains unresolved.

## Outcomes

| Check | Result |
|---|---:|
| Conditions passing |17/17|
| Raw quantities replayed |545,664|
| Distinct precursor rows |1,187,416|
| Final vector slots |280,664|
| Source missing slots / zeros |4,782 /62,860|
| Maximum relative raw error |5.063e-15|
| Maximum relative technical / biological mean error |9.298e-15 /7.394e-15|
| Maximum relative final error |5.066e-15|
| Original source mock-zero rows |10,746|

[Every condition](results/comparison.csv), [complete metrics](results/conditions.json), and an empty gzip discrepancy ledger are preserved. Mock baselines occur in CaM(b1,b2;1,424 rows), IF2(b1;3,370), UBE2S(2;1,396), and RF1(b1,b2;4,556). Their filtered layers lack time0 for the named biological labels. This does not establish that all upstream raw zero-control acquisitions are absent. Final forced zeros do not independently measure zero peptide amount.

The closure verifier matches the separately implemented prior H4 invariants and detects a1% in-memory perturbation of one positive raw quantity and a2× perturbation of one positive final value: one raw and one final mismatch, plus four technical/eight biological mean mismatches. Original files remain unchanged. This sensitivity check supplements, rather than substitutes for, original-source replay.

## Reproduction and decision

`reproduce/launch.sh` freezes the actual command; launch.json pins Presto5a10094, dirty state, source/algorithm hashes and one-thread environment. Source snapshots and verification command/hash are retained. Runtime58.247s; peakRSS1,430,355,968bytes.32 digestion source files rehashed here; prior family1018 verified the whole250-file generation. No training, pretraining, synthetic biological labels, losses or assay-head mapping; no predictive train/validation/test split because this is a source integrity audit. IC50/binding metrics and prediction dumps do not apply.

Next use a shared source-linked contiguous recipe that excludes mock-baseline dependence and RF1 conflict, preserves all dispositions and admits only repeated positive temporal support. Passing arithmetic is not biological predictive quality or calibrated concentration. Code, original logs, every condition metric and mutation ledger remain in this directory; large original assets remain in the pinned shared package.
