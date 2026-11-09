# Shared contiguous digestion time-course admission

**The canonical Hitlist recipe admits30,566 peptide–biological-label curves with125,423 positive measurements. Independent source/target/group replay passes.** Only11 parent constructs survive: six training, three validation and two test. There is no biological fit or held-out temporal quality evaluation in this family.

Registered2026-10-23; agent/model Codex / GPT-6. Recipe `digestion_contiguous_timecourses_v1` through the existing EvidenceStore materialize/verify API. [Prospective contract](reproduce/plan.md).

## Data and eligibility

Original Soh2024 MSV000092631/PXD044463 purified human standard20S digestion,17 conditions/15 literal constructs. Input2.0.8 candidate manifest `b3cfffc444f922803625735bd1e712a2f734513fc4bd8e83f8fd83759f55410c`. Recipe output manifest `9d5e7cbd53a39c659d39dc8800413bce41414e39e50fd441f40413caa027830e`. All quantitative processing passes the independent17-condition family1057 replay.

Use source PCP/no splice alternative, canonical20, exactly one independently found I/L-equivalent span in the literal construct, source-selected final vectors, actual filtered time0 for every source technical label plus raw time0 agreeing with acquisition metadata. Exclude RF1 condition conflicts. Collapse alias/repeated-time contributions only when equal within frozen tolerance. Keep original slots, missing/zero states and exclusions. Never normalize quantities or turn missing values into negatives. Require at least three positive nonzero times and repeated biological-label curves sharing at least three such times. No MHC length/affinity filter; admitted source peptides happen to span7–30 residues. No support claim outside those observed lengths.

Before outcome selection, all source final peptides (including excluded/annotation bridges) and identical constructs define connected components. Sorted component hashes receive fixed round-robin fivefold assignments; fold0 test,1 validation,2–4 training. All Ffh ratios stay in one construct group. No salt search, repartition or response-based selection. There are15 source components; eligibility leaves11. These group exact products/constructs, not homology or independent studies.

| Partition | Parent constructs | Conditions | Curves | Positive targets |
|---|---:|---:|---:|---:|
| Train |6|7|23,054|91,226|
| Validation |3|3|3,832|15,037|
| Test |2|2|3,680|19,160|

All53,413 final source vectors form50,372 alias/biological-label groups;19,806 groups are excluded. Reasons overlap: missing actual baseline10,641; non-exclusive source contiguous claim7,094; nonunique/no literal contiguous span6,912; inadequate repeated common times2,880; unresolved condition time2,198; fewer than three positive times907; annotation-only206; conflicting alias/repeated-slot values1. [Full counts](results/recipe-summary.json).

## Verification and reviewed source migration

A verifier importing no Hitlist code independently checks all280,664 original slots, all50,372 curve decisions,125,423 targets, original zero evidence, exact spans, duplicate handling, contexts, graph components and splits. All five portable Parquet relations match their original source hashes. Admission50.335s/3.235GiB peakRSS; independent replay7.867s/2.001GiB. No model predictions or held-out response metrics are generated; this admission-only scope preserves holdouts for a prospective fit. No pretraining, synthetic labels, losses, MHC/TCR tasks or nM/binder metrics apply.

PR773 merged while this family ran. Its reviewed source package has manifest `ad62414d511b74a6aa1e4da457fdff68d9f4527874d1a2e385c65e5b98df41e8`. An immutable source archive of Hitlist3e0b999 (candidate2.0.9) admits that generation separately. The first compatibility check stopped because context fields_json embeds nine newly added source-census counts; original code/log/export remain. The explicit [metadata amendment](reproduce/reviewed-metadata-amendment.md) permits only those additional counts and the declared units/slot-status labels. Verification-only replay then confirms **every numeric target, membership, exclusion, mapping and partition is unchanged**. A separate preflight launcher-hash mistake is also preserved under attempts; it read no outcome data and changed no export.

Reviewed recipe manifest: `42cda75e98c3144c5b0226d24b5ebee7d17c687ad6559a6c94d7e6d51b13e33d`. New target unit `source_MS1_peak_area_arbitrary_units`, scale `aSPIRE_background_adjusted_biological_mean`; every source slot has explicit value status. All old columns in constructs/raw/filtered/final relations are identical. Compatibility verification17.586s/4.409GiB peakRSS. Source-review work appeared concurrently in the linked Hitlist worktree; this experiment therefore freezes a full immutable code archive and does not depend on its mutable checkout.

## Reproduction and limits

Actual commands, commits/dirty state, snapshots and source hashes live in reproduce/. Initial producer623afcb plus frozen recipe changes; reviewed producer3e0b999. Local CPU only, no Modal/GPU. Original and reviewed complete exports are under `artifacts/2026-10-23_1123_codex_digestion-timecourse-admission/{recipe,recipe-reviewed}`; portable typed sources remain inside them. Compressed derived exports and receipts are retained in results; raw source assets remain in the pinned shared packages. Source source-search/synthesis filtering remains unvalidated independently, and positive-product selection excludes unseen/censored trajectories. Technical numeric labels are not assured paired acquisition identities; source biological labels are not independent studies. Actual baseline support does not guarantee substrate-only controls. Purified enzyme origin is not APC machinery context.

35 focused source/recipe tests passed before reviewed metadata. On immutable3e0b999, the unchanged normal guarded runner passed3,350 tests/10warnings in176.08s with two workers (5.71GiB available); this excludes integration tests by the runner's declared default. It is not evidence for subsequent upstream edits. Source PR773 is merged; recipe publication/review/deployment remain distinct from this completed admission study. Shared request775 owns the recipe. Next check training-only temporal repeatability, then freeze a parsimonious sequence-to-cut-rate comparison with an explicit MS-response observation contract. Full APC concentration qualification remains open.

Later review conservation: `/private/tmp/hitlist-digestion-209-review-candidate`, manifest5f616170b5514cb4d8f0c7b7843d232959e9514eebbca6483f976e17b0846615, strengthens raw zero-count evidence. Its examples and components are byte-identical to this family; one already-excluded curve receives an additional baseline reason. No admitted target changes. Upstream review also computed descriptive validation/test repeatability, encountered after our train-only diagnostic. Those partitions are no longer untouched with respect to every response diagnostic; no kinetic model has been fitted or scored, and no split or model choice is changed using those summaries. The future comparison is exploratory same-study transfer, requiring external validation.
