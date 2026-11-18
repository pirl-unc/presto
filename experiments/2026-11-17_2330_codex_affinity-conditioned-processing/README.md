# Affinity-conditioned shared processing

2026-11-17–18 UTC · Codex / GPT-6 · **Complete: six fits, thirteen independent checkpoint evaluations and five input-control suites verified.** TEST outcomes remain sealed.

The experiment improves both-class matching and trains the processing contribution through the canonical Presto trainer. The prior general-purpose checkpoint is retained. The new selected boundary model is a provisional processing candidate: pooled matched ranking improves in both classes, but class-II paired ordering remains uncertain. See [the complete decision](DECISION.md), [CPU usage](USAGE.md), and [architecture](ARCHITECTURE.md).

## Predictive results and decision

All six H100 fits, twelve selected/terminal evaluations plus parent rescoring, and three integration fits are complete. Both selected boundary-model seeds pass the ordinary-presentation and measured-affinity guards. These guards allow a small point-estimate decline; they are not noninferiority proofs.

| Metric | Parent | Boundary model, selected seed 42 | Selected seed 43 |
|---|---:|---:|---:|
| Ordinary class-I AUROC | .8759 | .8709 | .8718 |
| Ordinary class-II AUROC | .8373 | .8373 | .8417 |
| Rank-matched class-I AUROC | .6263 | .6437 | .6431 |
| Rank-matched class-II AUROC | .5482 | .5605 | .5607 |
| Rank-matched class-I pair wins | .6890 | .7129 | .6986 |
| Rank-matched class-II pair wins | .5839 | .5839 | .5912 |

Class-I AUROC gains have positive paired protein/condition intervals. Class-II condition-cluster intervals cross zero for both selected replicates; its eight condition clusters and lack of a convincing pair-win gain limit the claim. Four paired/terminal checkpoints fail the ordinary class-I retention guard. The complete 13-model tables, all 24 paired contrasts, 156 retention checks, stricter-caliper/allele sensitivities and decomposition readouts are preserved in [analysis](results/analysis/metrics.csv) and [decision](DECISION.md).

![Both-class common-cohort AUROC](results/figures/presentation-auroc.png)

[Within-pair ordering figure](results/figures/matched-pair-wins.png) · [Processing-supply figure](results/figures/supply-pair-wins.png). PNG and PDF versions are retained and visually checked. Every panel compares identical rows; cohorts differ between panels.

Independent verification reproduces 1,066 numeric checks across 78 model/cohort combinations. All six closed fits preserve source/scaler/draw-order contracts, admit complete TRAIN auxiliary pairs, and have no frozen parameters. Core tests: 202 passed; analysis/control unit tests: six passed. Both default and provisional candidate pass label-free CPU class-I/class-II presentation and IC50 inference. See `results/fit-verification.json`, `results/evaluation-verification.json`, and the inference receipts.

Full-fit per-worker wall times range from 1,930 to 3,311 seconds (sum 15,384.87 seconds); independent selected/terminal/parent scoring totals 2,175.95 seconds. These are summed worker durations, not elapsed family wall time. Exact fit/evaluation/control timing and actual hardware are in `results/runtime-summary.json`; external CPU scorer timings are separately recorded in `results/scoring-runtime-summary.json`.

The user-requested Hitlist 2.1.12 refresh is a separate generation: [refresh record](../../tasks/hitlist-2.1.12-refresh/README.md). Both broad public MS exports have been regenerated through its loaders, preserving evidence roles and existing indexes. Their old common fields are unchanged, and refreshed serotype fields are added. No dataset/version labels in this family were rewritten.

## What is fixed and verified

- Core commit `0c21bc4da1066ebfadf713dca478d655c31d4b8b`, branch `codex/affinity-conditioned-processing-20261117`; clean implementation frozen under `reproduce/runtime/` with231-file manifest.202 relevant tests pass; actual parent expansion leaves all existing parameters and four CPU predictions bitwise unchanged. Local Torch2.15.0; Modal Torch2.14.1 pinned separately.
- Native Hitlist2.1.9/43e22662 and GENCODE39 sources unchanged:209695 class-I and178179 class-II positive/background pairs,159/42studies. Both flanks known for170339I and149939II positives. No hidden study or positive cap.
- All375896 predictor-length-eligible positives considered. Same-protein/same-length backgrounds reuse already protected native TRAIN peptide identities. Initial up-to8 candidates, then up-to32 where unmatched;1,443,325 initial and138,306 adaptive allele/peptide queries,282+249 successful CPU jobs.
- The canonical matcher fixes the biological MHC candidate set, requires the same best BA-rank candidate, and constrains **every candidate** to≤0.5BA percentile points **and**≤0.3log10(rank+0.01). The floor handles printed zeros. This is not equivalent to0.1log10(nM), which means a maximum1.259-fold nM ratio. EL scores never select labels or matching acceptance.
- Primary policy chosen on cached TRAIN balance/coverage. Flat0.5pp all-candidate matching left class-I BA-only AUROC.5714; the hybrid reduced it to.5086. See [matching audit](results/MATCHING_AUDIT.md). Near.5 pooled AUROC does not establish exact within-pair balance.
- Final conditional TRAIN has**3777I pairs across76studies** and**9114II pairs across28studies**. Most are monoallelic (3774I/9107II). BA-only AUROC.5089I/.5020II; BA pair win.5743I/.5558II, a remaining limitation. All12891 pairs pass source-coordinate/context, typed-MHC, condition, lineage and matching checks. Native ordinary training remains intact.
- NetMHCpan4.2c and NetMHCIIpan4.3e are pinned, not represented as latest releases.800 fresh predictions across4old-cache alleles/class reproduce BA/EL values exactly; current package content manifests and private archive hash are retained. Predictor training independence is not established by this parity check.

## Actual APC support in the matched TRAIN subset

The [condition census](results/PROCESSING_CONTEXT.md) links every admitted factor to its Hitlist PMID and pair count. Class I includes ERAP1 knockdown (22 present / 22 explicitly absent), ERAP2 knockout (43 present), IFN-γ (55 present), and one flucloxacillin pair. Class II includes HLA-DM transfection (3,370 present without an attributed absent arm), TAX1BP1 knockdown (28 present / 43 absent), CIITA transduction (469 present), and very small IFN-γ/drug subsets (6 IFN-γ, 5 decitabine, 4 tazemetostat; combinations overlap). Cell type is known for 3,471 I and 7,156 II pairs. Both flanks are known for 3,296 I and 8,034 II pairs. Quantitative expression annotations are absent.

These are annotation counts in peptide/background pairs under the same condition, not isolated intervention/control comparisons. Class-specific pathway masks remain active. All ordinary training and its broader condition support are retained.

## Training and output contract

All supported parameters remain trainable. Ordinary replay is64rows/update:50%measured concentration,25%class-I presentation,25%class-II presentation, with square-root study mass for presentation. Literal concentration endpoints/scalers/censor policies are inherited unchanged. MS positives remain source observations; backgrounds remain sampled unobserved peptides, never experimentally measured nonbinders.

Matched training adds4complete pairs/class/update at loss weight0.5. The loss is mean softplus of negative positive-minus-background **processing-supply** score. That score includes APC supply and the optional junction residual; it excludes binding, register selection and MS observation terms. Aux backgrounds refer to existing positive IDs, appear once, and are excluded from ordinary replay and ordinary training diagnostics. Pair IDs/draws and RNG are persisted for exact resume. Shared encoder updates can still alter measured affinity, which must be checked empirically.

The optional junction module reads six ordered residues on each side of each N/C boundary, using existing per-residue encoder states, class/side/terminus flags and the shared64-dimensional APC state. A single MLP serves both boundaries/classes, adds207360parameters, and has a zero-initialized scalar readout. Missing exterior context contributes zero; a true known protein end is distinct. Its corrections add to the actual processing term before presentation aggregation. Neither this score nor observational APC effects identify physical cleavage rates, absolute presentation yield, or causal knockout/drug effects.

## Registered comparison

| Arm | Ordinary replay | Matched supply loss | Junction module |
|---|---|---|---|
| continued | yes | no | no |
| paired | yes | yes | no |
| paired-junction | yes | yes | yes |

Common parent checkpointSHA`1b6011396a563f52ea1d6da4bb1d4e69b8dbf0103148ba829225581c4d312f68`. Seeds42/43,2000updates,AdamWlr3e-5/weight-decay1e-4,warmup100/cosinefloor0.1. Validation every500updates, selection by equal-class mean of equal-study presentation losses; separately evaluate fixed terminal checkpoints. Full-data16-update diagnostics precede predictive fits. Requested hardwareH100!; parent rescoring observed NVIDIAH10080GB HBM3/85,009,367,040bytes.

## Evaluation and leakage contract

Old benchmarks remain unchanged:340matched-I/158matched-II pairs, old APC/broad validation, new protected ordinary validation, and measured concentration validation. The new hybrid-rank cohort is separately named and hashed: **209 class-I pairs from 4 studies and 137 class-II pairs from 3 studies**. All retained validation pairs are monoallelic. Its largest study supplies 125/209 I pairs and 71/137 II pairs; this is a protein/condition validation cohort, not a held-out-study or multi-allelic confirmation. The current parent reproduces old matched AUROC.614152I/.569220II and scores.626336I/.548244II on the new cohort. This difference is a population change, not a model gain.

Broader preparation and its independent audit embargoed **all** historical validation candidates, not only the previously selected pairs. New auxiliary backgrounds are a subset of those already protected TRAIN identities. Exact/shared-core8/Hamming1 protections and whole source-family split groups remain in force. TEST outcomes are intentionally not evaluated during model development; existing identity-only exclusions persist. Historical fixed APC/broad/matched validation had zero original-affinity-ancestor exposure;346original affinity-validation rows had exposure, so report the existing ancestry-screened sensitivity as well.

Both-class ordinary and matched AUROC/AUPRC/pair wins, measured IC50 Spearman/Pearson/log10RMSE/500nM threshold metrics, paired protein-family and condition bootstraps, supply/junction decomposition and APC/flank controls are required for closure. Checkpoint/seed selection uncertainty is not captured by conditional bootstrap intervals. Operational affinity/ordinary retention guards are in the [specification](reproduce/spec.md).

## Reproduce and artifacts

- [Specification](reproduce/spec.md), [launch contract](reproduce/launch.json), [source/runtime manifest](reproduce/runtime-manifest.json), [literature notes](reproduce/literature-notes.md).
- Private licensed predictor archive: `artifacts/affinity-conditioned-processing/external-tools.tar.gz`; excluded from source publication. Pinned local packages remain unchanged.
- Modal volume `presto-checkpoints`, root `/volume/2026-11-17_2330_codex_affinity-conditioned-processing/`; raw source/caches/proposals/pair tables/fits/evaluations remain there. Prior source population `/volume/2026-11-17_1155_codex_shared-presentation-expansion/inputs/`.
- Local compact artifacts under `results/`, per-example validation exports under `results/evaluation/`, exact calls/source snapshots under `reproduce/`, logs under `launch_logs/`. No TEST prediction dump is produced because TEST is sealed.

Transport failures are retained: initial directory upload omitted required Linux data symlinks and was stopped/tasks0; archive/probes fixed it. First validation export encountered ambiguous flank residues; canonical unknown masking fixed it with unchanged pair counts, preserving the partial export. Local combined tests initially exhausted temporary disk; successful-fixture cleanup allowed the full202-test pass. None of these attempts is a predictive comparison.

## Completed input controls

Five checkpoints (parent, paired seed-43 selected, both selected boundary replicates, and boundary seed-42 terminal) each have original/APC-unknown/APC-shuffled/flank-unknown/flank-shuffled predictions on four fixed cohorts. All 147,350 control records pass identity, label, MHC and original-score reproduction checks; all 20 paired control contrasts are retained. See [control summary](results/analysis/input-control-summary.csv).

For the selected seed-42 boundary model, hiding APC state lowers ordinary class-II AUROC from .8373 to .8185; within-study APC shuffle gives .8359. Hiding flanks lowers new matched class-II AUROC from .5605 to .5515 and pair wins from .5839 to .5401. However, role-preserving paired flank shuffle gives AUROC .5628 and the same .5839 win rate. Thus input use is verified, while improved peptide-specific flank ordering is not established for class II. These inference controls neither identify isolated interventions nor replace trained ablations.

Lifecycle: all 22 listed experiment apps stopped, zero tasks. `results/modal-lifecycle-verification.json` records the check. An idle failed startup audit app was explicitly stopped during closure.
