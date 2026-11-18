# Processing decision and complete comparison

2026-11-18 UTC · Codex / GPT-6 · Core `0c21bc4da1066ebfadf713dca478d655c31d4b8b`.

**Retain the previous general-purpose checkpoint. Keep the new matching and paired-training infrastructure, and expose the boundary-aware selected checkpoint as a provisional processing candidate.** There is a replicated pooled-AUROC gain, but no convincing class-II paired-ordering gain and no uniformly superior model. This is a completed comparison, not a claim that processing or isolated interventions are solved.

The provisional candidate is `paired-junction-s42-n2000/raw/selected.pt`, selected at update 500 by equal-class, equal-study ordinary presentation loss. Its SHA256 is `e4898af3a7a442a999177273687ea4e4ce5d4216e908f2e435b2d73226e56963`. Seed 43 is the replicate. Both selected boundary checkpoints pass all 13 operational retention guards. Candidate choice remains a development decision; TEST is sealed. The stronger seed-42 terminal result is retained as a fixed-endpoint sensitivity, not substituted for the selected checkpoint.

The default remains the previous expanded-state-sqrt seed-42 checkpoint, SHA256 `1b6011396a563f52ea1d6da4bb1d4e69b8dbf0103148ba829225581c4d312f68`. [Usage](USAGE.md) supports both without changing the installed model or adding an external NetMHC inference dependency.

## New hybrid-rank benchmark

209 class-I and 137 class-II pairs, respectively 4 and 3 studies; all are monoallelic. Same source protein, peptide length, condition and biological MHC candidates. Every candidate satisfies the absolute 0.5-percentile-point and relative 0.3-log-rank calipers; BA only, no EL matching. Rows and matching policy are fixed across models. Ordinary validation has 390 I and 597 II positive/background pairs, a different population.

| Model | Ordinary I / II AUROC | Matched I / II AUROC | Matched I / II pair wins | Retention guards |
|---|---:|---:|---:|---|
| Parent | 0.8759 / 0.8373 | 0.6263 / 0.5482 | 0.6890 / 0.5839 | reference |
| continued, seed 42, selected | 0.8777 / 0.8436 | 0.6246 / 0.5476 | 0.6842 / 0.5620 | pass |
| continued, seed 42, terminal | 0.8769 / 0.8449 | 0.6279 / 0.5488 | 0.6890 / 0.5766 | pass |
| continued, seed 43, selected | 0.8761 / 0.8438 | 0.6229 / 0.5492 | 0.6842 / 0.5766 | pass |
| continued, seed 43, terminal | 0.8763 / 0.8444 | 0.6219 / 0.5472 | 0.6842 / 0.5839 | pass |
| paired, seed 42, selected | 0.8703 / 0.8358 | 0.6414 / 0.5581 | 0.7081 / 0.5985 | fails ordinary I |
| paired, seed 42, terminal | 0.8699 / 0.8371 | 0.6540 / 0.5644 | 0.7129 / 0.6204 | fails ordinary I |
| paired, seed 43, selected | 0.8723 / 0.8398 | 0.6417 / 0.5579 | 0.6890 / 0.5985 | pass |
| paired, seed 43, terminal | 0.8704 / 0.8362 | 0.6491 / 0.5667 | 0.7321 / 0.6204 | fails ordinary I |
| paired-junction, seed 42, selected | 0.8709 / 0.8373 | 0.6437 / 0.5605 | 0.7129 / 0.5839 | pass |
| paired-junction, seed 42, terminal | 0.8723 / 0.8417 | 0.6616 / 0.5722 | 0.7321 / 0.5839 | pass |
| paired-junction, seed 43, selected | 0.8718 / 0.8417 | 0.6431 / 0.5607 | 0.6986 / 0.5912 | pass |
| paired-junction, seed 43, terminal | 0.8707 / 0.8420 | 0.6552 / 0.5741 | 0.7321 / 0.5693 | fails ordinary I |

Both selected boundary replicates improve new matched AUROC from .6263 to .6437/.6431 (I), and .5482 to .5605/.5607 (II). Ordinary class I declines by .00494/.00410, while class II is retained or improves. These modest class-I declines pass the prespecified .005 point guard; that is not a statistical noninferiority result.

Class-I pair wins increase from .6890 to .7129/.6986. Class-II pair wins are .5839/.5912 versus .5839 for the parent. At the fixed terminal endpoint the boundary module gives larger AUROC gains but class-II pair wins are unchanged/decreased. Four checkpoints fail the ordinary class-I guard, as shown; none fails an affinity guard. Increasing matched-loss training is therefore not a free improvement.

## Unchanged historical nM-matched benchmark

340 I and 158 II pairs. This is the benchmark where the previous broader-training comparison regressed from .6342 to .6142 for class I. It is not Claude's original 10,998-pair cohort, and its results cannot be compared numerically to that report. The new matcher changes which pairs qualify; parent .6142 versus .6263 across the old/new benchmarks is a population difference, not a model gain.

| Model | Ordinary I / II AUROC | Matched I / II AUROC | Matched I / II pair wins | Retention guards |
|---|---:|---:|---:|---|
| Parent | 0.8759 / 0.8373 | 0.6142 / 0.5692 | 0.6588 / 0.6076 | reference |
| continued, seed 42, selected | 0.8777 / 0.8436 | 0.6141 / 0.5691 | 0.6618 / 0.5886 | pass |
| continued, seed 42, terminal | 0.8769 / 0.8449 | 0.6156 / 0.5701 | 0.6588 / 0.6013 | pass |
| continued, seed 43, selected | 0.8761 / 0.8438 | 0.6156 / 0.5711 | 0.6647 / 0.6013 | pass |
| continued, seed 43, terminal | 0.8763 / 0.8444 | 0.6150 / 0.5685 | 0.6618 / 0.6076 | pass |
| paired, seed 42, selected | 0.8703 / 0.8358 | 0.6279 / 0.5770 | 0.6794 / 0.6139 | fails ordinary I |
| paired, seed 42, terminal | 0.8699 / 0.8371 | 0.6346 / 0.5822 | 0.7088 / 0.6266 | fails ordinary I |
| paired, seed 43, selected | 0.8723 / 0.8398 | 0.6288 / 0.5765 | 0.6794 / 0.6139 | pass |
| paired, seed 43, terminal | 0.8704 / 0.8362 | 0.6320 / 0.5843 | 0.7029 / 0.6329 | fails ordinary I |
| paired-junction, seed 42, selected | 0.8709 / 0.8373 | 0.6301 / 0.5791 | 0.6853 / 0.6013 | pass |
| paired-junction, seed 42, terminal | 0.8723 / 0.8417 | 0.6403 / 0.5908 | 0.7000 / 0.5949 | pass |
| paired-junction, seed 43, selected | 0.8718 / 0.8417 | 0.6319 / 0.5800 | 0.6824 / 0.6139 | pass |
| paired-junction, seed 43, terminal | 0.8707 / 0.8420 | 0.6389 / 0.5938 | 0.7029 / 0.6076 | fails ordinary I |

## Paired uncertainty and component interpretation

All 24 prespecified parent/continuation/objective/module contrasts have 2,000 paired protein-family and condition bootstrap draws where the grouping is valid. Complete pair membership, common predictions/labels and grouping semantics were verified. Intervals are pointwise, conditional on this development cohort and fitted models; they omit selection uncertainty and multiplicity correction. Study-component IDs are not treated as protein clusters.

For the selected boundary candidate minus parent on the new benchmark, condition-cluster AUROC differences are +.01738, 95% CI [.01041, .02507] for I and +.01225, CI [-.00661, .03316] for II. Class-II has only eight condition clusters. Its positive AUROC point estimate replicates, but condition-level uncertainty crosses zero in both seeds. Class-II pair-win uncertainty also crosses zero. The seed-42 terminal AUROC intervals are positive for both classes, but that is insufficient to erase the cross-seed and pair-ordering limitations.

Continued ordinary training improves ordinary class-II AUROC but leaves affinity-matched performance approximately unchanged. Matched supply supervision improves the supply readout and pooled conditional ranking. The extra boundary module improves pooled ranking further, but does not establish a corresponding class-II pair-ordering benefit. Full decomposition and fixed stricter-caliper, allele, rank and BA-difference-sign sensitivities are in `results/analysis/`; small strata are descriptive, never replacement primary endpoints.

The combined stricter subset (at most .1 percentile point, .1 log-rank and .1 log-nM) retains 72 I and 58 II pairs. Selected boundary AUROC improves in both seeds for both classes, but class-II pair wins decline from .5862 to .5690 in both. This supports retaining the pooled-versus-paired distinction; the new primary cohort is not replaced by a favorable subset. An earlier descriptive label for the hybrid-plus-nM filter was clarified, and the separately registered combined filter is now explicit.

Input controls remove or shuffle APC signatures and source contexts while preserving labels, peptides and MHC. Whole-pair, role-preserving flank shuffle retains the marginal positive/background flank distributions; it tests source linkage, not all flank information. Unknown-input controls also change availability. These are fixed-checkpoint sensitivity tests, not trained ablations or isolated causal effects. Exact original-input predictions must reproduce the independent evaluations before interpreting a control.

## Measured affinity and verification

Original and ancestry-screened IC50 Spearman/Pearson/log10(nM) RMSE and 500-nM classification metrics are retained, along with all other available assay endpoints, losses and per-example predictions. For the selected boundary candidate, ancestry-screened I Spearman/RMSE are .68676/1.03526 versus .68141/1.04258 for parent; II .49483/.84218 versus .48996/.83597. All supported parameters were trainable; affinity retention is empirical, not a consequence of freezing. The original affinity cohort has 346 ancestor-exposed rows and is not substituted for its screened sensitivity.

All six 2,000-update fits and three 16-update integrations pass coverage. Ordinary draw order/source weights/scalers agree across arms, auxiliary pairs remain TRAIN-only, and the boundary head receives real gradients/updates. Thirteen model evaluations on six cohorts independently reproduce 1,066 numeric checks across 78 combinations. Core tests: 202 passed; bootstrap/control tests: six passed. CPU label-free class-I/class-II presentation and IC50 inference succeeds. Input-control checks and final lifecycle receipts are linked in the README. No TEST metrics or predictions were produced because the split remains sealed.

## Limits and next data generation

BA is the accepted working affinity estimate. Residual BA within-pair discrimination, limited studies, monoallelic validation, unobserved backgrounds, and unknown external-predictor training overlap remain limitations. Same-protein/condition pairs do not identify protein-expression or absolute-yield effects. The matched subset has no quantitative expression annotations and sparse clean drug/KO comparisons; actual PMID-linked support is in `results/PROCESSING_CONTEXT.md`.

The user subsequently requested Hitlist 2.1.12. Its refreshed loader-based exports and dependency verification are a separate source generation recorded in `tasks/hitlist-2.1.12-refresh/`. This family's frozen 2.1.9 data and model runtime are unchanged. Any next fit must regenerate protected TRAIN/VAL inputs and current annotation audits from that new generation; swapping source/version labels onto these results is invalid. A next bounded model comparison should target final within-pair presentation ordering and ordinary retention explicitly, rather than enlarging a head based only on pooled AUROC.

## Completed input controls

Five checkpoints (parent, paired seed-43 selected, both selected boundary replicates, and boundary seed-42 terminal) each have original/APC-unknown/APC-shuffled/flank-unknown/flank-shuffled predictions on four fixed cohorts. All 147,350 control records pass identity, label, MHC and original-score reproduction checks; all 20 paired control contrasts are retained. See [control summary](results/analysis/input-control-summary.csv).

For the selected seed-42 boundary model, hiding APC state lowers ordinary class-II AUROC from .8373 to .8185; within-study APC shuffle gives .8359. Hiding flanks lowers new matched class-II AUROC from .5605 to .5515 and pair wins from .5839 to .5401. However, role-preserving paired flank shuffle gives AUROC .5628 and the same .5839 win rate. Thus input use is verified, while improved peptide-specific flank ordering is not established for class II. These inference controls neither identify isolated interventions nor replace trained ablations.
