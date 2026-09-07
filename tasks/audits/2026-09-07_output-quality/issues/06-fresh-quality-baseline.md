There is no registered real-data training result demonstrating predictive quality for the post-#47 sequence-only encoder across supported outputs. Existing unit training and narrow historical runs establish useful prerequisites, but cannot substantiate that every output learns or agrees with known ground truth.

Audit target: `1535b5a208b2ab59b70b83d850b1f0a7e1959537`. This is the experimental acceptance work for #46/#47, dependent on the supervision and evaluation repairs identified by this audit.

Prerequisites: #48 (coverage census/gates), #49 (binding descriptors), #50 (T-cell bag-panel supervision), #51 (complete held-out extraction), and #52 (correct tied-score AP). These repairs are required before claiming a complete multi-output result; a fitting diagnostic may proceed with its narrower limitations declared.

### Evidence already available—and its limits

- [PR #45 smoke](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/experiments/2026-09-05_0957_codex_pr45-integrity-e2e/README.md#L1): old code `c4dc1cf`, 216 Hitlist samples, 130/43/43 split, d32/l1/h4, one epoch and **two training batches**, no synthetic augmentation. It explicitly disclaims model-quality interpretation.
- Its validation/test exact-binding Spearman is **-0.204/-0.159**, RMSE **1.621/1.628 log10(nM)**, 500 nM balanced accuracy **0.5/0.5**. These numbers neither validate the current model nor prove it cannot learn with sufficient optimization.
- The broader merged condition in that family was **preflight only**. The subsequent lineage/funnel closure was also data-only.
- The [groove-corrected replacement family](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/experiments/2026-09-04_1121_claude_groove-corrected-baseline/README.md#L1) says its GPU conditions were not launched.
- Older September 2–3 model runs predate the encoder and supervision changes. Some mapping-policy conclusions carry a null-to-NAN validity erratum; high elution scores often compare real positives with generated decoys.
- [Current fixed probes](https://github.com/pirl-unc/presto/blob/1535b5a208b2ab59b70b83d850b1f0a7e1959537/scripts/train_iedb.py#L625) emit predicted KD/processing/presentation for SLLQHLIGL and requested alleles. They do not attach a measured value, qualifier, assay endpoint, observation ID, or train/validation/test membership. Inter-allele variance is sensitivity, not ground-truth accuracy.
- CLI defaults are `test_frac=0`, optional support gates, optional checkpoint/run directory, and a MIL instance cap. Defaults alone are not a complete scientific acceptance contract.

### Required run specification

- [ ] Freeze the exact code revision, dependency versions/source commits, dataset artifacts and hashes, curation/filtering, source choices, assay/qualifier policies and per-output support manifest. Use fresh compatible weights; no silent old-checkpoint migration.
- [ ] Choose supported endpoints from measured support, with explicit exclusions and reasons. Distinguish direct outcomes from proxies/auxiliaries and count aliases once.
- [ ] First run a bounded fitting check on a representative **real training subset** with known observed labels. Keep these examples identified as training examples; do not reuse validation/test observations for fitting.
- [ ] Save initialization-to-trained per-output losses, effective weights, gradients/update counts, prediction ranges and representative errors. Show improvement over constant/prevalence/mean baselines where applicable; aggregate loss alone is insufficient.
- [ ] Train the canonical model with a declared adequate optimization budget and peptide-disjoint train/validation/test splits; pin data seed separately from split/model seeds. Check cross-source duplicate and synthetic-parent overlap. Use validation only for model selection.
- [ ] Reload the chosen checkpoint in a fresh model and evaluate all supported row/bag/column outputs under the exact evaluation contract. Preserve per-example validation/test dumps and complete loss terms.

### Ground-truth and quality acceptance

- [ ] Create a durable known-example table with observation/source ID, peptide, original/resolved MHC, biological context, assay family/column, raw measured value, unit, qualifier, split, prediction and residual/censor satisfaction. Include positive and negative examples and class II where support permits; justify sparse/absent categories.
- [ ] Match prediction to the measured assay endpoint: IC50 evidence is not an exact KD label; elution evidence is not direct precursor-frequency ground truth.
- [ ] Report exact-value Spearman/Pearson/RMSE and qualifier-aware binding metrics at 500 nM: accuracy, balanced accuracy, precision, recall, F1, AUROC and corrected AP.
- [ ] Report task-appropriate binary discrimination/calibration, categorical performance and graded-target agreement. Separate real-only outcomes from generated-decoy discrimination; include sample sizes and uncertainty, and label undefined metrics.
- [ ] Check physical-unit conversions, documented alias equality and guaranteed algebraic relations. Investigate contradictory predictions in their actual assay/context rather than enforcing unsupported causal inequalities.
- [ ] Define endpoint-specific success/failure criteria prospectively. A result can validate a supported subset while leaving other outputs explicitly unvalidated.
- [ ] Register the experiment and close it with launcher/env/source snapshots, model/checkpoint identity, hardware/runtime, all raw logs, summaries, curves, known-example tables and held-out predictions.

For Modal, request **H100!** explicitly and record observed hardware. runplz 4.4.4 can support orchestration; long detached Modal runs still require the capability tracked in pirl-unc/runplz#165 or an existing supported Modal launcher. Large artifacts should use a Volume.

This issue requests a concrete learning/quality demonstration, not another synthetic smoke or an all-output claim inferred from passing tests.
