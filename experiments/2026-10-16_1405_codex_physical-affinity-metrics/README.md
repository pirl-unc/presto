# Physical affinity metric audit — #116

The completed data-only replay corrects four training classifications at exactly
500 nM and six undefined constant-baseline correlations. Validation/test binding
labels, exact-value RMSE/MAE/Spearman, predictions, checkpoints, partitions and the
failed quality decision remain unchanged. No fitting, selection or new biological
evidence was introduced. Codex / GPT-6 performed this audit.

## Frozen inputs and reproduction

- Clean executed recipe: `9a8de46dee32dbe626153f7b7442fbe8c8ce0e1f`, tag
  `experiment-physical-affinity-metrics-20261016`; auditor Presto 0.1.18 candidate.
- Original producer: released Presto 0.1.16 / Hitlist 1.78.0, recipe
  `1067714c768af2e4f3184154c85903bb3c46cf4b`. Its immutable
  [public qualification release](https://github.com/pirl-unc/presto/releases/tag/effective-target-fit-qualification-v1)
  retains all 90 verified archive files, predictions and checkpoint bytes.
- Archive SHA-256: `134c5ce774ad473f7f5a61273cdad864f31d17e1428e5703405a3e5243448bc9`;
  inventory SHA-256: `8539de79868d541c53688c7af30e388f834042338bfdb5dd77a4b40abca88494`.
- Replay the frozen tag with its checkout installed, datacache 1.27.1, NumPy 2.5.3,
  pandas 2.3.3 and Torch 2.14.1: `reproduce/launch.sh`. Actual local overrides were
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and an isolated checkout import path;
  no shared environment was changed. `audit.json` records exact code/dependency pins.
- CPU only, no cloud/GPU. Builder runtime 0.835 s, process peak RSS 623,738,880 bytes;
  acquisition/import time is excluded from runtime, but included in process peak RSS.

## Scope and result

All 2,614 measured IC50 observations retain their original nM values and censor
operators: train 2,074 (1,599 exact), validation 265 (210 exact), test 275 (214 exact).
No synthetic observations were added. This is the same frozen 9,053-observation
cohort with 7,241/906/906 partition membership. The original fit used 200 shuffled
AdamW updates of the canonical d32/l1/h4 model, uncertainty-weighted active measured
objectives, no pretraining or synthetic examples, and complete aggregate validation
for checkpoint selection. This replay does not run optimization or modify losses.

The threshold estimator now uses physical nM targets where available, with explicit
fallback counts for transformed-only diagnostics. All canonical IC50 observations
in this audit have physical nM targets. Predictions retain their continuous
log10(nM) cutoff; conservative censor-direction grouping is unchanged. Exactly
constant Pearson inputs produce an undefined result, not a tiny rounding residual.

| Quantity | Historical report | Corrected additive report |
| --- | ---: | ---: |
| Positive training labels among 2,070 classifiable observations | 764 | 768 |
| Selected-model training threshold accuracy | 0.630918 | 0.628986 |
| Constant-mean Pearson entries (3 splits × 2 target spaces) | tiny finite values | undefined |
| Validation/test labels changed | — | 0 |
| Validation/test Presto effective RMSE | 1.165477 / 1.310036 | unchanged |
| Validation/test ridge effective RMSE | 1.037626 / 1.138279 | unchanged |
| Validation quality screen | failed | failed |

`metric-changes.tsv` contains all 20 split/target-space/predictor comparisons.
`boundary-observations.tsv` identifies the four changed training observations and
retains peptide, source/resolved MHC, assay endpoint/method, nM value/operator,
PMID/DOI/reference/assay links, cell type/line and condition fields. Blank fields
mean unavailable in the source export; no cellular context was invented for these
purified-MHC measurements. `canonical-thresholds.json` contains complete selected
model threshold metrics for all three splits. The original archive preserves the
full observation tables, membership, losses, support reports and predictions.

## Downloads and limits

[Additive audit downloads](https://github.com/pirl-unc/presto/releases/tag/affinity-metric-numerics-audit-v1)
include these tables, audit/code receipts, reproduction instructions and hashes.
Acquisition, audit generation, publication staging and independent public retrieval
use datacache. `code/verify.py` independently checks the public rows and reporting
with NumPy and scikit-learn; its receipt records the actual outcome.

Ridge remains the provisional validation baseline. Shared studies, unknown donor/
specimen/acquisition identities, species-role ambiguity (#115), near-constant model
predictions (#117) and undeclared outputs remain limitations. This audit neither
establishes biological independence nor makes an all-output readiness claim.
