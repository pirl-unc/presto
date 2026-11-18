# Foundation Review PRs + Iterative Shared Processing/Presentation & APC Semantics Retraining

Date: 2026-11-18
Base branch: `codex/consolidated-foundation-20261118` (`decf1280`, PR #158)

## Goals

1. **Stage A — PR 1 (Foundation review items 1–4)**:
   Ship a clean, well-tested pull request (`codex/foundation-review-1-4-20261118` -> `codex/consolidated-foundation-20261118`) covering all registry, locking/CLI isolation/atomic-write/GPU-RNG, manifest-split/kinetics, and evaluation-strata/streaming-predict fixes (items 1–4) along with dedicated regression tests, CLI import-isolation checks, and pinned Ruff lint/format verification.

2. **Stage B — PR 2 (Foundation review items 5–7)**:
   Ship the follow-on pull request (`codex/foundation-review-5-7-20261118` -> `codex/foundation-review-1-4-20261118`) completing items 5–7:
   - `evaluation/external_benchmark.py`: physical concentration unit conversion (`nM`, `uM`/`µM`/`μM`, `mM`, `M`, `pM`) in `extract_record`, canonical `target_log10` bound scoring in `censored_metrics` (falling back to `log10(raw_nM)` only when `target_log10` is absent), and schema-preserving empty DataFrame handling across `extract_records`, `unit_metrics`, `pooled_metrics`, `censored_metrics`, `qualitative_metrics`, `exclusion_census`, and `score_dump`.
   - `evaluation/paired_benchmark.py`: molecular/context truth alignment in `prediction_identity` (including MHC grooves/classes/species, TCR chains, and predictive context fields), publication-only study provenance in `study_id` (never falling back to split `group_id`), and strict `>` lower-bound handling at the threshold cutoff in `affinity_threshold_metrics` (`> 500 nM` at `threshold_nm=500` is a definite negative, while `>= 500 nM` remains ambiguous).
   - `evaluation/performance_report.py`: validate that `prediction_artifacts[*].split` and `metrics[*].split` match roles actually declared in `data.datasets`.
   - Regression tests covering all 15 review findings plus full focused suite and Ruff lint/format gates.

3. **Stage C — Shared-Mechanism Model Retraining, `mhctools` Class-I/II Benchmarking, and APC KO/Cytokine/Over-Expression Semantics**:
   - **Canonical Registry & Hardware**: Register every benchmark and training iteration under `experiments/YYYY-MM-DD_HHMM_antigravity_<slug>` and `experiments/experiment_log.md` per `experiments/EXPERIMENT_WORKFLOW.md`, running GPU jobs detached on Modal with `PRESTO_MODAL_GPU=H100!` and keeping `TEST` outcomes sealed.
   - **Comprehensive `mhctools` Baseline Suite (Class I & Class II)**:
     Enumerate and score all runnable predictors in `mhctools` (including NetMHCpan BA/EL, NetMHCIIpan BA/EL, MHCflurry Affinity/Presentation, and all additional runnable `mhctools` class-I/II predictors) on identical held-out validation cohorts:
     1. Ordinary presentation validation (Class I & Class II; pooled AUROC/AUPRC, within-study/within-allele AUROC, and paired win rates).
     2. Affinity-matched processing validation (Class I & Class II; testing processing/presentation discrimination beyond BA).
     3. Condition-annotated APC cohorts (Class I & Class II).
   - **APC Gene KO / Cytokine / Over-Expression Semantics & Conditioned Data Fit**:
     Audit actual Hitlist 2.1.12 / v4 source records and validation cohorts for gene knockouts/knockdowns (`B2M`, `TAP1`, `TAP2`, `TAPBP`, `ERAP1`, `ERAP2`, `CALR`, `CANX`, `PDIA3`, `HLA-A/B/C`, `TAX1BP1`, etc.), cytokines (`IFNG`), and overexpression/transduction/transfection (`CIITA`, `HLA-DM`, `HLA-B`, `overexpression:*`).
     Verify and train shared class-I/II processing + APC state + junction mechanisms (including cross-arm counterfactual state-contrast / pathway-consistent supervision and final-logit + supply matched-pair ranking on the corrected v4 + replay cohorts) so that:
     1. Processing and presentation metrics improve across iterations and beat `mhctools` comparators on both Class I and Class II via shared mechanisms.
     2. Specifying the true experimental/cellular APC conditions matches held-out data demonstrably better than omitting (`unknown`) or shuffling those conditions.
     3. Model predictions obey expected biological semantics for gene KO, cytokine exposure, and overexpression/transduction/transfection configurations without degrading intrinsic affinity retention.
