# Shared class-I/class-II presentation expansion

2026-11-17 · Codex / GPT-6 · **Completed: 12 end-to-end H100 fits, selected and fixed-terminal evaluations, independent verification, and usable shared checkpoint.**

Use **`expanded-state-sqrt-s42-n6000`**, selected at update 6,000 by the registered equal-class/equal-study presentation BCE criterion. Broad native training through the centralized adapter, repaired study sampling, and shared APC state improve ordinary presentation prediction in both classes while retaining measured affinity. All supported parameters train jointly. TEST remains sealed; these are adaptive development-validation results.

| cohort | class | rows | previous_AUROC | shared_AUROC | delta | paired_95CI | cluster |
| --- | --- | --- | --- | --- | --- | --- | --- |
| expanded | I | 780 | 0.8503 | 0.8759 | 0.0255 | [+0.0124, +0.0375] | protein |
| expanded | II | 1194 | 0.6872 | 0.8373 | 0.1501 | [+0.0768, +0.1997] | protein |
| apc | I | 1860 | 0.9239 | 0.9257 | 0.0018 | [-0.0026, +0.0064] | protein |
| apc | II | 372 | 0.7627 | 0.7742 | 0.0115 | [-0.0173, +0.0373] | protein |
| broad | I | 4830 | 0.9500 | 0.9532 | 0.0032 | [+0.0008, +0.0055] | condition |
| matched | I | 680 | 0.6342 | 0.6142 | -0.0201 | [-0.0323, -0.0080] | protein |
| matched | II | 316 | 0.5763 | 0.5692 | -0.0071 | [-0.0379, +0.0252] | protein |

The `expanded` rows are the new protected validation cohort; `apc` and `broad` are unchanged older cohorts. Each comparison uses the same examples. The older class-II result of .7627 and the new .8373 describe different cohorts and must not be treated as one before/after comparison. Affinity-matched class-I AUROC **regresses**; this model is an ordinary-presentation working default, not a universal processing-benchmark winner.

[Checkpoint decision](DECISION.md) · [Run inference](USAGE.md) · [Architecture](ARCHITECTURE.md) · [All metrics](results/analysis/report.md) · [Detailed specification and amendments](reproduce/spec.md)

![Common-cohort comparison](results/analysis/comparison-expanded-state-sqrt-s42-n6000.png)

## Data and curation contract

Pinned Hitlist 2.1.9 checkout `43e226621434a483141764ed54faba725bbb3869`; observation-cache SHA256 `0b34bd12471882eb28d4f9a1dad6cc48252fb93a3644f912f69a0a186228f847`. Shared inputs/reference receipts: `../../data/shared-presentation-20261117/`. No shared-environment packages changed.

| Stage | Class I | Class II |
|---|---:|---:|
| Uncapped public human MS export | 2,812,724 rows | 1,328,975 rows |
| Native admission after fixed sequence exclusions | 507,846 positives; 232 studies | 329,576 positives; 69 studies |
| Final TRAIN | 209,695 positive/background pairs; 159 studies | 178,179 pairs; 42 studies |
| Additional validation | 390 pairs; 76 studies | 597 pairs; 9 studies |

Expanded TRAIN contains 775,748 rows. Fixed concentration replay contains 159,796 observations. The pilot control retains the original 10,632 APC and 65,629 broad presentation TRAIN rows; that pilot had only 845 class-II MS positives. Old validation contains 2,536 affinity, 2,232 APC and 4,830 broad rows. New validation contains 1,974 rows. Ordinary validation therefore totals 11,572 rows. Separate affinity-matched validation has 996 rows (340 class-I and 158 class-II pairs). Finite 6,000-update budgets do not visit every eligible row; each fit preserves actual draw order and distinct exposure counts at selection and completion.

The centralized native adapter admits observed MS detections with literal lineage and reported typed candidate MHC sets. It does not require an assigned intervention. Known PTMs, contaminants and predicted-only restrictions are excluded; unspecified PTM status follows the declared sequence-only-unknown policy. DR reference-alpha assumptions remain explicit; DP/DQ partners are never invented.

Reported cell identity is independent of treatment-arm attribution. Unresolved treatments remain unknown. Expanded TRAIN supplies 23 cell categories, 31 intervention factors and IFNG; 326,031 positives have reported cell identity. The shared APC encoder has 87 inputs and 64 latent dimensions. No measured expression targets or calibrated knockout-severity labels are fabricated.

GENCODE v39 mapping considers all exact reference occurrences and related protein families. Unanimous source flanks are retained; ambiguity masks both pair members. Possible reference source assignments are not asserted to be measured source proteins. Each positive receives one same-length source-protein background excluding cached observed identities and protected relatives. MS detections map to `elution_observation=1`; unobserved sampled windows map to 0 with explicit provenance. These are sampled comparisons, not measured nonbinders. NetMHCpan and NetMHCIIpan do not supply TRAIN outcomes.

Exact/shared-8mer/Hamming≤1 and source-family protections cover frozen validation, matched candidates, the sealed identity-only embargo and new validation. Parent ancestry includes original affinity TRAIN, not just later replay. Independent audit found **0 violations** across four TRAIN populations and **0 historical sequence/family exposures** in new validation. Receipts pin 295,335 protected identities, including the 180,116-identity sealed embargo. TEST outcomes were never read.

Historical affinity validation has 346 related rows under the stricter current criterion, with 0 exact matches. Both unchanged and ancestry-screened 2,190-row cohorts are reported. Fixed APC/broad/matched presentation cohorts have 0 such ancestor sequence exposures. See `results/audit.json`, `ancestry-check.json` and `fixed-validation-ancestor-exposure.json`.

## Available APC evidence

[Actual native TRAIN APC support](results/analysis/apc-training-support.md) lists each factor, explicit present/absent counts and source PMIDs after exclusions. `results/context-conditional-support.json` audits 387,874 TRAIN positives, 987 new-validation positives and 1,116 fixed-APC-validation positives. HAP1 knockouts, ERAP1 knockdown and TAX1BP1 knockdown have explicit present/absent annotations within study/reported-line strata. Drug factors have assigned present observations but no explicit drug-absent claims in this native projection. Concurrent signatures and condition IDs are retained.

Class II includes 69,992 HLA-DM-transfection, 10,245 CIITA-transduction and 610 TAX1BP1-knockdown positives (1,017 explicit TAX1BP1-control positives), plus 142 decitabine, 98 tazemetostat and 186 IFNG positives; concurrent counts overlap. The HLA-DM observations support conditional presentation training, but lack a clean intervention contrast. Some reported lines are broad cell labels. These annotations do not establish causal or absolute-yield effects. Curation requests are tracked in [Hitlist #905](https://github.com/pirl-unc/hitlist/issues/905), covering issues #899–905.

## Shared implementation and training

Canonical isolated source: `../../artifacts/harmonized-apc-source`, branch `codex/harmonized-apc-20261116`. Original six fits use clean frozen core `2da11504f71f49c3d1780b4c982db07966d7063b`; the six square-root fits use `5f9bfe37dfb5e82cecdf762ef1e59ff885668e40`. Both runtime snapshots contain 228 files. Root unrelated work is preserved and nothing was pushed. Later artifact commits do not change these runtime identities.

- `data/presentation.py`: shared reference mapping, source families, flank ambiguity, sequence protection and source-matched backgrounds.
- `data/modular_hitlist_ms.py`: native admission, unknown/control semantics and independently reported cell identity.
- `data/modular_context.py`: explicit TRAIN-vocabulary projection to unknown, retaining omitted assertions in provenance. Public inference is strict by default.
- `training/modular_sampling.py`: study strata and deterministic, resumable finite weighted draws.
- `training/modular_run.py`: equal-class/equal-study presentation BCE selection, failing on missing classes or provenance.

All arms warm-start parent SHA256 `1d0d4122abf022830ea97fcc247b1d28c0645d7d42d206130cdf8a5d7762815a`, a width-128, two-layer, four-head sequence model with prior staged affinity/processing training. **All supported parameters train end to end; none are frozen.** Shared APC width is 64 when enabled. Intrinsic affinity has no APC input. Class II retains candidate nine-residue registers; no register-orientation or pooling change was made.

Each fit executes 6,000 updates, batch 64, AdamW learning rate 1e-4 and weight decay 1e-4, 100-step warmup/cosine floor .1, validation every 1,000 steps. Draw mass is 50% measured concentration, 25% class-I presentation, 25% class-II presentation. Concentration replay preserves literal IC50/EC50/KD/proxy endpoint identities, original qualifier/censor handling, and canonical standardized log10(nM) losses. Presentation uses BCE. Mapping is assay-specific concentration versus `elution_observation`; no predictor-label or intervention-yield loss is added.

Selection minimizes mean study BCE within each presentation class, then averages the two classes. Affinity is evaluated separately. Operational retention screen: no Spearman decrease greater than .02 or log10(nM) RMSE rise greater than .03 versus the previous default on either original or ancestry-screened IC50 validation. This is an operational guard, not formal non-inferiority.

| Recipe / seed | Selected update | New I AUROC | New II AUROC | Older APC I | Older APC II |
| --- | --- | --- | --- | --- | --- |
| expanded-ii-transfer-s42 | 1000 | 0.8528 | 0.7747 | 0.9136 | 0.7315 |
| expanded-none-s42 | 4000 | 0.8617 | 0.8037 | 0.9123 | 0.7348 |
| expanded-none-s43 | 4000 | 0.8650 | 0.8092 | 0.9126 | 0.7431 |
| expanded-none-sqrt-s42 | 6000 | 0.8745 | 0.8235 | 0.9206 | 0.7748 |
| expanded-none-sqrt-s43 | 5000 | 0.8731 | 0.8244 | 0.9202 | 0.7722 |
| expanded-pathway-state-sqrt-s42 | 6000 | 0.8757 | 0.8374 | 0.9258 | 0.7740 |
| expanded-pathway-state-sqrt-s43 | 4000 | 0.8729 | 0.8313 | 0.9239 | 0.7727 |
| expanded-state-s42 | 1000 | 0.8602 | 0.7617 | 0.9132 | 0.7373 |
| expanded-state-s43 | 4000 | 0.8658 | 0.8113 | 0.9161 | 0.7431 |
| expanded-state-sqrt-s42 | 6000 | 0.8759 | 0.8373 | 0.9257 | 0.7742 |
| expanded-state-sqrt-s43 | 4000 | 0.8730 | 0.8314 | 0.9239 | 0.7728 |
| pilot-balanced-s42 | 1000 | 0.8391 | 0.6871 | 0.9167 | 0.7616 |

`pilot-balanced` uses the old cohort with APC64; `expanded-ii-transfer` retains prior class-I data and expands class II. `expanded-none` and `expanded-state` use identical broad data without/with APC64. `expanded-pathway-state-sqrt` additionally supplies a zero-initialized 2→128 class projection before the shared processing nonlinearity (256 weights). The latter gives negligible predictive benefit across both seeds, so the default leaves it off.

Original recipes allocate equal mass to presentation studies. They overallocate studies with ≤10 pairs: 26–27% of actual presentation draws despite <.1% of eligible rows. The adaptive `-sqrt` recipes use study mass proportional to the square root of admitted TRAIN rows, normalized within each class. Tiny-study expected mass becomes 1.96% I/1.54% II; largest-study mass is 4.86% I/19.14% II. Every admitted row remains eligible. Data, parent, optimizer, class balance and selection criterion stay unchanged.

[Complete selected and terminal comparisons](results/analysis/report.md), [actual runtime/selection](results/analysis/selected-runs.csv), and [learning curves](results/interim-presentation-curves-sqrt.png) retain all conditions, including regressions. The 12 terminal evaluations are alternative checkpoints from the same 12 fits, not 12 additional independent fits. Where selection is at step 6,000, selected/terminal weights coincide; serialized artifact hashes may differ.

## Results and APC attribution

The preferred shared model reaches combined ordinary AUROC **.938402 I / .822642 II**. Exact IC50 Spearman is **.670499 I / .498009 II**, with RMSE **1.048022 / .828544 log10(nM)**. Both original and ancestry-screened retention guards pass. All regression and ≤500 nM accuracy, balanced accuracy, precision, recall, F1, AUROC and AUPRC metrics are in `results/analysis/metrics.csv` and `affinity-retention.csv`.

Broader data and repaired sampling account for most of the class-II gain. At the same 6,000-update endpoint, APC versus no-APC improves older class-I APC validation by **+.00512 / +.00524 AUROC** in seeds 42/43, with positive protein- and condition-cluster intervals. New class-II increments are **+.01373 / +.01247**. The primary seed protein-cluster interval is [.00027, .02760], but its condition interval crosses zero; both replication intervals cross zero. Thus the incremental class-II state evidence is suggestive and sensitive to uncertainty definition, rather than consistently resolved.

| Preferred checkpoint, new cohort | Reported state AUROC | All state unknown | Within-study shuffle | Pair win: reported / unknown / shuffled | Rows with changed shuffled signature |
|---|---:|---:|---:|---|---:|
| Class I | .875871 | .869507 | .876108 | .874359 / .871795 / .874359 | 72/780 |
| Class II | .837274 | .815479 | .836098 | .815745 / .825796 / .814070 | 90/1194 |

Controls move complete state signatures pairwise within study/class, preserving sequences, flanks, MHC sets and outcomes. No-state models are exactly invariant. Most groups have little within-study variation, and shuffle largely retains performance. Unknown state is missing information, not an untreated control. These checks support state-associated prediction but **do not establish isolated gene/drug effects or absolute knockout severity**. All controls and paired intervals are under `results/context-ablation/`.

Affinity-matched class I regresses from .634239 to .614152 (delta −.02009; protein-cluster 95% CI [−.03234, −.00802]). Class II is .576310→.569220 with an interval crossing zero. Keep the previous checkpoint as a matched-benchmark reference. Ordinary ranking gains do not establish improved processing discrimination at fixed predicted affinity.

## Evaluation and verification

Selected checkpoints follow the original loss rule. Symmetric fixed-6,000-update square-root evaluations were registered before terminal outcomes; original-recipe terminal evaluations are retrospective update-matched controls. Follow-ups used validation adaptively. No arbitrary intermediate AUROC peak or TEST outcome was substituted for the selection rule.

Frozen NetMHCpan 4.2c / NetMHCIIpan 4.3e BA predictions define separate working-affinity matching. EL scores are comparators. This is not a latest-version claim (see `reproduce/register-literature-notes.md`). MHCflurry actual source 2.3.28 / presentation weights 2.3.0, with/without flanks, is reused on byte-identical matched rows with hashes in `results/frozen-comparator-reuse.json`. Its bundled-training screen leaves 106/340 class-I pairs; NetMHC screens leave only 6/340 I and 1/158 II pairs, below the 10-pair reporting minimum. Full comparator results are descriptive, not comparator-unseen generalization. The initial Claude 10,998-pair report is a different cohort and is not independently validated here.

Paired uncertainty uses 1,000 identical-example resamples by protein family or sample/condition. Intervals condition on fitted models and development cohorts; they exclude adaptive-selection and seed uncertainty. Older broad class-I validation has only three study-component split groups and mostly incomplete source pairs, so use condition-cluster AUROC intervals and omit whole-cohort pair-win/protein intervals. Earlier exploratory files affected by this correction are archived under `results/analysis-correction/`; four independent bootstrap tests verify the corrected computation.

Independent verification checks **12 selected + 12 terminal** exports: all 11,572 ordinary examples, outcomes/losses, exact selection criterion, source/checkpoint hashes, coverage, full trainability, terminal history and actual draws. Core commit 5f9bfe37 passes **251 regression tests** plus six focused pathway checks. Original predictions remain unchanged with the new option disabled. Two resume/staging tests and all original-score/no-state APC controls pass. Full-data 16-step diagnostics were integration checks only; qualified models pass complete support coverage.

The preferred checkpoint passes CPU inference through the canonical public `ModularPredictor`, with four label-free class-I/II presentation and IC50 requests and hash verification. Remote evaluation checks public/internal prediction parity and context-invariant intrinsic affinity. See `results/example-predictions.json` and [USAGE.md](USAGE.md).

**No TEST metrics or prediction dumps exist for this family:** TEST is intentionally sealed throughout adaptive development. Identity-only embargo protects it without reading outcomes.

## Hardware, reproduction and artifacts

Requested GPU: **H100!**. Observed: NVIDIA H100 80GB HBM3, 85,009,367,040 bytes, torch 2.14.1+cu130. Preparation/audit use 8 CPU / 64 GiB. Fits match update budget and draw mass, not FLOPs/wall time; candidate sets and admission costs differ. Preparation took 2,519.89 s, context refresh 263.78 s, audit 245.76 s. Preferred fit active training: 4,902.47 s; elapsed wall time: 7,397.22 s including interruptions. All runtimes are in `results/analysis/selected-runs.csv` and per-fit receipts.

Network interruptions required exact-state resume for six unfinished runs. Original source/manifests, optimizer, RNG, sampler/order and selection state were preserved and checked; host changes/downtime are recorded without claiming bitwise cross-host equivalence. Completed worker/evaluation outputs were reused only after hash checks. Asynchronous coordinator call IDs, original failed transport attempts, and process-local TLS-verified DNS fallback are preserved in `reproduce/launch.json`, frozen wrapper manifests, `results/recovery-closeout.json` and logs. **All work completed; final container inventory is empty** (`results/containers-closed.json`). No system network settings or shared packages changed.

Commands below run from this experiment directory using the existing environment. Exact original invocations, non-default overrides, git status, sources and amendments are frozen under `reproduce/`.

```sh
reproduce/launch.sh --action prepare
reproduce/launch.sh --action refresh-context
reproduce/launch.sh --action audit
reproduce/launch.sh --action fit --arms pilot-balanced,expanded-none,expanded-state --seed 42 --steps 6000
PRESTO_MODAL_GPU='H100!' modal run code/launch_pathway.py \
  --arms expanded-none-sqrt,expanded-state-sqrt,expanded-pathway-state-sqrt --seed 42 --steps 6000
# Repeat the square-root comparison with seed 43; complete matrix is in launch.json.
python code/verify_results.py
python code/analyze.py --preferred expanded-state-sqrt-s42-n6000
python code/analyze.py --preferred expanded-state-sqrt-s42-n6000 --bootstrap-only --bootstrap-cluster condition
python code/report.py
python code/decision.py --preferred expanded-state-sqrt-s42-n6000
python code/predict_example.py
```

Remote root: `presto-checkpoints:/2026-11-17_1155_codex_shared-presentation-expansion`. Preferred remote checkpoint: `fits/expanded-state-sqrt-s42-n6000/raw/selected.pt`, SHA256 `1b6011396a563f52ea1d6da4bb1d4e69b8dbf0103148ba829225581c4d312f68`. Parent: `/volume/2026-11-08_1800_codex_unified-parity-retrain/fits-v3/reference/raw/selected.pt`. Previous default: `/volume/2026-11-16_2035_codex_joint-apc-capacity/fits/w64-s42-n2400/raw/selected.pt`, SHA256 `cbaf066af510d19d59a223812c2884541cf0725f0cc243fecc7e8417b8157b3b`.

Per-example dumps are local under `results/fits/<run>/raw/validation-predictions.jsonl.gz`, `results/evaluation/<run>/` and `results/context-ablation/<run>/`; matching remote keys and byte hashes are in `results/collection-*.json`. Prepared data hashes and leakage receipts are in `results/preparation.json`, `audit.json`, `context-refresh.json` and per-fit manifests. Compact source packaging excludes large raw checkpoints/predictions/derived parquet while retaining their exact paths and hashes in `reproduce/artifact-omissions.json`; use `code/collect.py` to retrieve declared remote artifacts. Locally derived tables remain reproducible from the per-example dumps. See [USAGE.md](USAGE.md) for checkpoint retrieval.

Rebuilding TRAIN from raw sources requires the pinned repo-level `data/shared-presentation-20261117/` inputs, retained outside git; a fresh checkout must provision those exact files from their recorded source receipts. Prepared sources already exist in the registered Modal experiment. Checkpoint/prediction retrieval and inference need no new Hitlist export.
