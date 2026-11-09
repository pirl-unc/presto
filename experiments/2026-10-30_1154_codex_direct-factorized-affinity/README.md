# Direct factorized affinity decoder

Date: 2026-10-30. Agent/model: Codex / GPT-6. Status: launched on Modal; predictive results pending.

User-selected shared-v7 architecture: one 16-dimensional learned vector per
categorical value of quantity, preparation, mechanism and detector. Concatenate
these four slots with the common candidate latent and use one width-192 nonlinear
head (Linear/GELU/Linear) to predict the standardized log-concentration. The five
concentration endpoints share this head. There are no separate affinity heads or
small additive residuals. Unknown/conflicted method slots are fixed zeros;
source IDs are not decoder inputs. The same biological encoder and normalized
register pooling run for both MHC classes. Kinetics, stability, qualitative binding
and ligand outputs retain their own measurement definitions in the same model.

## Frozen data and training contract

Identical source bytes, observations, values, inequalities and split membership to
`../2026-10-29_2303_codex_assay-measurement-layer`. This is the curated Hitlist/IEDB
cohort, with existing study holdouts plus four retrospective fluorescence holdout
components and a cross-output peptide embargo. There are 3,518,242 training rows.
All source SHA256 values and original projection counts are frozen in
`reproduce/parent-prepared.json`. No new synthetic labels or new data curation:
inherited generated ligand comparisons retain their original contracts. Detector
metadata is source-tagged, not comprehensive verified protocol metadata.

Two fresh fits: direct-s42 and direct-s43, 4,000 updates each, batch128, width192,
3 layers, 6 heads, FF768, dropout0.1, constant AdamW3e-4, weight decay1e-4.
Same endpoint/class sampling as the pooled controls: 60% ligand, 20% literal IC50,
15% other physical measurements, 5% qualitative binding. No pretraining or warm
start. Selection: minimum equal-endpoint validation loss, evaluated each1,000
updates, checkpoints each500. Exact squared standardized residuals; one-sided
squared inequality violations; BCE for binary outcomes, unchanged weights.
The observation's quantity selects its established target scale and loss; its
method selects one output of the fixed shared decoder panel. Registry expansion
covers every usable concentration method in TRAIN, not just IC50. Known factors
in an unexposed combination mean indirect training, not direct combination evidence.

This family changes head sharing and decoder form relative to the pooled controls;
it does not isolate either change alone. All holdouts have previously been
inspected; no blind or biochemical qualification claim. The shared-v6 comparison
remains an immutable historical experiment, with any cancellation recorded there.

## Reproduction and artifacts

Exact invocation/environment and git/source fingerprints: `reproduce/launch.json`
and `reproduce/launch.sh`; model snapshot path is recorded there. New Modal source
upload reuses the already authorized data in volume `presto-checkpoints`.
Requested hardware: H100!, at most two concurrent GPUs. Actual hardware and memory
will be recorded in each report. Raw outputs: volume directory named after this
experiment; local verified downloads under repository `artifacts/<experiment>/`.
`analysis/watch.py` collects receipts, prediction dumps and metrics automatically.
Dashboard: http://127.0.0.1:54505/report.html .

## Verification and decision

Software fixtures test all categorical gradients/updates, both MHC classes,
shared quantity decoding, opposite assay-dependent ranking, exact/censored loss,
checkpoint/version behavior and the canonical manifest trainer. Artificial numbers
in those fixtures are only implementation/expressivity checks.

Held-out validation/test results, runtime, actual hardware, exact counts, bound
satisfaction, per-quantity/method/class regression and binding-threshold metrics,
ligand AUROC/AUPRC and full per-example dumps remain pending. No predictive gain
or winner is claimed. Architectural default follows the user's explicit decision.

## Launch receipt

Source revision: `7083bd899a29e8b251924079ca2941238e396784`. Frozen snapshot
contains 219 files (3,571,688 bytes). Modal app:
https://modal.com/apps/iskandr/main/ap-3BRMaO8FDAZyvk1gaz3JYs .
Coordinator: `fc-01M66S2W6JXF5JTYW580TJAG2D`. Two H100! fits; the old app was
stopped first and no Presto GPU containers remained. The CLI message “App
completed” means the detached local submission exited; remote fit status comes
from the coordinator/collector. `report.html` is an additive presentation update
showing frozen pooled controls beside new results; frozen training sources and
the original dashboard bytes are unchanged.

## Completed software verification

Full suite: **2,896 passed, four skipped**, 402.26 seconds. All 82 focused
source/decoder/support regressions pass; repository and experiment-source lint
and strict docs pass. A real completed v6 checkpoint produces identical CPU
predictions and fixed panel values under the frozen runtime and the new historical
reader for all four IC50 class/method cells. See `results/software-verification.json`.
The initial broad pass had two fixtures missing the now-required known assay
registry; those fixtures now explicitly declare their source-reported method.

One fit's startup inspection confirms NVIDIA H100 80GB HBM3, 85,009,367,040 device
bytes (81,559 MiB reported by nvidia-smi), Torch 2.14.1+cu130 and CUDA 13.0. The
11 source files total 30,252,937,352 bytes; staging/indexing occurs before the first
optimizer update. Final per-run hardware receipts will be collected automatically.


## Metric-interpretation audit (2026-10-30)

The combined MHC-I IC50 correlation is strongly affected by assay/study separation.
The 141 exact radioactivity values have a median of 9 nM; the 28 exact fluorescence
values have a median of 8,327.85 nM. As a post hoc distribution diagnostic, assigning
score 0 to radioactivity and 1 to fluorescence yields Spearman 0.63616 over these
169 outcomes without using sequence. This is not a trained or prospectively tested
baseline: the group direction was inspected after evaluation. The combined model
correlation must not replace within-assay evidence or be described as demonstrating
better peptide ranking by itself. Each class/method cell represents one study.

The reported ~24% censor satisfaction is 61/258 and 62/258 at the interim validation
readout. All 258 are fluorescence IC50 observations: 175 MHC-I values >=100,000 nM
and 83 MHC-II values > or >=10,000 nM. Unsatisfied predictions are too low in IC50
(too strong in inferred binding). The canonical loss has the correct one-sided
squared log-residual sign; bounded rows are not omitted from training loss. For
collected seed 43, failed >=100,000 nM predictions have median 30,660.99 nM. This is
not merely a rounding issue. The pass fraction alone omits violation magnitude.

MHC-II validation ligands comprise 694 positives and 3,470 generated negatives,
all from study pmid:35561310: 3,460 same-protein and 10 reference-proteome negatives.
475 positives and 2,375 negatives have two candidate MHCs, with the rest having one.
These unobserved comparator peptides are not experimentally verified nonbinders.
The ~0.79 AUROC is a genuine model limitation on this population. No matched
NetMHCIIpan score exists here, so a numeric cross-tool gap remains unmeasured.
The NetMHCIIpan-4.3 paper (https://pmc.ncbi.nlm.nih.gov/articles/PMC10672173/,
Fig. 2 and Methods) uses different cross-validation and negative constructions;
its development-model per-molecule median AUROCs are visually approximately
0.93-0.95. That is context, not a head-to-head comparison. Short optimization and
omitted source-protein flanks are possible contributors in our fits, not established
causes. Audit details: `results/metric-interpretation-audit.json`.

<!-- verified-assay-closeout -->

## Verified matched results

| Condition | Split | Class | Detector | Exact n | Spearman | RMSE log10(nM) |
|---|---|---|---|---:|---:|---:|
| direct-s43 | validation | I | radioactivity | 141 | 0.5001 | 1.2860 |
| direct-s43 | validation | I | fluorescence | 28 | 0.5835 | 1.2440 |
| direct-s43 | validation | II | radioactivity | 1129 | 0.4040 | 0.8757 |
| direct-s43 | validation | II | fluorescence | 41 | 0.8170 | 0.5175 |
| direct-s43 | test | I | radioactivity | 100 | 0.2990 | 1.0062 |
| direct-s43 | test | I | fluorescence | 157 | 0.3146 | 1.1589 |
| direct-s43 | test | II | radioactivity | 123 | 0.2285 | 0.7609 |
| direct-s43 | test | II | fluorescence | 634 | 0.0997 | 1.6655 |
| direct-s42 | validation | I | radioactivity | 141 | 0.4551 | 1.4393 |
| direct-s42 | validation | I | fluorescence | 28 | 0.5315 | 0.8277 |
| direct-s42 | validation | II | radioactivity | 1129 | 0.3534 | 0.9118 |
| direct-s42 | validation | II | fluorescence | 41 | 0.8450 | 0.5654 |
| direct-s42 | test | I | radioactivity | 100 | 0.2067 | 1.0212 |
| direct-s42 | test | I | fluorescence | 157 | 0.2344 | 1.0796 |
| direct-s42 | test | II | radioactivity | 123 | 0.1888 | 0.8897 |
| direct-s42 | test | II | fluorescence | 634 | 0.1256 | 1.5856 |

Both validation/test populations and serialized observations match.
Scipy independently reproduces IC50 correlations; scikit-learn independently
reproduces available ligand-background AUROC/AUPRC. Full endpoint/study and
ligand metrics, parameter coverage and actual hardware are in each results/
condition report; collection.json records every prediction dump and hash.
Winner/promotion remains pending scientific review and study-aware uncertainty.


## Stability and training-population audit (2026-10-30)

Both 4,000-update direct fits are finished. Stability shares the biological
encoder, but half-life, koff, td50 and tm are excluded from the new categorical
concentration decoder and use separate endpoint heads. Half-life is in minutes,
koff in 1/s, and td50 is a 50% dissociation temperature in degrees Celsius, not a
time or a Tm measurement. Tm is supported but has no admitted training rows.

The following exact-value metrics were independently recomputed from hash-verified
prediction dumps, including checking target transforms against raw values. These
are previously inspected development holdouts, not a fresh blind test.

| Seed | Split | Output | Class | Train n | Exact eval n | Spearman | RMSE (target space) |
|---|---|---|---|---:|---:|---:|---:|
| 42 | validation | half_life | I | 8,588 | 71 | 0.4139 | 1.4207 (log10(minutes)) |
| 42 | validation | td50 | I | 489 | 145 | 0.1960 | 6.2685 (degrees_Celsius) |
| 42 | test | half_life | I | 8,588 | 62 | 0.0648 | 1.3481 (log10(minutes)) |
| 42 | test | half_life | II | 135 | 148 | -0.0726 | 0.6664 (log10(minutes)) |
| 42 | test | td50 | I | 489 | 140 | 0.3163 | 5.5768 (degrees_Celsius) |
| 43 | validation | half_life | I | 8,588 | 71 | 0.4451 | 1.2158 (log10(minutes)) |
| 43 | validation | td50 | I | 489 | 145 | 0.4390 | 5.5851 (degrees_Celsius) |
| 43 | test | half_life | I | 8,588 | 62 | 0.3160 | 1.2088 (log10(minutes)) |
| 43 | test | half_life | II | 135 | 148 | -0.1806 | 0.8581 (log10(minutes)) |
| 43 | test | td50 | I | 489 | 140 | 0.3824 | 5.3164 (degrees_Celsius) |

MHC-II half-life has only 135 TRAIN observations, 148 TEST observations and no
validation examples. Its negative test correlations do not demonstrate useful
transfer. Koff has five class-I and two class-II TRAIN rows and no held-out
measurement; td50 has only seven class-II TRAIN rows and no class-II holdout.
These outputs are not scientifically qualified by this run.

Current class-II TRAIN comprises 306,303 observed ligands and 306,302 generated
unobserved comparisons, with the original/generated split established by the
independent full-cohort census and matching the unchanged class-II ligand count
in this family's preparation. It also has 61,143 concentration observations, of
which 35,512 are literal IC50. These are observations, not necessarily independent
peptide-MHC pairs. Original TRAIN background generation is approximately 1:1;
evaluation uses 5:1.

The NetMHCIIpan-4.3 publication reports 675,364 positive ligands, 6,886,973
generated negatives and 129,110 binding-affinity data points in its full development
corpus ([Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10672173/)). Our TRAIN-only
human/curation contract and its published full development corpus are not identical
populations. Both measured coverage and the negative sampling ratio contribute to
the apparent size gap; the negatives do not add quantitative affinity measurements.

Generated backgrounds retain binary elution labels. Random IC50 thresholds would
be synthetic modeling assumptions, not measured bounds. No affinity relabeling or
additional fit was performed in this audit. Full counts, metrics, formula and input
hashes are in `results/stability-and-data-audit.json`.


## Future-work handoff

The [post-fit handoff](../agents/codex/plans/2026-10-30_post-fit-handoff.md) records
limitations, priorities and issues #138–#142. Source stability qualification already
has an overlap/zero/construct audit and is tracked in Hitlist #875; the published
28,166 rows must not be described as wholly new measurements. No synthetic
affinity bounds or follow-up fits were introduced by publication.
